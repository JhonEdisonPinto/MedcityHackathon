# -*- coding: utf-8 -*-
"""
indexar.py — Indexa los datos de Medata en ChromaDB para el RAG de MedCity.

Ejecutar UNA SOLA VEZ antes de lanzar el dashboard:
    python indexar.py

Enriquece los documentos con datos originales (Database/Originales/):
  - Perfil emprendedor: estrato, cabeza de hogar, tipos de negocio, edad
  - Sedes WiFi detalladas por barrio
  - Segmentación de usuarios con métodos de autenticación

Genera documentos de texto enriquecido por barrio, por comuna, por
emprendedores sin WiFi, y contexto global.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Asegurar que el directorio raíz esté en el path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import upsert_rag_documents  # noqa: E402

# ── Rutas de datos ────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "dashboard" / "data"
DB_DIR = ROOT / "Database"
DB_ORIG = DB_DIR / "Originales"


# ── Utilidades ────────────────────────────────────────────────────────────────

def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _norm_barrio(s: str) -> str:
    return _strip_accents(s).lower().strip()


# ── Carga de CSVs ─────────────────────────────────────────────────────────────

def _load() -> dict[str, pd.DataFrame]:
    """Carga todos los datasets: procesados + originales."""
    data = {
        # Dashboard (procesados)
        "master":       pd.read_csv(DATA_DIR / "master_wifi_barrio.csv"),
        "perfil":       pd.read_csv(DATA_DIR / "perfil_consumidor_wifi_barrio.csv"),
        "flujo_barrio": pd.read_csv(DATA_DIR / "flujo_wifi_barrio.csv"),
        "densidad_b":   pd.read_csv(DATA_DIR / "densidad_emprendedora_barrio.csv"),
        "densidad_c":   pd.read_csv(DATA_DIR / "densidad_emprendedora_comuna.csv"),
        "flujo_comuna": pd.read_csv(DATA_DIR / "flujo_wifi_comuna.csv"),
        "edad_global":  pd.read_csv(DATA_DIR / "distribucion_edad_wifi.csv"),
        "flujo_sede":   pd.read_csv(DATA_DIR / "flujo_wifi_sede.csv"),
        # Originales (datos ricos)
        "artesanos_orig": pd.read_csv(DB_ORIG / "registro_artesano_y_producto_formado_y_cualificados_en_diseno.csv"),
        "wifi_puntos":    pd.read_csv(DB_DIR / "wifi_puntos_medellin_comuna.csv"),
        "segmentacion":   pd.read_csv(DB_DIR / "segmentacion_de_usuarios.csv"),
    }
    # sedeWifi_por_barrio solo en originales
    sede_barrio_path = DB_ORIG / "sedeWifi_por_barrio.csv"
    if sede_barrio_path.exists():
        data["sede_barrio"] = pd.read_csv(sede_barrio_path)
    return data


# ── Procesamiento de datos originales ─────────────────────────────────────────

def _procesar_artesanos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega perfil emprendedor por barrio desde datos originales.
    Recupera: estrato, cabeza_de_hogar, tipo_emprendimiento, edad, sexo.
    """
    df = df.copy()
    df["barrio_norm"] = df["barrio_vereda_ciudadano"].apply(_norm_barrio)
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["estrato"] = pd.to_numeric(df["estrato"], errors="coerce")

    # Agrupar por barrio
    grouped = df.groupby("barrio_norm").agg(
        n_emprendedores=("edad", "count"),
        edad_media=("edad", "mean"),
        edad_min=("edad", "min"),
        edad_max=("edad", "max"),
        estrato_medio=("estrato", "mean"),
        estrato_moda=("estrato", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan),
        pct_mujeres=("sexo", lambda x: round((x == "femenino").sum() / len(x) * 100, 1)),
        pct_cabeza_hogar=("cabeza_de_hogar", lambda x: round((x == "si").sum() / len(x) * 100, 1)),
        tipos_negocio=("tipo_de_emprendimiento", lambda x: ", ".join(x.value_counts().head(3).index.tolist())),
        tipo_dominante=("tipo_de_emprendimiento", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "sin_tipo"),
        ideas_negocio=("idea_de_negocio", lambda x: ", ".join(x.value_counts().head(3).index.tolist())),
        comuna=("comuna_ciudadano", "first"),
        zona=("zona_ciudadano", "first"),
    ).reset_index()

    grouped["edad_media"] = grouped["edad_media"].round(1)
    grouped["estrato_medio"] = grouped["estrato_medio"].round(1)

    # Diversificación: tipos únicos / total emprendedores
    tipos_por_barrio = df.groupby("barrio_norm")["tipo_de_emprendimiento"].nunique().reset_index()
    tipos_por_barrio.columns = ["barrio_norm", "n_tipos_unicos"]
    grouped = grouped.merge(tipos_por_barrio, on="barrio_norm", how="left")
    grouped["indice_diversificacion"] = (
        grouped["n_tipos_unicos"] / grouped["n_emprendedores"]
    ).round(3)

    return grouped


def _procesar_segmentacion_por_barrio(
    segmentacion: pd.DataFrame,
    wifi_puntos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega métodos de autenticación por barrio cruzando segmentación con sedes WiFi.
    """
    # Normalizar nombres de sede
    segmentacion = segmentacion.copy()
    wifi_puntos = wifi_puntos.copy()

    if "Sedes" in segmentacion.columns:
        segmentacion["sede_norm"] = segmentacion["Sedes"].apply(_norm_barrio)
    else:
        return pd.DataFrame()

    wifi_puntos["sede_norm"] = wifi_puntos["sede"].apply(_norm_barrio)
    wifi_puntos["barrio_norm"] = wifi_puntos["barrio"].apply(_norm_barrio)

    # Merge para asignar barrio a cada sede
    merged = segmentacion.merge(
        wifi_puntos[["sede_norm", "barrio_norm", "comuna"]],
        on="sede_norm", how="left"
    )
    merged = merged.dropna(subset=["barrio_norm"])

    # Columnas de autenticación (si existen)
    auth_cols = [c for c in merged.columns if "Autenticaci" in c or "autenticaci" in c]

    agg_dict = {}
    for col in auth_cols:
        agg_dict[col] = "sum"

    if not agg_dict:
        return pd.DataFrame()

    result = merged.groupby("barrio_norm").agg(agg_dict).reset_index()

    # Calcular porcentajes
    total_col = result[list(agg_dict.keys())].sum(axis=1)
    for col in agg_dict:
        safe_name = col.replace("Autenticación_", "pct_auth_")
        result[safe_name] = (result[col] / total_col * 100).round(1)

    return result


# ── Score de oportunidad ──────────────────────────────────────────────────────

def _calcular_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    score = 0.40 × vacio_oferta  +  0.30 × trafico_norm
          + 0.20 × diversificacion  +  0.10 × (1 - saturacion)
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    cols_num = ["total_registros", "n_emprendedores", "total_usuarios"]
    for c in cols_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["trafico_norm"] = scaler.fit_transform(df[["total_registros"]]) if "total_registros" in df.columns else 0
    df["emp_norm"] = scaler.fit_transform(df[["n_emprendedores"]]) if "n_emprendedores" in df.columns else 0

    if "total_usuarios" in df.columns and "total_registros" in df.columns:
        ratio = (df["total_usuarios"] / (df["total_registros"] + 1)).fillna(0)
        df["diversif_norm"] = scaler.fit_transform(ratio.values.reshape(-1, 1))
    else:
        df["diversif_norm"] = 0

    df["score_oportunidad"] = (
        0.40 * (1 - df["emp_norm"])
        + 0.30 * df["trafico_norm"]
        + 0.20 * df["diversif_norm"]
        + 0.10 * (1 - df["emp_norm"])
    ).round(4)

    med_trafico = df["trafico_norm"].median()
    med_emp = df["emp_norm"].median()

    def _cuadrante(row: pd.Series) -> str:
        alto_trafico = row["trafico_norm"] >= med_trafico
        mucho_emp = row["emp_norm"] >= med_emp
        if alto_trafico and not mucho_emp:
            return "OPORTUNIDAD"
        if alto_trafico and mucho_emp:
            return "CONSOLIDADA"
        if not alto_trafico and mucho_emp:
            return "SATURADA"
        return "EMERGENTE"

    df["cuadrante"] = df.apply(_cuadrante, axis=1)
    return df


# ── Construcción de documentos por barrio (ENRIQUECIDOS) ──────────────────────

def _docs_barrio(dfs: dict[str, pd.DataFrame]) -> tuple[list[str], list[dict], list[str]]:
    master = dfs["master"].copy()
    perfil = dfs["perfil"].copy()
    flujo  = dfs["flujo_barrio"].copy()
    densb  = dfs["densidad_b"].copy()

    # Procesar datos originales de emprendedores
    print("  → Procesando perfil emprendedor desde datos originales...")
    perfil_emp = _procesar_artesanos(dfs["artesanos_orig"])

    # Normalizar nombres de barrio
    for df in [master, perfil, flujo, densb]:
        if "barrio_norm" in df.columns:
            df["barrio_norm"] = df["barrio_norm"].apply(_norm_barrio)
        if "barrio" in df.columns:
            df["barrio"] = df["barrio"].apply(_norm_barrio)

    # Merge principal (21 barrios con WiFi)
    merged = master.merge(perfil, on="barrio_norm", how="left", suffixes=("", "_perfil"))
    merged = merged.merge(
        flujo[["barrio_norm", "total_registros", "media_registros_sede", "max_registros_sede"]],
        on="barrio_norm", how="left", suffixes=("", "_flujo")
    )
    merged = merged.merge(
        densb[["barrio", "n_emprendedores", "zona"]].rename(columns={"barrio": "barrio_norm"}),
        on="barrio_norm", how="left"
    )

    # Merge perfil emprendedor original
    merged = merged.merge(
        perfil_emp[[
            "barrio_norm", "edad_media", "estrato_medio", "estrato_moda",
            "pct_mujeres", "pct_cabeza_hogar", "tipos_negocio", "tipo_dominante",
            "ideas_negocio", "indice_diversificacion", "n_tipos_unicos",
        ]],
        on="barrio_norm", how="left"
    )

    if "total_registros_flujo" in merged.columns:
        merged["total_registros"] = merged["total_registros"].combine_first(merged["total_registros_flujo"])

    merged["n_emprendedores"] = merged["n_emprendedores"].fillna(0).astype(int)
    merged = _calcular_score(merged)

    documents, metadatas, ids = [], [], []

    for _, row in merged.iterrows():
        barrio = row["barrio_norm"]

        # Columnas demográficas de edad
        pct_18_28 = row.get("pct_De_18_a_28_Aanios", row.get("pct_De_18_a_28_Años", row.get("pct_Rango_edad_De_18_a_28_Años", 0)))
        pct_29_50 = row.get("pct_De_29_a_50_Aanios", row.get("pct_De_29_a_50_Años", row.get("pct_Rango_edad_De_29_a_50_Años", 0)))
        pct_14_17 = row.get("pct_De_14_a_17_Aanios", row.get("pct_De_14_a_17_Años", row.get("pct_Rango_edad_De_14_a_17_Años", 0)))
        pct_mayor50 = row.get("pct_Mayor_de_50_Aanios", row.get("pct_Mayor_de_50_Años", row.get("pct_Rango_edad_Mayor_de_50_Años", 0)))

        pct_smartphone = row.get("pct_Smartphone", row.get("pct_Dispositivo_conexión_Smartphone", 0))
        pct_pc = row.get("pct_Computadora_personal", row.get("pct_Dispositivo_conexión_Computadora_personal", 0))

        sexo_f = row.get("Sexo_Femenino", 0)
        sexo_m = row.get("Sexo_Masculino", 0)
        total_sexo = (sexo_f + sexo_m) or 1
        pct_femenino = round(sexo_f / total_sexo * 100, 1)
        pct_masculino = round(sexo_m / total_sexo * 100, 1)

        trafico = int(row.get("total_registros", 0) or 0)
        usuarios = int(row.get("total_usuarios_x", row.get("total_usuarios", 0)) or 0)
        n_emp = int(row.get("n_emprendedores", 0))
        n_sedes = int(row.get("n_sedes_x", row.get("n_sedes", 0)) or 0)
        score = float(row.get("score_oportunidad", 0))
        cuadrante = row.get("cuadrante", "EMERGENTE")
        zona = row.get("zona", "sin_zona")
        edad_dom = row.get("rango_edad_dominante_x", row.get("rango_edad_dominante", "De 18 a 28 Años"))
        ratio_sp = float(row.get("ratio_smartphone_pc_x", row.get("ratio_smartphone_pc", 0)) or 0)
        comuna_id = int(row.get("comuna_x", row.get("comuna", 0)) or 0)

        # Datos enriquecidos del emprendedor
        edad_media_emp = row.get("edad_media", None)
        estrato_medio = row.get("estrato_medio", None)
        estrato_moda = row.get("estrato_moda", None)
        pct_mujeres_emp = row.get("pct_mujeres", None)
        pct_cabeza_hogar = row.get("pct_cabeza_hogar", None)
        tipos_negocio = row.get("tipos_negocio", "")
        tipo_dominante = row.get("tipo_dominante", "")
        ideas_negocio = row.get("ideas_negocio", "")
        idx_diversif = row.get("indice_diversificacion", None)
        n_tipos = row.get("n_tipos_unicos", 0)

        usuarios_por_emp = round(usuarios / n_emp, 0) if n_emp > 0 else usuarios

        # Clasificación de oportunidad
        if cuadrante == "OPORTUNIDAD":
            oport_texto = (
                f"El barrio {barrio} tiene ALTA OPORTUNIDAD de negocio: "
                f"flujo de {trafico:,} conexiones WiFi pero solo {n_emp} emprendimientos activos, "
                f"lo que equivale a {int(usuarios_por_emp):,} usuarios por emprendimiento."
            )
        elif cuadrante == "CONSOLIDADA":
            oport_texto = (
                f"El barrio {barrio} es una zona CONSOLIDADA: "
                f"alto tráfico ({trafico:,} registros WiFi) con {n_emp} emprendimientos establecidos."
            )
        elif cuadrante == "SATURADA":
            oport_texto = (
                f"El barrio {barrio} muestra signos de SATURACIÓN: "
                f"bajo tráfico relativo pero {n_emp} emprendimientos compitiendo."
            )
        else:
            oport_texto = (
                f"El barrio {barrio} es una zona EMERGENTE: "
                f"bajo tráfico y pocos emprendimientos ({n_emp}), con potencial de crecimiento."
            )

        # Mismatch demográfico (Capa 3 del plan)
        mismatch_texto = ""
        if edad_media_emp and not pd.isna(edad_media_emp) and pct_18_28 > 50:
            if edad_media_emp > 40:
                mismatch_texto = (
                    f"\nMISMATCH DEMOGRÁFICO DETECTADO: Los consumidores WiFi son mayoritariamente "
                    f"jóvenes (18-28 años: {pct_18_28:.1f}%), pero la edad media del emprendedor "
                    f"es {edad_media_emp:.0f} años. Hay un gap generacional que representa "
                    f"oportunidad para negocios dirigidos a jóvenes."
                )

        # Saturación de sector (Capa 3)
        saturacion_texto = ""
        if idx_diversif is not None and not pd.isna(idx_diversif) and n_emp > 2:
            if idx_diversif < 0.3:
                saturacion_texto = (
                    f"\nSATURACIÓN DE SECTOR: Índice de diversificación bajo ({idx_diversif:.2f}). "
                    f"El {tipo_dominante} domina con la mayoría de los {n_emp} emprendimientos. "
                    f"Considerar rubros alternativos."
                )
            elif idx_diversif > 0.7:
                saturacion_texto = (
                    f"\nECOSISTEMA DIVERSIFICADO: Índice de diversificación alto ({idx_diversif:.2f}). "
                    f"{n_tipos} tipos de negocio diferentes entre {n_emp} emprendimientos."
                )

        rec_negocio = _recomendar_negocio(
            pct_18_28=pct_18_28, pct_29_50=pct_29_50,
            pct_smartphone=pct_smartphone, ratio_sp=ratio_sp,
            pct_femenino=pct_femenino, cuadrante=cuadrante
        )

        # ── Sección de perfil emprendedor (nueva) ──
        perfil_emp_texto = ""
        if edad_media_emp and not pd.isna(edad_media_emp):
            perfil_emp_texto = (
                f"\nPERFIL DEL EMPRENDEDOR (datos originales Medata):\n"
                f"- Edad media del emprendedor: {edad_media_emp:.0f} años\n"
                f"- Estrato socioeconómico medio: {estrato_medio if estrato_medio and not pd.isna(estrato_medio) else 'N/A'}\n"
                f"- Estrato más frecuente: {int(estrato_moda) if estrato_moda and not pd.isna(estrato_moda) else 'N/A'}\n"
                f"- Mujeres emprendedoras: {pct_mujeres_emp:.1f}%\n"
                f"- Cabezas de hogar: {pct_cabeza_hogar:.1f}%\n"
                f"- Tipos de negocio: {tipos_negocio}\n"
                f"- Tipo dominante: {tipo_dominante}\n"
                f"- Ideas de negocio: {ideas_negocio}\n"
                f"- Índice de diversificación: {idx_diversif:.2f} ({n_tipos} tipos únicos)\n"
            )

        doc = (
            f"BARRIO: {barrio.title()} | COMUNA: {comuna_id} | ZONA: {zona}\n"
            f"SCORE DE OPORTUNIDAD: {score:.2f}/1.00 | CUADRANTE: {cuadrante}\n\n"
            f"TRÁFICO Y DEMANDA:\n"
            f"- Conexiones WiFi públicas totales: {trafico:,}\n"
            f"- Usuarios únicos estimados: {usuarios:,}\n"
            f"- Puntos WiFi (sedes): {n_sedes}\n"
            f"- Emprendimientos actuales: {n_emp}\n"
            f"- Usuarios por emprendimiento: {int(usuarios_por_emp):,}\n\n"
            f"PERFIL DEL CONSUMIDOR WiFi:\n"
            f"- Edad dominante: {edad_dom}\n"
            f"- Jóvenes 18-28 años: {pct_18_28:.1f}%\n"
            f"- Adultos 29-50 años: {pct_29_50:.1f}%\n"
            f"- Adolescentes 14-17 años: {pct_14_17:.1f}%\n"
            f"- Mayores de 50 años: {pct_mayor50:.1f}%\n"
            f"- Género consumidor: {pct_femenino:.1f}% femenino, {pct_masculino:.1f}% masculino\n"
            f"- Dispositivo principal: Smartphone ({pct_smartphone:.1f}%)\n"
            f"- Computadora personal: {pct_pc:.1f}%\n"
            f"- Ratio smartphone/PC: {ratio_sp:.1f}x (cultura mobile-first)\n"
            f"{perfil_emp_texto}\n"
            f"ANÁLISIS DE OPORTUNIDAD:\n"
            f"{oport_texto}{mismatch_texto}{saturacion_texto}\n\n"
            f"NEGOCIOS RECOMENDADOS PARA ESTE BARRIO:\n"
            f"{rec_negocio}\n\n"
            f"FUENTE: Medata Medellín — WiFi público + registro artesanos y emprendedores"
        )

        meta = {
            "zone": barrio,
            "comuna": comuna_id,
            "zona_geografica": str(zona),
            "cuadrante": cuadrante,
            "score_oportunidad": score,
            "trafico_wifi": trafico,
            "total_usuarios": usuarios,
            "emprendedores": n_emp,
            "usuarios_por_emprendimiento": int(usuarios_por_emp),
            "pct_18_28": float(pct_18_28),
            "pct_29_50": float(pct_29_50),
            "pct_smartphone": float(pct_smartphone),
            "pct_femenino": float(pct_femenino),
            "tipo_documento": "barrio",
            "estrato_medio": float(estrato_medio) if estrato_medio and not pd.isna(estrato_medio) else 0,
            "tiene_perfil_emprendedor": bool(edad_media_emp and not pd.isna(edad_media_emp)),
        }

        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"barrio_{barrio.replace(' ', '_')}")

    return documents, metadatas, ids


# ── Documentos de barrios CON emprendedores pero SIN WiFi ─────────────────────

def _docs_barrio_sin_wifi(dfs: dict[str, pd.DataFrame]) -> tuple[list[str], list[dict], list[str]]:
    """
    Barrios con emprendedores registrados pero sin cobertura WiFi medida.
    Importante para completar el mapa de oportunidades (Capa 3).
    """
    perfil_emp = _procesar_artesanos(dfs["artesanos_orig"])
    master = dfs["master"].copy()
    master["barrio_norm"] = master["barrio_norm"].apply(_norm_barrio)

    barrios_wifi = set(master["barrio_norm"].unique())
    sin_wifi = perfil_emp[~perfil_emp["barrio_norm"].isin(barrios_wifi)]
    sin_wifi = sin_wifi[sin_wifi["barrio_norm"] != "otro"]  # Excluir categoría genérica

    documents, metadatas, ids = [], [], []

    for _, row in sin_wifi.iterrows():
        barrio = row["barrio_norm"]
        n_emp = int(row["n_emprendedores"])
        if n_emp < 2:
            continue  # Solo barrios con al menos 2 emprendedores

        edad_media = row.get("edad_media", 0)
        estrato_medio = row.get("estrato_medio", 0)
        estrato_moda = row.get("estrato_moda", 0)
        pct_mujeres = row.get("pct_mujeres", 0)
        pct_cabeza = row.get("pct_cabeza_hogar", 0)
        tipos = row.get("tipos_negocio", "")
        tipo_dom = row.get("tipo_dominante", "")
        ideas = row.get("ideas_negocio", "")
        idx_div = row.get("indice_diversificacion", 0)
        n_tipos = int(row.get("n_tipos_unicos", 0))
        comuna_raw = str(row.get("comuna", "0"))
        # Manejar formato "03-manrique" extrayendo solo el número
        m = __import__("re").match(r"(\d+)", comuna_raw)
        comuna = int(m.group(1)) if m else 0
        zona = row.get("zona", "sin_zona")

        doc = (
            f"BARRIO: {barrio.title()} | COMUNA: {comuna} | ZONA: {zona}\n"
            f"NOTA: Este barrio NO tiene cobertura WiFi pública medida por Medata.\n"
            f"Los datos provienen exclusivamente del registro de emprendedores.\n\n"
            f"PERFIL EMPRENDEDOR:\n"
            f"- Emprendedores registrados: {n_emp}\n"
            f"- Edad media: {edad_media:.0f} años\n"
            f"- Estrato socioeconómico medio: {estrato_medio:.1f}\n"
            f"- Estrato más frecuente: {int(estrato_moda) if not pd.isna(estrato_moda) else 'N/A'}\n"
            f"- Mujeres emprendedoras: {pct_mujeres:.1f}%\n"
            f"- Cabezas de hogar: {pct_cabeza:.1f}%\n"
            f"- Tipos de negocio: {tipos}\n"
            f"- Tipo dominante: {tipo_dom}\n"
            f"- Ideas de negocio: {ideas}\n"
            f"- Índice de diversificación: {idx_div:.2f} ({n_tipos} tipos)\n\n"
            f"ANÁLISIS:\n"
            f"Sin datos de tráfico WiFi no se puede calcular vacío de oferta. "
            f"Sin embargo, con {n_emp} emprendedores activos en estrato {estrato_medio:.1f}, "
            f"el barrio muestra actividad económica con {'alta diversificación' if idx_div > 0.5 else 'concentración en ' + tipo_dom}.\n\n"
            f"FUENTE: Medata Medellín — Registro artesanos y emprendedores"
        )

        meta = {
            "zone": barrio,
            "comuna": comuna,
            "zona_geografica": str(zona),
            "tipo_documento": "barrio_sin_wifi",
            "emprendedores": n_emp,
            "estrato_medio": float(estrato_medio) if not pd.isna(estrato_medio) else 0,
            "tiene_wifi": False,
        }

        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"barrio_sin_wifi_{barrio.replace(' ', '_')}")

    return documents, metadatas, ids


def _recomendar_negocio(
    pct_18_28: float, pct_29_50: float,
    pct_smartphone: float, ratio_sp: float,
    pct_femenino: float, cuadrante: str
) -> str:
    recomendaciones = []

    if pct_18_28 > 50:
        recomendaciones.append(
            "- Cafeterías con WiFi y espacios de co-working (alta demanda de jóvenes digitales)"
        )
        recomendaciones.append(
            "- Tiendas de tecnología y accesorios móviles (perfil digital 18-28)"
        )
        recomendaciones.append(
            "- Servicios de educación y capacitación (cursos, bootcamps)"
        )
    if pct_29_50 > 35:
        recomendaciones.append(
            "- Restaurantes y servicios de alimentación de calidad (adultos con capacidad de compra)"
        )
        recomendaciones.append(
            "- Servicios financieros y seguros (adultos 29-50 con mayor estabilidad económica)"
        )
    if ratio_sp > 80:
        recomendaciones.append(
            "- Negocios con modelo mobile-first: apps, pagos QR, delivery, reservas digitales"
        )
    if pct_femenino > 48:
        recomendaciones.append(
            "- Servicios de belleza, bienestar y moda (alta proporción de usuarias)"
        )
    if cuadrante == "OPORTUNIDAD":
        recomendaciones.append(
            "- PRIORIDAD ALTA: cualquier negocio de servicio con bajo costo de entrada "
            "(bajo tráfico de competencia + alto tráfico de clientes potenciales)"
        )
    elif cuadrante == "EMERGENTE":
        recomendaciones.append(
            "- Negocios de bajo riesgo: servicios básicos de primera necesidad, "
            "tiendas de barrio mejoradas, puntos de servicios digitales"
        )

    return "\n".join(recomendaciones) if recomendaciones else "- Sin recomendación específica con los datos disponibles"


# ── Documentos por comuna (ENRIQUECIDOS) ─────────────────────────────────────

def _docs_comuna(dfs: dict[str, pd.DataFrame]) -> tuple[list[str], list[dict], list[str]]:
    densidad_c = dfs["densidad_c"].copy()
    flujo_c    = dfs["flujo_comuna"].copy()

    # Perfil emprendedor agregado a nivel comuna
    art = dfs["artesanos_orig"].copy()
    art["comuna_ciudadano"] = pd.to_numeric(art["comuna_ciudadano"], errors="coerce")
    art["edad"] = pd.to_numeric(art["edad"], errors="coerce")
    art["estrato"] = pd.to_numeric(art["estrato"], errors="coerce")

    emp_comuna = art.groupby("comuna_ciudadano").agg(
        edad_media_emp=("edad", "mean"),
        estrato_medio_emp=("estrato", "mean"),
        pct_mujeres_emp=("sexo", lambda x: round((x == "femenino").sum() / len(x) * 100, 1)),
        pct_cabeza_hogar=("cabeza_de_hogar", lambda x: round((x == "si").sum() / len(x) * 100, 1)),
        tipo_dominante=("tipo_de_emprendimiento", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "sin_tipo"),
        n_tipos_unicos=("tipo_de_emprendimiento", "nunique"),
    ).reset_index()
    emp_comuna.rename(columns={"comuna_ciudadano": "comuna"}, inplace=True)

    flujo_c["comuna"] = pd.to_numeric(flujo_c["comuna"], errors="coerce")
    densidad_c["comuna"] = pd.to_numeric(densidad_c["comuna"], errors="coerce")

    merged = densidad_c.merge(
        flujo_c[["comuna", "n_sedes", "total_registros", "media_mensual_sede"]],
        on="comuna", how="left"
    )
    merged = merged.merge(emp_comuna, on="comuna", how="left")
    merged["total_registros"] = merged["total_registros"].fillna(0)
    merged["n_sedes"] = merged["n_sedes"].fillna(0).astype(int)

    documents, metadatas, ids = [], [], []

    for _, row in merged.iterrows():
        nombre = str(row.get("nombre_comuna", f"Comuna {int(row['comuna'])}"))
        comuna_id = int(row["comuna"])
        n_emp = int(row.get("n_emprendedores", 0))
        poblacion = int(row.get("poblacion_estimada", 0))
        densidad = float(row.get("densidad_x1000_hab", 0))
        trafico = int(row.get("total_registros", 0))
        n_sedes = int(row.get("n_sedes", 0))
        media_sede = float(row.get("media_mensual_sede", 0))

        # Datos emprendedor enriquecidos
        edad_media_emp = row.get("edad_media_emp", None)
        estrato_medio_emp = row.get("estrato_medio_emp", None)
        pct_mujeres = row.get("pct_mujeres_emp", None)
        pct_cabeza = row.get("pct_cabeza_hogar", None)
        tipo_dom = row.get("tipo_dominante", "")
        n_tipos = int(row.get("n_tipos_unicos", 0))

        if densidad < 0.05:
            nivel_oport = "MUY ALTA"
        elif densidad < 0.12:
            nivel_oport = "ALTA"
        elif densidad < 0.20:
            nivel_oport = "MEDIA"
        else:
            nivel_oport = "BAJA (zona consolidada)"

        # Sección emprendedor
        perfil_emp_sec = ""
        if edad_media_emp and not pd.isna(edad_media_emp):
            perfil_emp_sec = (
                f"\nPERFIL DEL EMPRENDEDOR TÍPICO:\n"
                f"- Edad media: {edad_media_emp:.0f} años\n"
                f"- Estrato socioeconómico medio: {estrato_medio_emp:.1f}\n"
                f"- Mujeres emprendedoras: {pct_mujeres:.1f}%\n"
                f"- Cabezas de hogar: {pct_cabeza:.1f}%\n"
                f"- Tipo de emprendimiento dominante: {tipo_dom}\n"
                f"- Diversidad de rubros: {n_tipos} tipos únicos\n"
            )

        doc = (
            f"COMUNA: {nombre} (ID: {comuna_id})\n"
            f"NIVEL DE OPORTUNIDAD EMPRENDEDORA: {nivel_oport}\n\n"
            f"DENSIDAD EMPRENDEDORA:\n"
            f"- Emprendimientos registrados: {n_emp}\n"
            f"- Población estimada: {poblacion:,} habitantes\n"
            f"- Emprendedores por cada 1,000 habitantes: {densidad:.4f}\n\n"
            f"INFRAESTRUCTURA WiFi:\n"
            f"- Sedes WiFi públicas: {n_sedes}\n"
            f"- Total conexiones registradas: {trafico:,}\n"
            f"- Promedio mensual por sede: {media_sede:,.0f} conexiones\n"
            f"{perfil_emp_sec}\n"
            f"INTERPRETACIÓN:\n"
            f"La comuna {nombre} tiene una densidad de {densidad:.4f} emprendedores "
            f"por 1,000 habitantes sobre una población de {poblacion:,}. "
            f"{'Con bajo nivel de emprendimiento relativo, representa una zona de alta oportunidad.' if nivel_oport in ('MUY ALTA','ALTA') else 'La zona presenta un nivel competitivo moderado-alto.'}"
            f"\n\nFUENTE: Medata Medellín — Registro emprendedores + WiFi público"
        )

        meta = {
            "zone": nombre,
            "comuna": comuna_id,
            "tipo_documento": "comuna",
            "n_emprendedores": n_emp,
            "poblacion": poblacion,
            "densidad_x1000_hab": densidad,
            "trafico_wifi": trafico,
            "n_sedes_wifi": n_sedes,
            "nivel_oportunidad": nivel_oport,
            "estrato_medio": float(estrato_medio_emp) if estrato_medio_emp and not pd.isna(estrato_medio_emp) else 0,
        }

        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"comuna_{comuna_id}_{nombre.lower().replace(' ', '_')}")

    return documents, metadatas, ids


# ── Documento de contexto global (ENRIQUECIDO) ───────────────────────────────

def _doc_global(dfs: dict[str, pd.DataFrame]) -> tuple[list[str], list[dict], list[str]]:
    edad_g = dfs["edad_global"]
    total = int(edad_g["total_usuarios"].sum())

    filas = []
    for _, row in edad_g.iterrows():
        filas.append(f"  - {row['rango_edad']}: {row['pct']:.2f}% ({int(row['total_usuarios']):,} usuarios)")

    # Estadísticas globales de emprendedores desde datos originales
    art = dfs["artesanos_orig"].copy()
    art["edad"] = pd.to_numeric(art["edad"], errors="coerce")
    art["estrato"] = pd.to_numeric(art["estrato"], errors="coerce")
    total_emp = len(art)
    edad_media_g = art["edad"].mean()
    estrato_medio_g = art["estrato"].mean()
    pct_mujeres_g = round((art["sexo"] == "femenino").sum() / total_emp * 100, 1)
    pct_cabeza_g = round((art["cabeza_de_hogar"] == "si").sum() / total_emp * 100, 1)
    tipo_dom_g = art["tipo_de_emprendimiento"].mode().iloc[0]
    n_tipos_g = art["tipo_de_emprendimiento"].nunique()
    n_barrios_emp = art["barrio_vereda_ciudadano"].nunique()

    # Distribución de estrato
    estrato_dist = art["estrato"].value_counts().sort_index()
    estrato_lines = []
    for est, count in estrato_dist.items():
        pct = round(count / total_emp * 100, 1)
        estrato_lines.append(f"  - Estrato {int(est)}: {pct}% ({count} emprendedores)")

    doc = (
        f"CONTEXTO GENERAL — MEDELLÍN SMART CITY (Medata)\n\n"
        f"PERFIL DEMOGRÁFICO DE USUARIOS WiFi PÚBLICOS:\n"
        f"Total de usuarios registrados: {total:,}\n"
        + "\n".join(filas) + "\n\n"
        f"PERFIL GLOBAL DEL EMPRENDEDOR (Registro artesanos Medata):\n"
        f"Total emprendedores registrados: {total_emp}\n"
        f"Distribuidos en {n_barrios_emp} barrios y corregimientos\n"
        f"- Edad media: {edad_media_g:.0f} años\n"
        f"- Estrato socioeconómico medio: {estrato_medio_g:.1f}\n"
        f"- Mujeres emprendedoras: {pct_mujeres_g}%\n"
        f"- Cabezas de hogar: {pct_cabeza_g}%\n"
        f"- Tipo de emprendimiento dominante: {tipo_dom_g}\n"
        f"- Tipos únicos de emprendimiento: {n_tipos_g}\n\n"
        f"DISTRIBUCIÓN POR ESTRATO SOCIOECONÓMICO:\n"
        + "\n".join(estrato_lines) + "\n\n"
        f"INTERPRETACIÓN:\n"
        f"El 53.6% de los usuarios del WiFi público de Medellín son jóvenes de 18 a 28 años, "
        f"seguidos por adultos de 29 a 50 años (30.6%). "
        f"El 100% de los barrios registra uso dominante de smartphone (>92%), "
        f"lo que indica que cualquier modelo de negocio debe ser mobile-first. "
        f"La ciudad tiene 21 barrios con cobertura WiFi medida por Medata y {n_barrios_emp} barrios con "
        f"registro de emprendedores, distribuidos en 16 comunas. "
        f"El emprendedor típico tiene {edad_media_g:.0f} años, es de estrato {estrato_medio_g:.1f}, "
        f"y el {pct_cabeza_g}% son cabezas de hogar.\n\n"
        f"FUENTE: Medata Medellín — distribución WiFi + registro artesanos y emprendedores"
    )

    return [doc], [{"tipo_documento": "global", "zone": "medellin"}], ["global_medellin"]


# ── Ejecución principal ───────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("MedCity — Indexación de datos para RAG")
    print("=" * 60)

    print("\n[1/6] Cargando datos de Medata (procesados + originales)...")
    dfs = _load()
    print(f"  → {len(dfs)} datasets cargados")

    print("\n[2/6] Construyendo documentos por barrio (con WiFi)...")
    docs_b, metas_b, ids_b = _docs_barrio(dfs)
    print(f"  → {len(docs_b)} barrios con WiFi")

    print("\n[3/6] Construyendo documentos de barrios sin WiFi...")
    docs_sw, metas_sw, ids_sw = _docs_barrio_sin_wifi(dfs)
    print(f"  → {len(docs_sw)} barrios con emprendedores pero sin WiFi")

    print("\n[4/6] Construyendo documentos por comuna...")
    docs_c, metas_c, ids_c = _docs_comuna(dfs)
    print(f"  → {len(docs_c)} comunas")

    print("\n[5/6] Construyendo documento de contexto global...")
    docs_g, metas_g, ids_g = _doc_global(dfs)

    all_docs  = docs_b + docs_sw + docs_c + docs_g
    all_metas = metas_b + metas_sw + metas_c + metas_g
    all_ids   = ids_b + ids_sw + ids_c + ids_g

    # Pinecone requiere IDs ASCII
    all_ids = [_strip_accents(i).replace(" ", "_") for i in all_ids]

    print(f"\n[6/6] Indexando {len(all_docs)} documentos en Pinecone...")
    n = upsert_rag_documents(
        documents=all_docs,
        metadatas=all_metas,
        ids=all_ids,
    )
    print(f"\n{'=' * 60}")
    print(f"[OK] {n} documentos indexados correctamente:")
    print(f"  - {len(docs_b)} barrios con WiFi")
    print(f"  - {len(docs_sw)} barrios sin WiFi (solo emprendedores)")
    print(f"  - {len(docs_c)} comunas")
    print(f"  - {len(docs_g)} documento global")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
