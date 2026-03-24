import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Setup paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# Resolver case-sensitivity (Windows: dashboard, Linux/Git: Dashboard)
_data_lower = ROOT / "dashboard" / "data"
_data_upper = ROOT / "Dashboard" / "data"
DATA = _data_lower if _data_lower.exists() else _data_upper

_db_orig = ROOT / "Database" / "Originales"
DB_ORIG = _db_orig

sys.path.insert(0, str(ROOT))

# ── Inyectar secrets de Streamlit Cloud → env vars ────────────────────────────
_SECRET_KEYS = [
    "PINECONE_API_KEY", "PINECONE_INDEX",
    "GROQ_API_KEY", "GROQ_MODEL", "GROQ_TEMPERATURE",
    "GROQ_MAX_TOKENS", "GROQ_REQUEST_TIMEOUT", "GROQ_MAX_RETRIES",
    "RAG_EMBEDDING_MODEL", "RAG_TOP_K",
]
for key in _SECRET_KEYS:
    if key not in os.environ:
        try:
            os.environ[key] = str(st.secrets[key])
        except (KeyError, FileNotFoundError):
            pass

from src.rag import run_medcity_graph  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ConectEMP — Medellín Emprendedora",
    page_icon="🏙️",
    layout="wide",
)

_LOGO = Path(__file__).parent / "assets" / "logo.png"

# ── Carga de datos (cacheada) ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    densidad_b = pd.read_csv(DATA / "densidad_emprendedora_barrio.csv")
    densidad_c = pd.read_csv(DATA / "densidad_emprendedora_comuna.csv")
    edad_g = pd.read_csv(DATA / "distribucion_edad_wifi.csv")
    flujo_b = pd.read_csv(DATA / "flujo_wifi_barrio.csv")
    flujo_c = pd.read_csv(DATA / "flujo_wifi_comuna.csv")
    master = pd.read_csv(DATA / "master_wifi_barrio.csv")
    artesanos = pd.read_csv(DB_ORIG / "registro_artesano_y_producto_formado_y_cualificados_en_diseno.csv")
    creditos = pd.read_csv(DB_ORIG / "creditos_otorgados_a_microempresarios.csv")
    creditos["monto"] = pd.to_numeric(creditos["monto"], errors="coerce")
    creditos["edad"] = pd.to_numeric(creditos["edad"], errors="coerce")
    creditos["comuna"] = pd.to_numeric(creditos["comuna"], errors="coerce")
    return densidad_b, densidad_c, edad_g, flujo_b, flujo_c, master, artesanos, creditos


densidad_b, densidad_c, edad_g, flujo_b, flujo_c, master, artesanos, creditos = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
_logo_col, _title_col = st.columns([1, 4])
with _logo_col:
    st.image(str(_LOGO), width=200)
with _title_col:
    st.title("ConectEMP")
    st.caption("Oportunidades de emprendimiento en Medellín — Datos Medata")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_chat = st.tabs(["📊 Dashboard", "💬 Asistente IA"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    # ── Preparar datos combinados ─────────────────────────────────────────────
    # Normalizar barrio de créditos para merge
    _cred_barrio_norm = (
        creditos["barrio"]
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[.#]", "", regex=True)
    )
    creditos_norm = creditos.assign(barrio_norm=_cred_barrio_norm)

    # Contar beneficiarios de créditos por barrio
    cred_por_barrio = (
        creditos_norm.groupby("barrio_norm")
        .agg(n_creditados=("monto", "count"))
        .reset_index()
        .rename(columns={"barrio_norm": "barrio"})
    )

    # Contar beneficiarios de créditos por comuna
    cred_por_comuna = (
        creditos.dropna(subset=["comuna"])
        .groupby("comuna")
        .agg(n_creditados=("monto", "count"), monto_total=("monto", "sum"))
        .reset_index()
    )

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_emp = int(densidad_b["n_emprendedores"].sum())
    total_barrios = len(densidad_b)
    total_usuarios = int(edad_g["total_usuarios"].sum())
    total_comunas = len(densidad_c)
    total_creditos = len(creditos)
    monto_total_cred = creditos["monto"].sum()
    total_combinado = total_emp + total_creditos

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total emprendedores", f"{total_combinado:,}")
    k2.metric("Artesanos registrados", f"{total_emp:,}")
    k3.metric("Microempresarios (créditos)", f"{total_creditos:,}")
    k4.metric("Usuarios WiFi", f"{total_usuarios:,}")
    k5.metric("Barrios con datos", total_barrios)
    k6.metric("Monto financiado", f"${monto_total_cred:,.0f}")

    st.divider()

    # ── Selector de vista ─────────────────────────────────────────────────────
    _VISTAS = [
        "🔍 Todos",
        "📈 Panorama General",
        "👥 Perfil Emprendedor",
        "📡 Conectividad WiFi",
        "💳 Detalle de Créditos",
    ]
    vista = st.selectbox("Selecciona el dashboard", _VISTAS, label_visibility="collapsed")

    # ── Fila 1: Barras + Dona ─────────────────────────────────────────────────
    if vista in ("🔍 Todos", "📈 Panorama General"):
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Top 15 barrios por emprendedores")
            top_combined = densidad_b[["barrio", "n_emprendedores", "zona"]].merge(
                cred_por_barrio, on="barrio", how="outer",
            )
            top_combined["n_emprendedores"] = top_combined["n_emprendedores"].fillna(0).astype(int)
            top_combined["n_creditados"] = top_combined["n_creditados"].fillna(0).astype(int)
            top_combined["total"] = top_combined["n_emprendedores"] + top_combined["n_creditados"]
            top_combined["zona"] = top_combined["zona"].fillna("sin dato")
            top15 = top_combined.nlargest(15, "total").sort_values("total")

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=top15["barrio"], x=top15["n_emprendedores"],
                name="Artesanos", orientation="h",
                marker_color="#2ecc71", text=top15["n_emprendedores"],
            ))
            fig_bar.add_trace(go.Bar(
                y=top15["barrio"], x=top15["n_creditados"],
                name="Microemp. (créditos)", orientation="h",
                marker_color="#e67e22", text=top15["n_creditados"],
            ))
            fig_bar.update_layout(
                barmode="stack",
                height=450,
                yaxis_title="",
                xaxis_title="Emprendedores",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig_bar.update_traces(textposition="inside")
            st.plotly_chart(fig_bar, width="stretch")

        with col2:
            st.subheader("Distribución de edad WiFi")
            fig_dona = px.pie(
                edad_g,
                values="total_usuarios",
                names="rango_edad",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_dona.update_layout(
                height=450,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            fig_dona.update_traces(textinfo="percent+label", textposition="outside")
            st.plotly_chart(fig_dona, width="stretch")

        st.divider()

    # ── Fila 2: Comunas + Perfil emprendedor ──────────────────────────────────
    if vista in ("🔍 Todos", "📈 Panorama General"):
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Emprendedores por comuna (combinado)")
            dc = densidad_c.merge(
                cred_por_comuna[["comuna", "n_creditados"]],
                on="comuna", how="left",
            )
            dc["n_creditados"] = dc["n_creditados"].fillna(0).astype(int)
            dc["total_emprendedores"] = dc["n_emprendedores"] + dc["n_creditados"]
            dc = dc.sort_values("total_emprendedores", ascending=False)

            fig_dens = go.Figure()
            fig_dens.add_trace(go.Bar(
                x=dc["nombre_comuna"], y=dc["n_emprendedores"],
                name="Artesanos", marker_color="#2ecc71",
                text=dc["n_emprendedores"],
            ))
            fig_dens.add_trace(go.Bar(
                x=dc["nombre_comuna"], y=dc["n_creditados"],
                name="Microemp. (créditos)", marker_color="#e67e22",
                text=dc["n_creditados"],
            ))
            fig_dens.update_layout(
                barmode="stack",
                height=400,
                xaxis_title="",
                yaxis_title="Emprendedores",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig_dens.update_xaxes(tickangle=45)
            fig_dens.update_traces(textposition="inside")
            st.plotly_chart(fig_dens, width="stretch")

        with col4:
            st.subheader("Género de emprendedores (combinado)")
            sexo_art = artesanos["sexo"].replace({"femenino": "Femenino", "masculino": "Masculino"})
            sexo_cred = creditos["sexo"].replace({"mujer": "Femenino", "hombre": "Masculino", "corporacion": "Corporación"})
            sexo_all = pd.concat([sexo_art, sexo_cred]).value_counts().reset_index()
            sexo_all.columns = ["Género", "Cantidad"]
            fig_sexo = px.pie(
                sexo_all,
                values="Cantidad",
                names="Género",
                hole=0.5,
                color_discrete_map={"Femenino": "#ff6b81", "Masculino": "#4ecdc4", "Corporación": "#f9ca24"},
            )
            fig_sexo.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_sexo.update_traces(textinfo="percent+value")
            st.plotly_chart(fig_sexo, width="stretch")

        st.divider()

    # ── Fila 3: Tipo emprendimiento + Estrato ─────────────────────────────────
    if vista in ("🔍 Todos", "👥 Perfil Emprendedor"):
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Tipos de emprendimiento (combinado)")
            tipos_art = artesanos["tipo_de_emprendimiento"].value_counts().reset_index()
            tipos_art.columns = ["Tipo", "Cantidad"]
            tipos_art["Fuente"] = "Artesanos"
            tipos_cred = creditos["actividad"].value_counts().reset_index()
            tipos_cred.columns = ["Tipo", "Cantidad"]
            tipos_cred["Fuente"] = "Microemp. (créditos)"
            tipos_all = pd.concat([tipos_art, tipos_cred], ignore_index=True)
            fig_tipo = px.bar(
                tipos_all,
                x="Cantidad",
                y="Tipo",
                orientation="h",
                color="Fuente",
                text="Cantidad",
                color_discrete_map={"Artesanos": "#2ecc71", "Microemp. (créditos)": "#e67e22"},
                barmode="group",
            )
            fig_tipo.update_layout(
                height=400,
                yaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig_tipo.update_traces(textposition="outside")
            st.plotly_chart(fig_tipo, width="stretch")

        with col6:
            st.subheader("Distribución por estrato socioeconómico")
            artesanos["estrato"] = pd.to_numeric(artesanos["estrato"], errors="coerce")
            estrato_counts = (
                artesanos["estrato"]
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .reset_index()
            )
            estrato_counts.columns = ["Estrato", "Cantidad"]
            fig_estrato = px.bar(
                estrato_counts,
                x="Estrato",
                y="Cantidad",
                color="Estrato",
                color_continuous_scale="Blues",
                text="Cantidad",
            )
            fig_estrato.update_layout(
                height=400,
                xaxis_title="Estrato socioeconómico",
                yaxis_title="Emprendedores",
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_estrato.update_traces(textposition="outside")
            st.plotly_chart(fig_estrato, width="stretch")

        st.divider()

    # ── Fila 4: Tráfico WiFi vs Emprendedores (scatter) ──────────────────────
    if vista in ("🔍 Todos", "📡 Conectividad WiFi"):
        st.subheader("Tráfico WiFi vs Emprendedores por barrio")
        st.caption("Tamaño = usuarios WiFi | Color = zona geográfica — Incluye artesanos + microempresarios creditados")

        scatter_df = flujo_b.merge(
            densidad_b.rename(columns={"barrio": "barrio_norm"}),
            on="barrio_norm",
            how="inner",
        )
        scatter_df = scatter_df.merge(
            cred_por_barrio.rename(columns={"barrio": "barrio_norm"}),
            on="barrio_norm",
            how="left",
        )
        scatter_df["n_creditados"] = scatter_df["n_creditados"].fillna(0).astype(int)
        scatter_df["total_emprendedores"] = scatter_df["n_emprendedores"] + scatter_df["n_creditados"]

        scatter_df = scatter_df.merge(
            master[["barrio_norm", "total_usuarios"]],
            on="barrio_norm",
            how="left",
        )
        scatter_df["total_usuarios"] = scatter_df["total_usuarios"].fillna(1000)

        fig_scatter = px.scatter(
            scatter_df,
            x="total_emprendedores",
            y="total_registros",
            size="total_usuarios",
            color="zona",
            hover_name="barrio_norm",
            text="barrio_norm",
            size_max=50,
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data={"n_emprendedores": True, "n_creditados": True},
        )
        fig_scatter.update_layout(
            height=500,
            xaxis_title="Emprendedores (artesanos + creditados)",
            yaxis_title="Conexiones WiFi totales",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_scatter.update_traces(textposition="top center", textfont_size=9)
        st.plotly_chart(fig_scatter, width="stretch")

        st.divider()

    # ── Fila 5: Cabeza de hogar + Zona ────────────────────────────────────────
    if vista in ("🔍 Todos", "👥 Perfil Emprendedor"):
        col7, col8 = st.columns(2)

        with col7:
            st.subheader("Emprendedores cabeza de hogar")
            cabeza_counts = artesanos["cabeza_de_hogar"].value_counts().reset_index()
            cabeza_counts.columns = ["Cabeza de hogar", "Cantidad"]
            fig_cabeza = px.pie(
                cabeza_counts,
                values="Cantidad",
                names="Cabeza de hogar",
                hole=0.5,
                color_discrete_map={"si": "#2ecc71", "no": "#95a5a6"},
            )
            fig_cabeza.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_cabeza.update_traces(textinfo="percent+value")
            st.plotly_chart(fig_cabeza, width="stretch")

        with col8:
            st.subheader("Emprendedores por zona geográfica")
            zona_counts = (
                artesanos["zona_ciudadano"]
                .value_counts()
                .reset_index()
            )
            zona_counts.columns = ["Zona", "Cantidad"]
            fig_zona = px.bar(
                zona_counts,
                x="Zona",
                y="Cantidad",
                color="Zona",
                text="Cantidad",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig_zona.update_layout(
                height=350,
                xaxis_title="",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_zona.update_traces(textposition="outside")
            st.plotly_chart(fig_zona, width="stretch")

        st.divider()

    # ── Fila 6: Detalle financiero de créditos ──────────────────────────────
    if vista in ("🔍 Todos", "💳 Detalle de Créditos"):
        st.subheader("💳 Detalle financiero de créditos a microempresarios")
        col9, col10 = st.columns(2)

        with col9:
            st.markdown("**Top 15 barrios por monto financiado**")
            cred_barrio_fin = (
                creditos.groupby("barrio")
                .agg(n_creditos=("monto", "count"), monto_total=("monto", "sum"))
                .sort_values("monto_total", ascending=False)
                .head(15)
                .reset_index()
            )
            cred_barrio_fin = cred_barrio_fin.sort_values("monto_total")
            fig_cred_barrio = px.bar(
                cred_barrio_fin,
                x="monto_total",
                y="barrio",
                orientation="h",
                color="n_creditos",
                color_continuous_scale="Oranges",
                text=cred_barrio_fin["monto_total"].apply(lambda x: f"${x:,.0f}"),
            )
            fig_cred_barrio.update_layout(
                height=450,
                yaxis_title="",
                xaxis_title="Monto total (COP)",
                coloraxis_colorbar_title="# Créditos",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_cred_barrio.update_traces(textposition="outside")
            st.plotly_chart(fig_cred_barrio, width="stretch")

        with col10:
            st.markdown("**Monto financiado por comuna**")
            cred_comuna_fin = (
                creditos.dropna(subset=["comuna"])
                .groupby("comuna")
                .agg(n_creditos=("monto", "count"), monto_total=("monto", "sum"))
                .sort_values("monto_total", ascending=False)
                .reset_index()
            )
            cred_comuna_fin["comuna"] = cred_comuna_fin["comuna"].astype(int).astype(str)
            fig_cred_comuna = px.bar(
                cred_comuna_fin,
                x="comuna",
                y="monto_total",
                color="n_creditos",
                color_continuous_scale="Teal",
                text=cred_comuna_fin["monto_total"].apply(lambda x: f"${x:,.0f}"),
            )
            fig_cred_comuna.update_layout(
                height=450,
                xaxis_title="Comuna",
                yaxis_title="Monto total (COP)",
                coloraxis_colorbar_title="# Créditos",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_cred_comuna.update_traces(textposition="outside")
            st.plotly_chart(fig_cred_comuna, width="stretch")

        st.divider()

        st.markdown("**Top 10 actividades financiadas (descripción detallada)**")
        desc_counts = (
            creditos["descripcion_de_actividad"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        desc_counts.columns = ["Descripción", "Créditos"]
        fig_desc = px.bar(
            desc_counts,
            x="Créditos",
            y="Descripción",
            orientation="h",
            color="Créditos",
            color_continuous_scale="Purples",
            text="Créditos",
        )
        fig_desc.update_layout(
            height=400,
            yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_desc.update_traces(textposition="outside")
        st.plotly_chart(fig_desc, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CHATBOT RAG
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(
        "Pregunta sobre barrios, comunas, perfiles emprendedores, "
        "oportunidades de negocio y rankings. "
        "Los datos provienen de **Medata** (WiFi público + registro de emprendedores)."
    )

    st.markdown(
        "**Ejemplos:** *¿Cuál es el perfil emprendedor en Castilla?* · "
        "*¿Dónde hay más oportunidad para emprender?* · "
        "*Recomiéndame un negocio para Belén* · "
        "*¿Cuáles barrios tienen más tráfico pero menos emprendimientos?*"
    )

    # ── Chat state ────────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pregúntale a ConectEMP..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando datos de Medata..."):
                try:
                    result = run_medcity_graph(user_query=prompt)
                    answer = result.get("answer", "No pude generar una respuesta.")
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources and sources != ["sin_datos"] and sources != ["sistema"]:
                        with st.expander("Fuentes consultadas"):
                            st.write(", ".join(sources))

                except Exception as e:
                    answer = f"Error al procesar la consulta: {e}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
