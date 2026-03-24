"""
langgraph_flow.py — Pipeline RAG con LangGraph para el asistente de emprendimiento.

Intenciones soportadas:
  - Caracterizar zona     → indicadores, perfil demográfico, tráfico WiFi
  - Encontrar oportunidad → vacío de oferta, saturación, ranking
  - Recomendar negocio    → top rubros según perfil consumidor
  - Comparar zonas        → ranking de barrios/comunas por indicador
  - Consulta general      → contexto global de Medellín

Grafo:
  START → router → extraer_entidades → validar_info
        ─(completa)─→ construir_query → recuperar_datos (Pinecone)
                     → construir_contexto → sintetizar (Groq) → END
        ─(falta)────→ pedir_info → END
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from config import GroqSettings, build_groq_llm, retrieve_rag_context

# ── Tipos ─────────────────────────────────────────────────────────────────────

Intent = Literal[
    "caracterizar_zona",
    "encontrar_oportunidad",
    "recomendar_negocio",
    "comparar_zonas",
    "general",
]

# Barrios y comunas conocidos para extracción de entidades
# Incluye 21 barrios WiFi + barrios con datos de emprendedores (Database/Originales)
_BARRIOS_CONOCIDOS = [
    # WiFi medidos
    "la candelaria", "aranjuez", "castilla", "manrique", "robledo",
    "belen", "buenos aires", "san javier", "popular", "doce de octubre",
    "villa hermosa", "santa cruz", "laureles estadio", "la america",
    "guayabal", "el poblado", "san antonio de prado", "altavista",
    "san cristobal", "palmitas", "santa elena",
    # Con datos de emprendedores (sin WiFi)
    "boston", "bombona", "campo valdes", "las palmas", "enciso",
    "trinidad", "prado", "el salvador", "los colores", "floresta",
    "simon bolivar", "suramericana", "naranjal", "carlos e. restrepo",
    "el velodro", "la esperanza", "santa lucia", "trinidad", "la pradera",
    "boyaca las brisas", "caribe", "pedregal", "paris", "granizal",
    "la pilarica", "calasanz", "belen rincon", "fatima", "san bernardo",
    "el diamante", "santa ines", "los alcazares", "berlin",
    "nueva villa de aburra", "el corazon", "caicedo", "san diego",
    "asomadera", "gerona", "sucre", "alejandro echavarria",
    "el tesoro", "la visitacion", "loreto", "san miguel",
]

_COMUNAS_CONOCIDAS = {
    "1": "popular", "2": "santa cruz", "3": "manrique", "4": "aranjuez",
    "5": "castilla", "6": "doce de octubre", "7": "robledo",
    "8": "villa hermosa", "9": "buenos aires", "10": "la candelaria",
    "11": "laureles estadio", "12": "la america", "13": "san javier",
    "14": "el poblado", "15": "guayabal", "16": "belen",
}

# Tipos de negocio normalizados (Capa 1 del plan)
_TIPOS_NEGOCIO = [
    "alimentos", "comida", "restaurante", "cafeteria", "panaderia",
    "belleza", "peluqueria", "estetica", "spa",
    "tecnologia", "sistemas", "celulares", "computadores",
    "comercio", "tienda", "miscelanea", "papeleria",
    "artesania", "manualidades", "arte",
    "confeccion", "ropa", "textil", "moda",
    "salud", "farmacia", "drogueria",
    "educacion", "capacitacion", "cursos",
    "servicios", "lavanderia", "cerrajeria",
]


class MedCityState(TypedDict, total=False):
    """Estado compartido entre todos los nodos del grafo."""
    user_query: str
    intent: Intent
    zone: str               # barrio o comuna extraído de la pregunta
    zone_type: str           # "barrio" | "comuna" | "general"
    tipo_negocio: str        # tipo de negocio mencionado por el usuario
    info_completa: bool      # si tenemos suficiente info para responder
    pregunta_seguimiento: str  # pregunta para pedir info faltante
    rag_query: str           # query optimizada para búsqueda semántica
    n_results: int           # cuántos docs pedir al RAG
    retrieved_docs: List[Dict[str, Any]]
    structured_context: str
    answer: str
    sources: List[str]
    _is_city_level: bool


# ── Utilidades ────────────────────────────────────────────────────────────────

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalize(s: str) -> str:
    return _strip_accents(s.lower().strip())


# ── NODO 1: Router de intención ──────────────────────────────────────────────

def router_intencion(state: MedCityState) -> MedCityState:
    """Detecta la intención del usuario entre las 5 intenciones soportadas."""
    q = _normalize(state.get("user_query", ""))

    kw_recomendar = [
        "recomienda", "recomendar", "que negocio", "que tipo de negocio",
        "abrir", "montar", "invertir", "mejor negocio", "idea de negocio",
        "que puedo vender", "que me conviene",
    ]
    kw_oportunidad = [
        "oportunidad", "saturacion", "saturada", "vacio", "brecha",
        "potencial", "demanda", "score", "ranking", "mejor zona",
        "donde hay mas", "donde conviene", "emprender", "donde emprender",
        "menos emprendimientos", "mas trafico", "sin artesanos",
        "brecha de credito", "acceso a credito",
        "credito", "creditos", "financiamiento", "financiar",
        "microempresario", "monto", "prestamo",
    ]
    kw_comparar = [
        "cuales barrios", "cuales comunas", "comparar", "ranking",
        "top barrios", "top comunas", "mejores barrios", "mejores zonas",
        "mas trafico pero menos", "listado de",
    ]
    kw_caracterizar = [
        "caracter", "perfil", "como es", "datos de", "informacion de",
        "describir", "indicadores", "estadisticas", "trafico", "wifi",
        "emprendedores en", "poblacion", "densidad", "tipico",
        "cuantos emprendedores", "cuantos negocios", "quien emprende",
        "cuantos creditos", "monto financiado", "creditos otorgados",
    ]
    # Keywords que indican consultas a nivel ciudad (necesitan doc global)
    kw_ciudad = [
        "medellin", "toda la ciudad", "en total", "cuantos artesanos",
        "cuantos emprendedores hay", "a nivel ciudad", "panorama general",
        "resumen general", "resumen de la ciudad", "todos los barrios",
        "todas las comunas", "en la ciudad",
    ]

    intent: Intent = "general"

    # Prioridad: comparar > recomendar > oportunidad > caracterizar
    if any(k in q for k in kw_comparar):
        intent = "comparar_zonas"
    elif any(k in q for k in kw_recomendar):
        intent = "recomendar_negocio"
    elif any(k in q for k in kw_oportunidad):
        intent = "encontrar_oportunidad"
    elif any(k in q for k in kw_caracterizar):
        intent = "caracterizar_zona"

    # Si menciona un barrio/comuna y la intención es general → caracterizar
    if intent == "general":
        for b in _BARRIOS_CONOCIDOS:
            if b in q:
                intent = "caracterizar_zona"
                break
        if intent == "general":
            for cnum in _COMUNAS_CONOCIDAS:
                if f"comuna {cnum}" in q or f"comuna{cnum}" in q:
                    intent = "caracterizar_zona"
                    break

    # Detectar consultas a nivel ciudad → comparar_zonas (necesita doc global + múltiples docs)
    is_city_level = any(k in q for k in kw_ciudad)
    if is_city_level and intent in ("general", "caracterizar_zona"):
        intent = "comparar_zonas"

    print(f"  [router] Intención detectada: {intent}"
          f"{' (nivel ciudad)' if is_city_level else ''}")
    return {"intent": intent, "_is_city_level": is_city_level}


# ── NODO 2: Extraer entidades ────────────────────────────────────────────────

def extraer_entidades(state: MedCityState) -> MedCityState:
    """
    Extrae zona (barrio/comuna) y tipo_negocio del texto del usuario.
      - Caracterizar → necesita zona
      - Oportunidad  → puede tener tipo_negocio
      - Recomendar   → necesita zona
    """
    q = _normalize(state.get("user_query", ""))
    zone = state.get("zone", "")
    zone_type = "general"
    tipo_negocio = ""

    # ── Extraer zona ──
    if not zone:
        # Barrios (match más largo primero)
        for barrio in sorted(_BARRIOS_CONOCIDOS, key=len, reverse=True):
            if barrio in q:
                zone = barrio
                zone_type = "barrio"
                break

        # Comunas por número
        if not zone:
            m = re.search(r"comuna\s*(\d{1,2})", q)
            if m:
                cnum = m.group(1)
                zone = _COMUNAS_CONOCIDAS.get(cnum, f"comuna {cnum}")
                zone_type = "comuna"

        # Comunas por nombre
        if not zone:
            for _cnum, cname in sorted(
                _COMUNAS_CONOCIDAS.items(), key=lambda x: len(x[1]), reverse=True
            ):
                if cname in q:
                    zone = cname
                    zone_type = "comuna"
                    break
    else:
        zone = _normalize(zone)
        zone_type = "barrio"

    # ── Extraer tipo de negocio ──
    for tipo in _TIPOS_NEGOCIO:
        if tipo in q:
            tipo_negocio = tipo
            break

    print(f"  [entidades] Zona: '{zone or 'general'}' ({zone_type})"
          f" | Negocio: '{tipo_negocio or 'no especificado'}'")
    return {"zone": zone, "zone_type": zone_type, "tipo_negocio": tipo_negocio}


# ── NODO 3: Validar información ─────────────────────────────────────────────

def validar_info(state: MedCityState) -> MedCityState:
    """
    Verifica si tenemos suficiente info para responder.
    Si falta info crítica, genera pregunta de seguimiento.

    Reglas:
    - caracterizar_zona  → REQUIERE zona
    - recomendar_negocio → REQUIERE zona
    - encontrar_oportunidad → puede funcionar sin zona (ranking global)
    - comparar_zonas → puede funcionar sin zona (ranking global)
    - general → siempre ok
    """
    intent = state.get("intent", "general")
    zone = state.get("zone", "")
    tipo_negocio = state.get("tipo_negocio", "")

    pregunta = ""

    if intent == "caracterizar_zona" and not zone:
        pregunta = (
            "¿En qué barrio o comuna te gustaría conocer los indicadores? "
            "Puedo darte datos de: Aranjuez, Castilla, Belén, Robledo, San Javier, "
            "La Candelaria, Popular, El Poblado, entre otros."
        )
    elif intent == "recomendar_negocio" and not zone:
        pregunta = (
            "¿En qué zona quieres ubicarte? Dime el barrio o comuna y te doy "
            "los top 3 rubros recomendados según el perfil de consumidores de la zona."
        )
    elif intent == "encontrar_oportunidad" and tipo_negocio:
        # Si ya tiene tipo de negocio, no necesita más info
        pass

    info_ok = not pregunta
    if not info_ok:
        print(f"  [validar] Info INCOMPLETA → pregunta de seguimiento")
    else:
        print(f"  [validar] Info completa → continuar pipeline")

    return {"info_completa": info_ok, "pregunta_seguimiento": pregunta}


def _ruta_validacion(state: MedCityState) -> str:
    """Decide si continuar con RAG o pedir info al usuario."""
    if state.get("info_completa", True):
        return "construir_query_rag"
    return "pedir_info"


# ── NODO 3b: Pedir información ──────────────────────────────────────────────

def pedir_info(state: MedCityState) -> MedCityState:
    """Devuelve pregunta de seguimiento como respuesta."""
    pregunta = state.get("pregunta_seguimiento", "")
    print(f"  [pedir_info] Pidiendo: {pregunta[:60]}...")
    return {"answer": pregunta, "sources": ["sistema"]}


# ── NODO 4: Construir query RAG ─────────────────────────────────────────────

def construir_query_rag(state: MedCityState) -> MedCityState:
    """
    Construye query semántica optimizada para Pinecone.
    Templates por intención para mejor retrieval.
    """
    intent = state.get("intent", "general")
    zone = state.get("zone", "")
    zone_type = state.get("zone_type", "general")
    tipo_negocio = state.get("tipo_negocio", "")
    user_q = state.get("user_query", "")

    n_results = 5

    if zone and zone_type == "barrio":
        base = f"BARRIO: {zone}"
        if intent == "caracterizar_zona":
            rag_q = (f"{base} perfil consumidor tráfico WiFi emprendimientos "
                     f"demográfico densidad edad dispositivo créditos microempresarios")
        elif intent == "encontrar_oportunidad":
            rag_q = (f"{base} score oportunidad cuadrante saturación vacío "
                     f"oferta usuarios por emprendimiento créditos financiamiento")
        elif intent == "recomendar_negocio":
            negocio_extra = f" {tipo_negocio}" if tipo_negocio else ""
            rag_q = (f"{base} negocios recomendados tipo emprendimiento "
                     f"oportunidad perfil consumidor{negocio_extra}")
        else:
            rag_q = f"{base} {user_q}"

    elif zone and zone_type == "comuna":
        rag_q = (f"COMUNA: {zone} densidad emprendedora oportunidad "
                 f"infraestructura WiFi población emprendimientos")

    elif intent == "encontrar_oportunidad":
        if tipo_negocio:
            rag_q = (f"oportunidad emprendimiento {tipo_negocio} saturación "
                     f"vacío oferta barrios Medellín créditos financiamiento")
        else:
            rag_q = ("zonas alta oportunidad score emprendimiento bajo "
                     "saturación alto tráfico WiFi vacío oferta créditos microempresarios")
        n_results = 8

    elif intent == "comparar_zonas":
        rag_q = ("ranking barrios score oportunidad tráfico WiFi "
                 "emprendimientos vacío oferta saturación Medellín")
        n_results = 10

    elif intent == "recomendar_negocio":
        rag_q = ("negocios recomendados oportunidad emprendimiento "
                 "consumidor perfil Medellín cuadrante")
        n_results = 8

    else:
        rag_q = user_q or "contexto general Medellín emprendimiento WiFi"
        n_results = 8  # Más docs para consultas generales

    print(f"  [rag_query] '{rag_q}' (top {n_results})")
    return {"rag_query": rag_q, "n_results": n_results}


# ── NODO 5: Recuperar datos ──────────────────────────────────────────────────

def recuperar_datos(state: MedCityState) -> MedCityState:
    """
    Recupera documentos de Pinecone con filtro por zona cuando aplica.
    Para comparaciones/rankings, recupera sin filtro para obtener múltiples zonas.
    """
    rag_q = state.get("rag_query", state.get("user_query", ""))
    n_results = state.get("n_results", 5)
    zone = state.get("zone", "")
    zone_type = state.get("zone_type", "general")
    intent = state.get("intent", "general")

    # Filtro por metadata solo para consultas de zona específica
    # Para comparaciones y oportunidades sin zona, necesitamos múltiples docs
    where_filter = None
    if zone and zone_type in ("barrio", "comuna") and intent not in ("comparar_zonas",):
        where_filter = {"zone": zone}

    try:
        items = retrieve_rag_context(rag_q, n_results=n_results, where=where_filter)
        # Si el filtro retorna vacío, reintentar sin filtro
        if not items and where_filter:
            print(f"  [recuperar] Sin resultados con filtro zone={zone}, buscando sin filtro...")
            items = retrieve_rag_context(rag_q, n_results=n_results)
    except Exception:
        try:
            items = retrieve_rag_context(rag_q, n_results=n_results)
        except Exception as e:
            print(f"  [recuperar] ERROR: {e}")
            items = []

    # Para comparaciones, oportunidades y consultas generales, también traer el doc global
    if intent in ("comparar_zonas", "encontrar_oportunidad", "general") or state.get("_is_city_level"):
        try:
            global_items = retrieve_rag_context(
                "contexto general Medellín", n_results=1,
                where={"tipo_documento": "global"},
            )
            items.extend(global_items)
        except Exception:
            pass

    docs: List[Dict[str, Any]] = []
    sources: List[str] = []
    seen_zones = set()
    for item in items:
        meta = item.get("metadata") or {}
        doc_text = item.get("document", "")
        dist = item.get("distance")
        zone_name = meta.get("zone", "")

        # Evitar docs duplicados de la misma zona
        doc_id = f"{zone_name}_{meta.get('tipo_documento', '')}"
        if doc_id in seen_zones:
            continue
        seen_zones.add(doc_id)

        docs.append({
            "text": doc_text,
            "metadata": meta,
            "distance": dist,
            "zone": zone_name,
            "tipo": meta.get("tipo_documento", ""),
        })
        if zone_name and zone_name not in sources:
            sources.append(zone_name)

    if not docs:
        sources = ["sin_datos"]

    dists = [round(d["distance"], 3) for d in docs[:3]] if docs else []
    print(f"  [recuperar] {len(docs)} documentos recuperados (distancias: {dists}...)")
    return {"retrieved_docs": docs, "sources": sources}


# ── NODO 6: Construir contexto ───────────────────────────────────────────────

def construir_contexto(state: MedCityState) -> MedCityState:
    """
    Ensambla el contexto completo con TEXTO RICO de los documentos RAG.
    Para comparaciones, incluye resumen comparativo al final.
    """
    docs = state.get("retrieved_docs", [])
    intent = state.get("intent", "general")
    zone = state.get("zone", "")
    tipo_negocio = state.get("tipo_negocio", "")

    sections = ["=== DATOS MEDATA MEDELLÍN ==="]
    sections.append(f"Intención: {intent} | Zona: {zone or 'general'}"
                    f"{' | Tipo negocio: ' + tipo_negocio if tipo_negocio else ''}")
    sections.append(f"Documentos relevantes: {len(docs)}\n")

    # Priorizar documento global (ponerlo primero si es consulta amplia)
    global_docs = [d for d in docs if d.get("tipo") == "global"]
    other_docs = [d for d in docs if d.get("tipo") != "global"]
    ordered_docs = global_docs + other_docs if (not zone or state.get("_is_city_level")) else docs

    # Insertar texto completo de cada documento
    # Si la zona solicitada no coincide con ningún doc, agregar nota
    zona_encontrada = any(
        _normalize(doc.get("zone", "")) == _normalize(zone)
        for doc in docs
    ) if zone else True

    if zone and not zona_encontrada and docs:
        sections.append(
            f"NOTA: No se encontraron datos específicos para '{zone}'. "
            f"Se muestran datos de zonas similares o cercanas como referencia.\n"
        )

    for i, doc in enumerate(ordered_docs, 1):
        dist = doc.get("distance", 0)
        relevancia = "ALTA" if dist < 1.0 else "MEDIA" if dist < 1.5 else "BAJA"
        sections.append(f"--- Documento {i} (relevancia: {relevancia}) ---")
        sections.append(doc.get("text", "Sin contenido"))
        sections.append("")

    # Para comparaciones, agregar tabla resumen de scores
    if intent in ("comparar_zonas", "encontrar_oportunidad") and len(docs) > 1:
        sections.append("--- TABLA COMPARATIVA ---")
        for doc in sorted(docs, key=lambda d: d.get("metadata", {}).get(
                "score_oportunidad", 0), reverse=True):
            meta = doc.get("metadata", {})
            z = meta.get("zone", "?")
            score = meta.get("score_oportunidad", "N/A")
            cuad = meta.get("cuadrante", meta.get("nivel_oportunidad", "N/A"))
            trafico = meta.get("trafico_wifi", "N/A")
            emp = meta.get("emprendedores", meta.get("n_emprendedores", "N/A"))
            upc = meta.get("usuarios_por_emprendimiento", "N/A")
            sections.append(
                f"  {z}: score={score}, cuadrante={cuad}, "
                f"tráfico={trafico}, emprendedores={emp}, "
                f"usuarios/emp={upc}"
            )
        sections.append("")

    ctx = "\n".join(sections)
    # Límite para no desbordar el contexto del LLM
    if len(ctx) > 16000:
        ctx = ctx[:16000] + "\n... [contexto truncado]"

    print(f"  [contexto] {len(ctx)} caracteres ensamblados")
    return {"structured_context": ctx}


# ── NODO 7: Sintetizar respuesta ─────────────────────────────────────────────

# Prompts específicos por intención
_PROMPTS_POR_INTENT = {
    "caracterizar_zona": (
        "TAREA: Describe la zona de forma completa pero sencilla. Incluye:\n"
        "- Total de emprendedores (artesanos + microempresarios creditados, SIEMPRE suma ambos)\n"
        "- Desglose: cuántos son artesanos y cuántos tienen crédito\n"
        "- Qué tipos de negocio hay (artesanales Y actividades financiadas)\n"
        "- Si hay WiFi público: cuánta gente lo usa y de qué edad\n"
        "- Quién emprende ahí (edad, género, estrato)\n"
        "- Créditos: cuántos se han dado, por cuánta plata, a qué actividades\n"
        "- ¿Es buena zona para emprender? ¿Hay mucha o poca competencia?\n"
        "- Un consejo práctico basado en todo lo anterior\n"
        "Si no hay WiFi en el barrio, menciónalo."
    ),
    "encontrar_oportunidad": (
        "TAREA: Analiza las oportunidades de emprendimiento comparando zonas:\n"
        "- Para cada zona menciona el TOTAL de emprendedores (artesanos + creditados)\n"
        "- Dónde hay mucha gente pero pocos negocios (oportunidad)\n"
        "- Dónde ya hay demasiada competencia\n"
        "- Dónde hay créditos disponibles y dónde no\n"
        "- Compara al menos 3-5 zonas con cifras concretas\n"
        "Ordena de mejor a peor oportunidad y cierra con un consejo práctico."
    ),
    "recomendar_negocio": (
        "TAREA: Recomienda 3 negocios para la zona. Para cada uno explica:\n"
        "- Qué negocio montar y por qué funcionaría ahí\n"
        "- Basado en: perfil de la gente, competencia actual (total emprendedores), créditos disponibles\n"
        "Al final, di cuál es la mejor opción y por qué."
    ),
    "comparar_zonas": (
        "TAREA: Compara las zonas con datos completos:\n"
        "- Total de emprendedores por zona (artesanos + microemp. creditados)\n"
        "- Cuánta gente pasa (tráfico WiFi)\n"
        "- Créditos otorgados y montos\n"
        "- Nivel de competencia en cada zona\n"
        "Ordena de mejor a peor para emprender. Cierra con recomendación."
    ),
    "general": (
        "TAREA: Responde la pregunta con los datos disponibles.\n"
        "IMPORTANTE: Cuando hables de emprendedores, SIEMPRE suma artesanos + microempresarios creditados.\n"
        "Da el total Y el desglose (ej: '294 emprendedores: 7 artesanos y 287 con crédito').\n"
        "Si se pregunta por totales, incluye TODOS los datos que aparecen.\n"
        "Si hay distribución por barrio, muéstrala completa.\n"
        "Si no tienes el dato, di 'No tenemos ese dato en nuestra base'.\n"
        "Usa el documento global para dar cifras totales de toda la ciudad."
    ),
}


def sintetizar_respuesta(state: MedCityState) -> MedCityState:
    """
    Genera la respuesta usando Groq LLM con prompts específicos por intención
    y el contexto RAG completo.
    Siempre incluye: dato concreto + fuente explícita + recomendación.
    """
    context = state.get("structured_context", "")
    user_q = state.get("user_query", "")
    intent = state.get("intent", "general")
    sources = state.get("sources", [])
    sources_str = ", ".join(sources) if sources else "Medata Medellín"

    tarea = _PROMPTS_POR_INTENT.get(intent, _PROMPTS_POR_INTENT["general"])

    formato_respuesta = (
        "Responde en 10-15 líneas. Estructura:\n"
        "1) Respuesta directa con cifras TOTALES (artesanos + creditados)\n"
        "2) Desglose: cuántos de cada tipo y qué hacen\n"
        "3) Contexto relevante (WiFi, créditos, competencia)\n"
        "4) Qué hacer con eso (consejo práctico)\n"
        f"5) Fuente: Datos Medata Medellín — {sources_str}"
    )

    if intent == "recomendar_negocio":
        formato_respuesta = (
            "Responde en 15-20 líneas. Estructura:\n"
            "1) Cómo es la zona: total emprendedores (desglose), tráfico, créditos\n"
            "2) Negocio 1: nombre + por qué (con datos)\n"
            "3) Negocio 2: nombre + por qué (con datos)\n"
            "4) Negocio 3: nombre + por qué (con datos)\n"
            "5) Cuál elegir primero y por qué\n"
            f"6) Fuente: Datos Medata Medellín — {sources_str}"
        )

    prompt = (
        "Eres un asistente amigable que ayuda a emprendedores en Medellín, Colombia.\n\n"
        "REGLAS:\n"
        "- Responde SOLO en español.\n"
        "- Usa lenguaje sencillo, como si hablaras con alguien que NO es experto en datos.\n"
        "- NO uses términos técnicos como 'mismatch demográfico', 'ratio', 'score', 'cuadrante', 'densidad emprendedora', 'índice de diversificación'.\n"
        "  En su lugar di cosas como: 'hay poca competencia', 'mucha gente pasa por ahí', 'buena oportunidad'.\n"
        "- NO inventes datos. Usa SOLO lo que está en el contexto.\n"
        "- Si no tienes un dato, di 'No tenemos ese dato'.\n"
        "- Incluye cifras reales (ejemplo: '294 emprendedores', '$500,000 en créditos').\n"
        "- REGLA DE TOTALES: Cuando el contexto tenga 'artesanos' y 'microemp. creditados', SIEMPRE da el total sumado Y el desglose. Nunca des solo uno de los dos.\n"
        "- Da respuestas COMPLETAS: no solo la cifra, explica qué significa y qué puede hacer el emprendedor con esa info.\n"
        "- Cierra siempre con un consejo práctico.\n"
        "- Si no hay datos directos de la zona, usa zonas cercanas y acláralo.\n\n"
        f"{tarea}\n\n"
        f"PREGUNTA: {user_q}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        "FORMATO:\n"
        f"{formato_respuesta}"
    )

    def _needs_continuation(text: str, finish_reason: str | None) -> bool:
        if not text:
            return False
        if finish_reason and str(finish_reason).lower() in {"length", "max_tokens"}:
            return True
        trimmed = text.rstrip()
        # Mid-word or missing sentence closure is a strong truncation signal.
        if not trimmed:
            return False
        if trimmed[-1].isalnum() and len(trimmed.split()) > 40 and not trimmed.endswith((".", "?", "!", ":")):
            return True
        return False

    try:
        cfg = GroqSettings.from_env()
        llm = build_groq_llm(cfg)
        print(f"  [llm] Enviando a {cfg.model_name}...")
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = str(response.content).strip()

        finish_reason = None
        try:
            finish_reason = (response.response_metadata or {}).get("finish_reason")
        except Exception:
            finish_reason = None

        if _needs_continuation(answer, finish_reason):
            continuation_prompt = (
                "Continua EXACTAMENTE desde donde te quedaste, sin repetir contenido. "
                "Completa toda la estructura solicitada y cierra con la línea de fuente.\n\n"
                f"RESPUESTA PARCIAL:\n{answer}"
            )
            response_2 = llm.invoke([HumanMessage(content=continuation_prompt)])
            answer_2 = str(response_2.content).strip()
            if answer_2:
                answer = f"{answer}\n{answer_2}".strip()

        print(f"  [llm] Respuesta generada ({len(answer)} chars)")
    except Exception as e:
        print(f"  [llm] Fallback sin LLM: {e}")
        docs = state.get("retrieved_docs", [])
        if docs:
            best = docs[0].get("text", "")
            lines = best.split("\n")[:20]
            answer = (
                "**Datos de Medata** (modo sin LLM):\n\n"
                + "\n".join(lines)
                + f"\n\n*Fuente: Medata Medellín — {sources_str}*\n"
                + "\nConfigura GROQ_API_KEY en .env para respuestas sintetizadas con IA."
            )
        else:
            answer = (
                "No encontré datos relevantes para tu consulta en la base de Medata.\n"
                "Prueba preguntando por un barrio específico como: "
                "Aranjuez, Castilla, Belén, San Javier, Popular, La Candelaria, etc."
            )

    return {"answer": answer}


# ── Construcción del grafo ────────────────────────────────────────────────────

def build_medcity_graph() -> Any:
    """
    Grafo LangGraph completo:

      START → router → extraer_entidades → validar_info
            ─(completa)─→ construir_query → recuperar_datos (Pinecone)
                         → construir_contexto → sintetizar (Groq) → END
            ─(falta)────→ pedir_info → END
    """
    graph = StateGraph(MedCityState)

    # Nodos del pipeline
    graph.add_node("router_intencion", router_intencion)
    graph.add_node("extraer_entidades", extraer_entidades)
    graph.add_node("validar_info", validar_info)
    graph.add_node("pedir_info", pedir_info)
    graph.add_node("construir_query_rag", construir_query_rag)
    graph.add_node("recuperar_datos", recuperar_datos)
    graph.add_node("construir_contexto", construir_contexto)
    graph.add_node("sintetizar_respuesta", sintetizar_respuesta)

    # Flujo lineal hasta validación
    graph.add_edge(START, "router_intencion")
    graph.add_edge("router_intencion", "extraer_entidades")
    graph.add_edge("extraer_entidades", "validar_info")

    # Bifurcación condicional
    graph.add_conditional_edges(
        "validar_info",
        _ruta_validacion,
        {
            "construir_query_rag": "construir_query_rag",
            "pedir_info": "pedir_info",
        },
    )

    # Rama principal: RAG → Contexto → LLM
    graph.add_edge("construir_query_rag", "recuperar_datos")
    graph.add_edge("recuperar_datos", "construir_contexto")
    graph.add_edge("construir_contexto", "sintetizar_respuesta")
    graph.add_edge("sintetizar_respuesta", END)

    # Rama: pedir info → END
    graph.add_edge("pedir_info", END)

    return graph.compile()


def run_medcity_graph(
    user_query: str,
    zone: str | None = None,
    tipo_negocio: str | None = None,
) -> MedCityState:
    """
    Punto de entrada principal. Ejecuta el grafo completo.

    Args:
        user_query: Pregunta del usuario en lenguaje natural.
        zone: (Opcional) Barrio o comuna pre-seleccionado del dashboard.
        tipo_negocio: (Opcional) Tipo de negocio pre-seleccionado.

    Returns:
        Estado final con 'answer' y 'sources'.
    """
    print(f"\n{'='*60}")
    print(f"MedCity RAG Pipeline")
    print(f"  Query: '{user_query}'")
    if zone:
        print(f"  Zona pre-seleccionada: '{zone}'")
    if tipo_negocio:
        print(f"  Tipo negocio: '{tipo_negocio}'")
    print(f"{'='*60}")

    app = build_medcity_graph()

    initial_state: MedCityState = {"user_query": user_query}
    if zone:
        initial_state["zone"] = zone
    if tipo_negocio:
        initial_state["tipo_negocio"] = tipo_negocio

    result = app.invoke(initial_state)

    print(f"{'='*60}")
    print(f"Pipeline completado. Respuesta: {len(result.get('answer', ''))} chars")
    print(f"{'='*60}\n")
    return result
