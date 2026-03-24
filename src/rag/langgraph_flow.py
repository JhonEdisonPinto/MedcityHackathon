from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from config import GroqSettings, build_groq_llm

Intent = Literal[
    "caracterizar_zona",
    "encontrar_oportunidad",
    "recomendar_negocio",
    "desconocida",
]


class MedCityState(TypedDict, total=False):
    user_query: str
    intent: Intent
    zone: str
    filters: Dict[str, Any]
    missing_fields: List[str]
    retrieved_records: List[Dict[str, Any]]
    structured_context: str
    answer: str
    sources: List[str]


def _detect_intent(query: str) -> Intent:
    q = (query or "").lower()

    if any(k in q for k in ["caracter", "perfil", "zona", "comuna", "barrio"]):
        if any(k in q for k in ["negocio abrir", "recomienda", "recomendar"]):
            return "recomendar_negocio"
        if any(k in q for k in ["oportunidad", "saturacion", "vacío", "vacio"]):
            return "encontrar_oportunidad"
        return "caracterizar_zona"

    if any(k in q for k in ["oportunidad", "saturacion", "vacío", "vacio", "brecha"]):
        return "encontrar_oportunidad"

    if any(k in q for k in ["recomienda", "recomendar", "que negocio", "qué negocio", "abrir"]):
        return "recomendar_negocio"

    return "desconocida"


def router_intencion(state: MedCityState) -> MedCityState:
    query = state.get("user_query", "")
    return {"intent": _detect_intent(query)}


def _next_from_router(state: MedCityState) -> str:
    intent = state.get("intent", "desconocida")
    if intent == "caracterizar_zona":
        return "intent_caracterizar_zona"
    if intent == "encontrar_oportunidad":
        return "intent_encontrar_oportunidad"
    if intent == "recomendar_negocio":
        return "intent_recomendar_negocio"
    return "intent_caracterizar_zona"


def intent_caracterizar_zona(_: MedCityState) -> MedCityState:
    return {}


def intent_encontrar_oportunidad(_: MedCityState) -> MedCityState:
    return {}


def intent_recomendar_negocio(_: MedCityState) -> MedCityState:
    return {}


def solicitar_zona(state: MedCityState) -> MedCityState:
    missing = list(state.get("missing_fields", []))
    if not state.get("zone") and "zone" not in missing:
        missing.append("zone")
    return {"missing_fields": missing}


def solicitar_filtro(state: MedCityState) -> MedCityState:
    filters = dict(state.get("filters", {}))
    missing = list(state.get("missing_fields", []))

    if "tipo_emprendimiento" not in filters:
        if "tipo_emprendimiento" not in missing:
            missing.append("tipo_emprendimiento")

    return {"filters": filters, "missing_fields": missing}


def recuperar_datos(state: MedCityState) -> MedCityState:
    # Nodo placeholder: conecta aquí tu consulta real a parquet/sql/rag store.
    zone = state.get("zone", "sin_zona")
    records = [
        {
            "zone": zone,
            "emprendedores": 0,
            "tipo_dominante": "sin_datos",
            "credito_promedio": 0,
            "trafico_wifi": 0,
        }
    ]
    return {"retrieved_records": records, "sources": ["medata_mock"]}


def construir_contexto_estructurado_rag(state: MedCityState) -> MedCityState:
    records = state.get("retrieved_records", [])
    intent = state.get("intent", "desconocida")
    filters = state.get("filters", {})

    lines = [
        "Contexto MedCity (estructurado):",
        f"- Intencion: {intent}",
        f"- Filtros: {filters if filters else 'sin_filtros'}",
        f"- Registros recuperados: {len(records)}",
    ]

    for i, rec in enumerate(records, start=1):
        lines.append(
            "- Registro "
            f"{i}: zona={rec.get('zone')}, emprendedores={rec.get('emprendedores')}, "
            f"tipo_dominante={rec.get('tipo_dominante')}, credito_promedio={rec.get('credito_promedio')}, "
            f"trafico_wifi={rec.get('trafico_wifi')}"
        )

    return {"structured_context": "\n".join(lines)}


def llm_sintesis_controlada(state: MedCityState) -> MedCityState:
    # Synthesis node wired to Groq LLM with strict output constraints.
    missing = state.get("missing_fields", [])
    context = state.get("structured_context", "")
    sources = ", ".join(state.get("sources", [])) or "sin_fuente"

    if missing:
        answer = (
            "Para responder con precision faltan datos: "
            f"{', '.join(missing)}. "
            "Comparte esos campos y vuelvo a calcular la recomendacion con fuentes reales."
        )
    else:
        prompt = (
            "Eres un asistente del MedCity Dashboard. Responde en espanol claro y breve. "
            "No inventes cifras. Usa solo el contexto dado. "
            "Siempre cierra con una linea de fuente explicita.\n\n"
            f"Consulta del usuario: {state.get('user_query', '')}\n"
            f"Fuentes disponibles: {sources}\n\n"
            "Contexto estructurado:\n"
            f"{context}\n\n"
            "Formato esperado:\n"
            "1) Hallazgo principal en 1-2 frases\n"
            "2) Recomendacion accionable en 1 frase\n"
            "3) Fuente: <fuente(s)>"
        )

        try:
            cfg = GroqSettings.from_env()
            llm = build_groq_llm(cfg)
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = str(response.content).strip()
        except Exception:
            answer = (
                "Sintesis basada en datos estructurados (modo fallback).\n"
                f"Hallazgo: {context}\n"
                "Recomendacion: valida GROQ_API_KEY para habilitar sintesis LLM completa.\n"
                f"Fuente: {sources}"
            )

    return {"answer": answer}


def build_medcity_graph() -> Any:
    graph = StateGraph(MedCityState)

    graph.add_node("router_intencion", router_intencion)

    graph.add_node("intent_caracterizar_zona", intent_caracterizar_zona)
    graph.add_node("intent_encontrar_oportunidad", intent_encontrar_oportunidad)
    graph.add_node("intent_recomendar_negocio", intent_recomendar_negocio)

    graph.add_node("solicitar_zona_caracterizar", solicitar_zona)
    graph.add_node("solicitar_filtro_oportunidad", solicitar_filtro)
    graph.add_node("solicitar_zona_recomendar", solicitar_zona)

    graph.add_node("recuperar_datos_caracterizar", recuperar_datos)
    graph.add_node("recuperar_datos_oportunidad", recuperar_datos)
    graph.add_node("recuperar_datos_recomendar", recuperar_datos)

    graph.add_node("construir_contexto_estructurado_rag", construir_contexto_estructurado_rag)
    graph.add_node("llm_sintesis_controlada", llm_sintesis_controlada)

    graph.add_edge(START, "router_intencion")
    graph.add_conditional_edges(
        "router_intencion",
        _next_from_router,
        {
            "intent_caracterizar_zona": "intent_caracterizar_zona",
            "intent_encontrar_oportunidad": "intent_encontrar_oportunidad",
            "intent_recomendar_negocio": "intent_recomendar_negocio",
        },
    )

    graph.add_edge("intent_caracterizar_zona", "solicitar_zona_caracterizar")
    graph.add_edge("solicitar_zona_caracterizar", "recuperar_datos_caracterizar")
    graph.add_edge("recuperar_datos_caracterizar", "construir_contexto_estructurado_rag")

    graph.add_edge("intent_encontrar_oportunidad", "solicitar_filtro_oportunidad")
    graph.add_edge("solicitar_filtro_oportunidad", "recuperar_datos_oportunidad")
    graph.add_edge("recuperar_datos_oportunidad", "construir_contexto_estructurado_rag")

    graph.add_edge("intent_recomendar_negocio", "solicitar_zona_recomendar")
    graph.add_edge("solicitar_zona_recomendar", "recuperar_datos_recomendar")
    graph.add_edge("recuperar_datos_recomendar", "construir_contexto_estructurado_rag")

    graph.add_edge("construir_contexto_estructurado_rag", "llm_sintesis_controlada")
    graph.add_edge("llm_sintesis_controlada", END)

    return graph.compile()


def run_medcity_graph(user_query: str, zone: str | None = None) -> MedCityState:
    app = build_medcity_graph()
    initial_state: MedCityState = {"user_query": user_query}
    if zone:
        initial_state["zone"] = zone
    return app.invoke(initial_state)
