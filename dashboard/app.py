import os
import sys
from pathlib import Path

import streamlit as st

# ── Setup paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Inyectar secrets de Streamlit Cloud → env vars (para config/) ─────────────
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
    page_title="MedCity Dashboard — Medellín Emprendedora",
    page_icon="🏙️",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏙️ MedCity Dashboard")
st.caption("Asistente inteligente de oportunidades de emprendimiento — Medellín, Colombia")

st.markdown(
    "Haz preguntas sobre barrios, comunas, perfiles emprendedores, "
    "oportunidades de negocio y rankings de zonas. "
    "Los datos provienen de **Medata** (WiFi público + registro de emprendedores)."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtros opcionales")

    zona = st.selectbox(
        "Barrio o zona (opcional)",
        [""] + sorted([
            "Aranjuez", "Belén", "Boston", "Buenos Aires", "Castilla",
            "Doce de Octubre", "El Poblado", "Guayabal", "La América",
            "La Candelaria", "Laureles Estadio", "Manrique", "Popular",
            "Robledo", "San Antonio de Prado", "San Cristóbal", "San Javier",
            "Santa Cruz", "Villa Hermosa", "Altavista", "Palmitas", "Santa Elena",
        ]),
        index=0,
    )

    tipo_negocio = st.selectbox(
        "Tipo de negocio (opcional)",
        ["", "alimentos", "belleza", "tecnología", "comercio",
         "artesanía", "confección", "salud", "educación", "servicios"],
        index=0,
    )

    st.divider()
    st.markdown("**Ejemplos de preguntas:**")
    st.markdown(
        "- ¿Cuál es el perfil emprendedor en Castilla?\n"
        "- ¿Dónde hay más oportunidad para emprender?\n"
        "- Recomiéndame un negocio para Belén\n"
        "- ¿Cuáles barrios tienen más tráfico pero menos emprendimientos?\n"
        "- ¿Cómo es el perfil del consumidor WiFi en la comuna 13?"
    )

# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input del usuario ─────────────────────────────────────────────────────────
if prompt := st.chat_input("Pregúntale a MedCity..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando datos de Medata..."):
            try:
                result = run_medcity_graph(
                    user_query=prompt,
                    zone=zona.lower() if zona else None,
                    tipo_negocio=tipo_negocio if tipo_negocio else None,
                )
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
