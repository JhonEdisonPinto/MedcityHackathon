import streamlit as st

st.set_page_config(page_title="MedCity Dashboard", layout="wide")

st.title("MedCity Dashboard")
st.caption("Hackathon Medellin - Prototipo inicial")

st.write(
    """
    Entorno inicial listo.

    Siguientes pasos:
    1. Cargar datos procesados desde `data/processed`.
    2. Construir KPIs de emprendimiento, credito y trafico wifi.
    3. Agregar mapa de oportunidad por comuna y chatbot RAG.
    """
)
