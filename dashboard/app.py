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
    page_title="MedCity Dashboard — Medellín Emprendedora",
    page_icon="🏙️",
    layout="wide",
)

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
    return densidad_b, densidad_c, edad_g, flujo_b, flujo_c, master, artesanos


densidad_b, densidad_c, edad_g, flujo_b, flujo_c, master, artesanos = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏙️ MedCity Dashboard")
st.caption("Oportunidades de emprendimiento en Medellín — Datos Medata")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_chat = st.tabs(["📊 Dashboard", "💬 Asistente IA"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_emp = int(densidad_b["n_emprendedores"].sum())
    total_barrios = len(densidad_b)
    total_usuarios = int(edad_g["total_usuarios"].sum())
    total_comunas = len(densidad_c)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Emprendedores", f"{total_emp:,}")
    k2.metric("Barrios con datos", total_barrios)
    k3.metric("Usuarios WiFi", f"{total_usuarios:,}")
    k4.metric("Comunas", total_comunas)

    st.divider()

    # ── Fila 1: Barras + Dona ─────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Top 15 barrios por emprendedores")
        top15 = densidad_b.nlargest(15, "n_emprendedores").sort_values("n_emprendedores")
        fig_bar = px.bar(
            top15,
            x="n_emprendedores",
            y="barrio",
            orientation="h",
            color="zona",
            text="n_emprendedores",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_bar.update_layout(
            height=450,
            yaxis_title="",
            xaxis_title="Emprendedores registrados",
            legend_title="Zona",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

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
        st.plotly_chart(fig_dona, use_container_width=True)

    st.divider()

    # ── Fila 2: Comunas + Perfil emprendedor ──────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Densidad emprendedora por comuna")
        dc = densidad_c.sort_values("densidad_x1000_hab", ascending=False)
        fig_dens = px.bar(
            dc,
            x="nombre_comuna",
            y="densidad_x1000_hab",
            color="densidad_x1000_hab",
            color_continuous_scale="RdYlGn",
            text=dc["densidad_x1000_hab"].round(2),
        )
        fig_dens.update_layout(
            height=400,
            xaxis_title="",
            yaxis_title="Emprendedores / 1,000 hab.",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_dens.update_xaxes(tickangle=45)
        fig_dens.update_traces(textposition="outside")
        st.plotly_chart(fig_dens, use_container_width=True)

    with col4:
        st.subheader("Género de emprendedores")
        sexo_counts = artesanos["sexo"].value_counts().reset_index()
        sexo_counts.columns = ["Género", "Cantidad"]
        fig_sexo = px.pie(
            sexo_counts,
            values="Cantidad",
            names="Género",
            hole=0.5,
            color_discrete_map={"femenino": "#ff6b81", "masculino": "#4ecdc4"},
        )
        fig_sexo.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_sexo.update_traces(textinfo="percent+value")
        st.plotly_chart(fig_sexo, use_container_width=True)

    st.divider()

    # ── Fila 3: Tipo emprendimiento + Estrato ─────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Tipos de emprendimiento")
        tipo_counts = (
            artesanos["tipo_de_emprendimiento"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        tipo_counts.columns = ["Tipo", "Cantidad"]
        fig_tipo = px.bar(
            tipo_counts,
            x="Cantidad",
            y="Tipo",
            orientation="h",
            color="Cantidad",
            color_continuous_scale="Viridis",
            text="Cantidad",
        )
        fig_tipo.update_layout(
            height=400,
            yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_tipo.update_traces(textposition="outside")
        st.plotly_chart(fig_tipo, use_container_width=True)

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
        st.plotly_chart(fig_estrato, use_container_width=True)

    st.divider()

    # ── Fila 4: Tráfico WiFi vs Emprendedores (scatter) ──────────────────────
    st.subheader("Tráfico WiFi vs Emprendedores por barrio")
    st.caption("Tamaño = usuarios WiFi | Color = zona geográfica — Los barrios en la esquina superior izquierda tienen mayor oportunidad (alto tráfico, pocos emprendedores)")

    # Merge flujo con densidad para scatter
    scatter_df = flujo_b.merge(
        densidad_b.rename(columns={"barrio": "barrio_norm"}),
        on="barrio_norm",
        how="inner",
    )
    scatter_df = scatter_df.merge(
        master[["barrio_norm", "total_usuarios"]],
        on="barrio_norm",
        how="left",
    )
    scatter_df["total_usuarios"] = scatter_df["total_usuarios"].fillna(1000)

    fig_scatter = px.scatter(
        scatter_df,
        x="n_emprendedores",
        y="total_registros",
        size="total_usuarios",
        color="zona",
        hover_name="barrio_norm",
        text="barrio_norm",
        size_max=50,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_scatter.update_layout(
        height=500,
        xaxis_title="Emprendedores registrados",
        yaxis_title="Conexiones WiFi totales",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=9)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Fila 5: Cabeza de hogar + Zona ────────────────────────────────────────
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
        st.plotly_chart(fig_cabeza, use_container_width=True)

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
        st.plotly_chart(fig_zona, use_container_width=True)


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

    if prompt := st.chat_input("Pregúntale a MedCity..."):
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
