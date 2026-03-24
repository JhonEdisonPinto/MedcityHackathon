# MedCity Dashboard — Ideas por Capas de Solución

**Hackathon Medellín · Colombia 5.0 / Smart Cities**
Reto: *MedCity Dashboard: Datos Abiertos para Decisiones que Transforman*

---

## Contexto

Dashboard interactivo + IA para caracterizar emprendimientos, detectar oportunidades y recomendar tipo de negocio por zona, usando 5 datasets de Medata. Equipo pequeño, ~4 horas de implementación.

**Datasets disponibles:**
- `creditos_otorgados_a_microempresarios.csv` → edad, actividad, barrio, comuna, monto, fecha
- `registro_artesano_y_producto_formado_y_cualificados_en_diseno.csv` → edad, sexo, tipo de emprendimiento, barrio, comuna, zona, estrato
- `cantidad_de_usuarios_por_sitio_dia.csv` → sede, día, registros, mes, año
- `segmentacion_de_usuarios.csv` → rango de edad, género, tipo de dispositivo

---

## Capa 1 — Ingesta y preparación de datos

| Idea | Dato / Herramienta | Valor |
|------|-------------------|-------|
| Crear tabla maestra de barrios/comunas normalizando nombres (tildes, mayúsculas, abreviaciones) | pandas + diccionario de equivalencias manual | Permite hacer JOIN entre los 5 datasets sin pérdida de registros |
| Enriquecer con GeoJSON de comunas y barrios de Medellín | GeoMedellín o Medata (shapefile comunas) | Habilita visualizaciones cartográficas sin geocodificación manual |
| Tabla unificada de emprendedores por barrio: unir los 3 datasets de créditos + artesanos | pandas merge/concat por barrio + deduplicar por perfil | Fuente única de "quién emprende dónde" |
| Agregar usuarios wifi de diario a mensual/semanal por sede | pandas groupby(sede, mes) | Reduce ruido y facilita comparaciones de tendencia |
| Asignar barrio a cada sede wifi usando tabla de sedes del dataset | lookup manual o fuzzy matching | Permite cruzar flujo de personas con emprendimientos del mismo barrio |
| Calcular columna `tipo_emprendimiento_normalizado` agrupando categorías similares | mapeo manual a 8-10 rubros grandes (alimentos, belleza, tecnología, comercio, artesanía, etc.) | Reduce fragmentación de categorías y mejora el análisis |

---

## Capa 2 — Lógica de caracterización de zonas

| Indicador | Cálculo | Valor |
|-----------|---------|-------|
| **Densidad emprendedora** por barrio | # emprendedores / estimado poblacional por comuna (DANE) | Identifica zonas de alta concentración vs. zonas vacías |
| **Tipo dominante de emprendimiento** por barrio/comuna | moda del `tipo_emprendimiento_normalizado` | Perfil productivo de la zona |
| **Índice de diversificación** de negocios | tipos únicos / total emprendimientos por barrio | Barrios mono-sector vs. ecosistemas mixtos |
| **Capacidad financiera promedio** | media de `monto_credito` por barrio | Proxy del músculo financiero del emprendedor local |
| **Perfil demográfico del emprendedor** | distribución de edad y sexo por barrio | Entender quién emprende en cada zona |
| **Flujo de usuarios wifi** | promedio de registros diarios por sede → agregado por barrio | Proxy de demanda/tráfico de consumidores potenciales |
| **Perfil del consumidor wifi** | distribución edad + dispositivo (`segmentacion_de_usuarios`) por sede/barrio | ¿Jóvenes con smartphone o adultos con PC? Orienta el tipo de negocio a recomendar |

---

## Capa 3 — Lógica de detección de oportunidades

| Idea | Cómo implementarla | Valor |
|------|-------------------|-------|
| **Vacío de oferta**: alto tráfico wifi pero pocos emprendimientos | Ratio `usuarios_wifi / num_emprendedores` por barrio; outliers positivos = oportunidad | Barrios con demanda sin oferta suficiente |
| **Saturación de sector**: >60% de emprendimientos del mismo tipo | Share por tipo en cada barrio; umbral configurable | Detectar rubros sobreofertados donde no conviene entrar |
| **Mismatch demográfico**: perfil usuario wifi ≠ perfil emprendedor | Comparar `rango_edad` de usuarios wifi vs. emprendedores por barrio | Si los usuarios son jóvenes y los negocios son para adultos → hay un gap |
| **Brecha de acceso a crédito**: montos bajos en zonas de alto tráfico | `monto_credito_promedio < percentil 25` AND `usuarios_wifi > percentil 75` | Zonas con potencial pero poco capital → recomendar acceso a crédito |
| **Zonas sin artesanos con alto tráfico** | Sedes wifi con alto tráfico + 0 artesanos registrados en el barrio | Oportunidad artesanal / cultural |
| **Score de oportunidad por barrio** | Suma ponderada: `(1 - saturación) + vacío_oferta + mismatch_demográfico` | Ranking único que ordena barrios por atractivo para nuevo emprendimiento |

---

## Capa 4 — Visualizaciones del dashboard

| Visualización | Herramienta | Qué muestra / Por qué es impactante |
|--------------|-------------|-------------------------------------|
| **Mapa coroplético de comunas** (color = score de oportunidad) | Folium + GeoJSON de comunas | Primera pantalla del dashboard; vista rápida de dónde hay más actividad |
| **Puntos de sedes wifi** con círculos proporcionales al tráfico | Folium MarkerCluster | Zonas de alta demanda potencial superpuestas al mapa |
| **Radar chart por zona** con 5 ejes: emprendimientos, créditos, tráfico wifi, diversificación, oportunidad | Plotly | Perfil completo de un barrio en una sola figura |
| **Top 5 tipos de emprendimiento por barrio** (barras horizontales) | Plotly Express | ¿Qué negocios dominan en la zona seleccionada? |
| **Tabla de oportunidades** ordenada por score con columna "razón" | Streamlit st.dataframe | Accionable directo para el emprendedor |
| **Histograma de créditos** por estrato o tipo de emprendimiento | Plotly Express | Contexto financiero del ecosistema |
| **Filtros interactivos** en sidebar: barrio, comuna, tipo de emprendimiento, rango de fecha | Streamlit sidebar widgets | Permite al usuario explorar su propio contexto |
| **4 tarjetas KPI** en la parte superior: total emprendedores, monto total créditos, sedes activas, tráfico promedio | `st.metric` de Streamlit | Resumen ejecutivo visual inmediato |

---

## Capa 5 — Flujo conversacional del chatbot

### Principio fundamental
El chatbot usa **RAG sobre datos reales de Medata**. El modelo nunca inventa cifras; cada respuesta cita la fuente.

### Flujo de 3 intenciones

```
1. "¿Qué quieres saber?"
   ├── Caracterizar una zona
   ├── Encontrar oportunidad
   └── Qué negocio abrir

2a. [Caracterizar zona]     → "¿En qué barrio o comuna?" → indicadores del barrio
2b. [Encontrar oportunidad] → "¿Tienes tipo de negocio en mente?" → filtra por saturación
2c. [Qué negocio abrir]     → "¿En qué zona quieres ubicarte?" → top 3 rubros recomendados

3. Respuesta siempre incluye:
   - Dato concreto     → "En Aranjuez hay 47 emprendimientos, 62% en alimentos"
   - Fuente explícita  → "Según Medata – creditos_microempresarios 2023"
   - Recomendación     → "El sector belleza está subrepresentado; solo 3% de los negocios"
```

### Implementación técnica del RAG

1. Convertir filas procesadas a documentos de texto por barrio:
   *"En el barrio X, comuna Y, hay Z emprendedores, sectores dominantes A y B, crédito promedio $C..."*
2. Vectorizar con `sentence-transformers` (modelo `all-MiniLM-L6-v2`, liviano, funciona offline)
3. Vector store: **ChromaDB local** (sin API key, sin infraestructura)
4. LLM solo sintetiza la respuesta; no completa datos faltantes

### 3 demos a preparar para la presentación

1. *"¿Qué tipo de negocio me recomiendas abrir en el barrio Boston?"*
2. *"¿Cuáles son los barrios con más tráfico wifi pero menos emprendimientos?"*
3. *"¿Qué perfil tiene el emprendedor típico de la comuna 13?"*

---

## Capa 6 — Stack tecnológico

### Backend / Procesamiento

| Componente | Herramienta | Razón |
|-----------|-------------|-------|
| Limpieza y análisis | **pandas + numpy** | Estándar, rápido de implementar |
| Geodatos | **geopandas + shapely** | Joins espaciales y exportación a GeoJSON |
| Almacenamiento | **Parquet o SQLite** | Más rápido que CSV para queries repetidas en el dashboard |

### Dashboard

| Componente | Herramienta | Razón |
|-----------|-------------|-------|
| Framework principal | **Streamlit** | Prototipo funcional en horas, sin frontend separado |
| Mapas | **Folium** + `streamlit-folium` | Mapas interactivos con GeoJSON, sin servidor de tiles propio |
| Gráficos | **Plotly Express** | Interactivo, integra nativo con Streamlit |

### Chatbot / IA

| Componente | Herramienta | Razón |
|-----------|-------------|-------|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Liviano, funciona offline |
| Vector store | **ChromaDB local** | Sin infraestructura ni costo |
| LLM síntesis | **Groq Api** (`openai/gpt-oss-120b `) | económico|
| Orquestación | **LangChain (LangGraph)**  | LangChain  |

### Datos geográficos complementarios
- GeoJSON de comunas de Medellín: **GeoMedellín** (`geomedellín.gov.co`) o Medata
- Tabla barrio → comuna → coordenada centroide: construible desde el GeoJSON en ~30 min

### Despliegue para la demo
- **Streamlit Cloud** (gratis, deploy en minutos desde GitHub) o correr local durante el hackathon

---

## Orden de implementación sugerido (4 horas)

| Tiempo | Tarea |
|--------|-------|
| 30 min | Limpieza y unificación de datasets → tabla maestra por barrio |
| 20 min | Cálculo de indicadores clave (densidad, tipo dominante, tráfico wifi) |
| 30 min | Score de oportunidad por barrio |
| 60 min | Dashboard Streamlit: mapa + filtros + tabla de oportunidades + 2-3 gráficos |
| 60 min | Chatbot RAG: indexar datos + flujo conversacional de 3 preguntas demo |
| 30 min | Integrar chatbot en el sidebar del dashboard |
| 10 min | Deploy en Streamlit Cloud + preparar demo |
