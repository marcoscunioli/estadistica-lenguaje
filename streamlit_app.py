import streamlit as st
import string
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
from io import StringIO
import unicodedata
import re
from wordcloud import WordCloud

# --- CONFIGURACIÓN DE PÁGINA Y CONSTANTES ---

st.set_page_config(
    page_title="CriptoAnalizador Definitivo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Frecuencias de letras estándar para diferentes idiomas.
FRECUENCIAS_IDIOMAS = {
    'Español': {
        'A': 12.53, 'B': 1.42, 'C': 4.68, 'D': 5.86, 'E': 13.68, 'F': 0.69, 
        'G': 1.01, 'H': 0.70, 'I': 6.25, 'J': 0.44, 'K': 0.01, 'L': 4.97, 
        'M': 3.15, 'N': 6.71, 'Ñ': 0.31, 'O': 8.68, 'P': 2.51, 'Q': 0.88, 
        'R': 6.87, 'S': 7.98, 'T': 4.63, 'U': 3.93, 'V': 0.90, 'W': 0.02, 
        'X': 0.22, 'Y': 0.90, 'Z': 0.52
    },
    'Inglés': {
        'A': 8.167, 'B': 1.492, 'C': 2.782, 'D': 4.253, 'E': 12.702, 'F': 2.228, 
        'G': 2.015, 'H': 6.094, 'I': 6.966, 'J': 0.153, 'K': 0.772, 'L': 4.025, 
        'M': 2.406, 'N': 6.749, 'O': 7.507, 'P': 1.929, 'Q': 0.095, 'R': 5.987, 
        'S': 6.327, 'T': 9.056, 'U': 2.758, 'V': 0.978, 'W': 2.360, 'X': 0.150, 
        'Y': 1.974, 'Z': 0.074
    },
    'Francés': {
        'A': 7.636, 'B': 0.901, 'C': 3.260, 'D': 3.669, 'E': 14.715, 'F': 1.066, 
        'G': 0.866, 'H': 0.737, 'I': 7.529, 'J': 0.545, 'K': 0.049, 'L': 5.456, 
        'M': 2.968, 'N': 7.095, 'O': 5.378, 'P': 3.021, 'Q': 1.362, 'R': 6.553, 
        'S': 7.948, 'T': 7.244, 'U': 6.311, 'V': 1.628, 'W': 0.114, 'X': 0.387, 
        'Y': 0.308, 'Z': 0.136
    }
}

# --- FUNCIONES DE PROCESAMIENTO Y CRIPTOGRAFÍA ---

def get_alphabet_from_lang(lang: str) -> str:
    """Obtiene la cadena del alfabeto para un idioma dado."""
    return "".join(FRECUENCIAS_IDIOMAS[lang].keys())

def normalizar_texto(texto: str) -> str:
    """Elimina acentos y convierte a mayúsculas."""
    texto_sin_acentos = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto_sin_acentos.upper()

def limpiar_texto(texto: str, alfabeto: str) -> str:
    """Filtra el texto para que contenga solo caracteres del alfabeto dado."""
    return "".join([char for char in normalizar_texto(texto) if char in alfabeto])

@st.cache_data
def aplicar_cifrado_cesar(texto: str, desplazamiento: int, alfabeto: str) -> str:
    """Aplica el cifrado César a un texto, respetando los caracteres no alfabéticos."""
    resultado = ""
    texto_norm = normalizar_texto(texto)
    for char in texto_norm:
        if char in alfabeto:
            idx = alfabeto.index(char)
            nuevo_idx = (idx + desplazamiento) % len(alfabeto)
            resultado += alfabeto[nuevo_idx]
        else:
            resultado += char
    return resultado

# --- FUNCIONES DE ANÁLISIS ESTADÍSTICO ---

@st.cache_data
def obtener_n_gramas(texto_limpio: str, n: int, top_n: int) -> list:
    """Calcula los n-gramas más comunes."""
    if len(texto_limpio) < n:
        return []
    n_gramas = Counter(texto_limpio[i:i+n] for i in range(len(texto_limpio) - n + 1))
    return n_gramas.most_common(top_n)

def calcular_ic(texto_limpio: str) -> float:
    """Calcula el Índice de Coincidencia (IC)."""
    n = len(texto_limpio)
    if n < 2: return 0.0
    conteo = Counter(texto_limpio)
    numerador = sum(f * (f - 1) for f in conteo.values())
    denominador = n * (n - 1)
    return numerador / denominador if denominador > 0 else 0.0

@st.cache_data
def calcular_chi_cuadrado(texto: str, freqs_esperadas: dict) -> float:
    """Calcula la puntuación de Chi-cuadrado del texto contra frecuencias esperadas."""
    n = len(texto)
    if n == 0: return float('inf')
    conteo_obs = Counter(texto)
    chi_cuadrado = 0.0
    for letra, frec_esp_porc in freqs_esperadas.items():
        conteo_esp = (frec_esp_porc / 100) * n
        diferencia = conteo_obs.get(letra, 0) - conteo_esp
        chi_cuadrado += (diferencia ** 2) / (conteo_esp if conteo_esp > 0 else 1)
    return chi_cuadrado

@st.cache_data
def crack_cesar(texto_cifrado: str, idioma: str) -> dict:
    """Descifra un texto con César probando todos los desplazamientos y usando Chi-cuadrado."""
    alfabeto = get_alphabet_from_lang(idioma)
    texto_limpio = limpiar_texto(texto_cifrado, alfabeto)
    freqs_esperadas = FRECUENCIAS_IDIOMAS[idioma]
    
    mejor_resultado = {'desplazamiento': -1, 'puntuacion': float('inf'), 'texto': ''}

    for d in range(len(alfabeto)):
        texto_prueba = aplicar_cifrado_cesar(texto_limpio, -d, alfabeto)
        puntuacion = calcular_chi_cuadrado(texto_prueba, freqs_esperadas)
        if puntuacion < mejor_resultado['puntuacion']:
            mejor_resultado = {
                'desplazamiento': d,
                'puntuacion': puntuacion,
                'texto': aplicar_cifrado_cesar(texto_cifrado, -d, alfabeto)
            }
    return mejor_resultado

# --- FUNCIONES DE VISUALIZACIÓN ---

def generar_wordcloud(ngramas: list):
    """Genera una imagen de nube de palabras a partir de n-gramas."""
    if not ngramas: return None
    freq_dict = {ngrama: count for ngrama, count in ngramas}
    wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate_from_frequencies(freq_dict)
    return wc.to_array()

def plot_frecuencias(df: pd.DataFrame) -> go.Figure:
    """Crea un gráfico de Plotly para comparar frecuencias observadas y esperadas."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Letra'], y=df['Observada (%)'], name='Observada',
        marker_color='#1f77b4', opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=df['Letra'], y=df['Esperada (%)'], name='Esperada',
        mode='lines+markers', line=dict(color='firebrick', width=2, dash='dash')
    ))
    fig.update_layout(
        title='Distribución de Frecuencias de Letras',
        xaxis_title="Letra", yaxis_title="Frecuencia (%)",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- INTERFAZ DE USUARIO (STREAMLIT) ---

def main():
    """Función principal que construye y ejecuta la aplicación Streamlit."""
    
    # --- PANEL LATERAL DE CONTROLES ---
    with st.sidebar:
        st.title("🛡️ CriptoAnalizador Definitivo")
        st.markdown("Herramienta todo-en-uno para análisis de textos y criptografía clásica.")
        
        idioma_seleccionado = st.selectbox(
            "Idioma de referencia", 
            list(FRECUENCIAS_IDIOMAS.keys()),
            help="Elige el idioma para comparar las frecuencias de letras."
        )
        alfabeto = get_alphabet_from_lang(idioma_seleccionado)

        st.divider()
        st.markdown("### 📥 Entrada de Texto")
        texto_entrada = st.text_area("Pega el texto aquí:", height=150, placeholder="Escribe o pega tu texto...")
        archivo_subido = st.file_uploader("O carga un archivo .txt", type="txt")
        if archivo_subido:
            try:
                texto_entrada = StringIO(archivo_subido.getvalue().decode("utf-8")).read()
                st.success("Archivo cargado.")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
                return
        
        st.divider()
        st.markdown("### ⚙️ Opciones de Análisis")
        num_ngramas = st.slider("Top N-gramas a mostrar", 5, 20, 10)

    # --- LÓGICA PRINCIPAL ---
    if not texto_entrada:
        st.info("👋 ¡Bienvenido! Ingresa texto en el panel lateral para comenzar.")
        return

    texto_limpio = limpiar_texto(texto_entrada, alfabeto)
    
    # Inicializar estado de sesión para el historial
    if 'historial' not in st.session_state:
        st.session_state.historial = []

    # --- PESTAÑAS DE VISUALIZACIÓN ---
    tab_analisis, tab_crypto, tab_comparativo, tab_acerca = st.tabs([
        "📊 Análisis Estadístico", "🔐 Herramientas Criptográficas", "🆚 Comparador", "ℹ️ Acerca de"
    ])

    with tab_analisis:
        st.header("Análisis Estadístico del Texto")
        
        if len(texto_limpio) < 20:
            st.warning("El texto es muy corto para un análisis estadístico fiable.")
            return
            
        # --- Métricas Clave ---
        conteo = Counter(texto_limpio)
        total_letras = len(texto_limpio)
        ic_calculado = calcular_ic(texto_limpio)
        correlacion = np.corrcoef(
            list(FRECUENCIAS_IDIOMAS[idioma_seleccionado].values()),
            [(conteo.get(l, 0) / total_letras) * 100 for l in alfabeto]
        )[0, 1]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total de Letras", f"{total_letras:,}")
        m2.metric("Índice de Coincidencia", f"{ic_calculado:.4f}")
        m3.metric("Correlación", f"{correlacion:.4f}", help="Correlación de Pearson con las frecuencias del idioma de referencia.")
        m4.metric("Letras Únicas", len(conteo))
        
        st.text_area("Texto Limpio Analizado", value=texto_limpio, height=100, disabled=True)

        # --- Análisis de Frecuencias ---
        st.subheader("Frecuencia de Letras (Monogramas)")
        frec_observada = {letra: (conteo.get(letra, 0) / total_letras) * 100 for letra in alfabeto}
        df_frec = pd.DataFrame({
            'Letra': list(alfabeto),
            'Esperada (%)': [FRECUENCIAS_IDIOMAS[idioma_seleccionado].get(l, 0) for l in alfabeto],
            'Observada (%)': [frec_observada.get(l, 0) for l in alfabeto],
            'Conteo': [conteo.get(l, 0) for l in alfabeto]
        })
        
        st.plotly_chart(plot_frecuencias(df_frec), use_container_width=True)
        with st.expander("Ver tabla de frecuencias detallada"):
            st.dataframe(df_frec.style.format({'Esperada (%)': '{:.2f}', 'Observada (%)': '{:.2f}'}), use_container_width=True)

        # --- Análisis de N-gramas y WordCloud ---
        st.subheader("Análisis de N-gramas")
        c1, c2 = st.columns([1, 2])
        with c1:
            tipo_ngrama = st.radio("Mostrar N-gramas:", ["Bigramas", "Trigramas"], horizontal=True)
            n = 2 if tipo_ngrama == "Bigramas" else 3
            ngramas = obtener_n_gramas(texto_limpio, n, num_ngramas)
            df_ngrama = pd.DataFrame(ngramas, columns=[tipo_ngrama, 'Frecuencia'])
            st.dataframe(df_ngrama, use_container_width=True)
        with c2:
            st.markdown(f"**☁️ Nube de {tipo_ngrama}**")
            wc_image = generar_wordcloud(ngramas)
            if wc_image is not None:
                st.image(wc_image, use_column_width=True)
            else:
                st.info("No hay suficientes datos para generar la nube de palabras.")

        # --- Guardar en Historial ---
        st.divider()
        if st.button("💾 Guardar este análisis en el historial", use_container_width=True):
            st.session_state.historial.append({
                'nombre': f"Análisis {len(st.session_state.historial) + 1} ({idioma_seleccionado}, {total_letras} letras)",
                'df': df_frec,
                'texto_limpio': texto_limpio
            })
            st.success("Análisis guardado.")

    with tab_crypto:
        st.header("Herramientas de Criptografía Clásica")
        
        # --- Cifrador/Descifrador César Interactivo ---
        st.subheader("🔐 Cifrado César Manual")
        c1, c2 = st.columns(2)
        modo = c1.radio("Modo:", ["Cifrar", "Descifrar"], horizontal=True)
        desplazamiento = c2.slider("Desplazamiento", 1, len(alfabeto) - 1, 3)
        
        op_despl = desplazamiento if modo == "Cifrar" else -desplazamiento
        texto_transformado = aplicar_cifrado_cesar(texto_entrada, op_despl, alfabeto)
        st.text_area("Texto Resultante:", value=texto_transformado, height=150)

        # --- Descifrador Automático ---
        st.subheader("🤖 Descifrador Automático (César Cracker)")
        st.markdown("Usa el test de **Chi-cuadrado** para encontrar el desplazamiento más probable.")
        if st.button("💥 ¡Crackear texto de entrada!", use_container_width=True):
            with st.spinner(f"Analizando texto contra el idioma '{idioma_seleccionado}'..."):
                resultado_crack = crack_cesar(texto_entrada, idioma_seleccionado)
            st.success(f"Análisis completo. Desplazamiento más probable: **{resultado_crack['desplazamiento']}**")
            st.metric("Puntuación Chi² (menor es mejor)", f"{resultado_crack['puntuacion']:.2f}")
            st.text_area("Texto probablemente descifrado:", value=resultado_crack['texto'], height=200)

    with tab_comparativo:
        st.header("Comparador de Análisis")
        if len(st.session_state.historial) < 2:
            st.info("Necesitas guardar al menos dos análisis en el historial para poder comparar.")
        else:
            opciones = [item['nombre'] for item in st.session_state.historial]
            c1, c2 = st.columns(2)
            sel1 = c1.selectbox("Selecciona el primer análisis:", opciones, index=len(opciones)-1)
            sel2 = c2.selectbox("Selecciona el segundo análisis:", opciones, index=len(opciones)-2)

            idx1 = opciones.index(sel1)
            idx2 = opciones.index(sel2)
            df1 = st.session_state.historial[idx1]['df']
            df2 = st.session_state.historial[idx2]['df']

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=df1['Letra'], y=df1['Observada (%)'], name=sel1))
            fig_comp.add_trace(go.Bar(x=df2['Letra'], y=df2['Observada (%)'], name=sel2))
            fig_comp.update_layout(barmode='group', title='Comparación de Frecuencias Observadas')
            st.plotly_chart(fig_comp, use_container_width=True)

            if c1.button("🧹 Limpiar historial", use_container_width=True):
                st.session_state.historial = []
                st.experimental_rerun()

    with tab_acerca:
        st.header("ℹ️ Acerca de CriptoAnalizador Definitivo")
        st.markdown("""
        Esta herramienta es el resultado de la fusión y mejora de varios scripts de análisis de texto, 
        combinando las mejores características para crear una suite de criptoanálisis educativa y potente.

        **Características Clave:**
        - **Análisis Multi-idioma:** Compara frecuencias contra Español, Inglés y Francés.
        - **Métricas Estadísticas:** Calcula Índice de Coincidencia, Correlación y Chi-cuadrado.
        - **Visualizaciones Interactivas:** Utiliza Plotly para gráficos dinámicos y claros.
        - **Herramientas Criptográficas:** Incluye cifrado César y un descifrador automático.
        - **Análisis Comparativo:** Guarda y compara diferentes textos para encontrar patrones.

        **Tecnologías:** Python, Streamlit, Pandas, Plotly, WordCloud.

        *Versión 1.0 - Agosto 2025*
        """)

if __name__ == "__main__":
    main()

