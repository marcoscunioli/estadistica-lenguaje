import streamlit as st
import string
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from io import StringIO
import unicodedata
import re
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="CriptoAnalizador Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Frecuencias de diferentes idiomas
FRECUENCIAS_IDIOMAS = {
    'Español': {
        'A': 12.53, 'B': 1.42, 'C': 4.68, 'D': 5.86, 'E': 13.68,
        'F': 0.69, 'G': 1.01, 'H': 0.70, 'I': 6.25, 'J': 0.44,
        'K': 0.00, 'L': 4.97, 'M': 3.15, 'N': 6.71, 'Ñ': 0.31,
        'O': 8.68, 'P': 2.51, 'Q': 0.88, 'R': 6.87, 'S': 7.98,
        'T': 4.63, 'U': 3.93, 'V': 0.90, 'W': 0.02, 'X': 0.22,
        'Y': 0.90, 'Z': 0.52
    },
    'Inglés': {
        'A': 8.167, 'B': 1.492, 'C': 2.782, 'D': 4.253, 'E': 12.702,
        'F': 2.228, 'G': 2.015, 'H': 6.094, 'I': 6.966, 'J': 0.153,
        'K': 0.772, 'L': 4.025, 'M': 2.406, 'N': 6.749, 'O': 7.507,
        'P': 1.929, 'Q': 0.095, 'R': 5.987, 'S': 6.327, 'T': 9.056,
        'U': 2.758, 'V': 0.978, 'W': 2.360, 'X': 0.150, 'Y': 1.974, 'Z': 0.074
    },
    'Francés': {
        'A': 7.636, 'B': 0.901, 'C': 3.260, 'D': 3.669, 'E': 14.715,
        'F': 1.066, 'G': 0.866, 'H': 0.737, 'I': 7.529, 'J': 0.545,
        'K': 0.049, 'L': 5.456, 'M': 2.968, 'N': 7.095, 'O': 5.378,
        'P': 3.021, 'Q': 1.362, 'R': 6.553, 'S': 7.948, 'T': 7.244,
        'U': 6.311, 'V': 1.628, 'W': 0.114, 'X': 0.387, 'Y': 0.308, 'Z': 0.136
    }
}

# Función para normalizar texto (eliminar acentos y caracteres especiales)
def normalizar_texto(texto):
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r'[^A-ZÑ\s]', '', texto)  # Conservar solo letras y espacios
    return texto

# Función de cifrado César
def aplicar_cifrado_cesar(texto, desplazamiento):
    alfabeto = string.ascii_uppercase + 'Ñ'
    resultado = ""
    for c in texto:
        if c in alfabeto:
            idx = alfabeto.index(c)
            nuevo_idx = (idx + desplazamiento) % len(alfabeto)
            resultado += alfabeto[nuevo_idx]
        else:
            resultado += c
    return resultado

# Función para calcular índice de coincidencia
def indice_coincidencia(texto):
    n = len(texto)
    frecuencias = Counter(texto)
    return sum(f*(f-1) for f in frecuencias.values()) / (n*(n-1)) if n > 1 else 0

# Función para autodescifrar César por frecuencias
@st.cache_data
def descifrar_cesar_auto(texto_cifrado, frecuencias_esperadas):
    correlaciones = []
    for desplazamiento in range(27):
        texto_descifrado = aplicar_cifrado_cesar(texto_cifrado, -desplazamiento)
        frec_obs = Counter(c for c in texto_descifrado if c in frecuencias_esperadas)
        total = sum(frec_obs.values())
        if total == 0:
            continue
        correlacion = sum(
            (frec_obs.get(letra, 0)/total) * (frec_esperada/100)
            for letra, frec_esperada in frecuencias_esperadas.items()
        )
        correlaciones.append((desplazamiento, correlacion))
    
    mejor_desplazamiento = max(correlaciones, key=lambda x: x[1])[0]
    return mejor_desplazamiento

# Función para obtener n-gramas
@st.cache_data
def obtener_ngramas(texto_limpio, n=2, top_n=10):
    ngramas = Counter()
    for i in range(len(texto_limpio)-n+1):
        ngrama = texto_limpio[i:i+n]
        ngramas[ngrama] += 1
    return ngramas.most_common(top_n)

# Función para generar wordcloud
def generar_wordcloud(ngramas):
    if not ngramas:
        return None
    text = ' '.join(ngrama for ngrama, count in ngramas)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wc

# --- INTERFAZ DE USUARIO ---

# Panel lateral
with st.sidebar:
    st.title("🔍 CriptoAnalizador Pro")
    st.markdown("""
    ## 📚 Guía Rápida
    1. Ingresá texto o cargá un archivo
    2. Seleccioná el idioma de referencia
    3. Usá cifrado César para pruebas
    4. Explorá las estadísticas y gráficos
    5. ¡Compará con otros textos!
    """)
    
    st.divider()
    st.markdown("### ⚙️ Opciones Avanzadas")
    idioma_seleccionado = st.selectbox("Idioma de referencia", list(FRECUENCIAS_IDIOMAS.keys()))
    umbral_frecuencia = st.slider("Umbral mínimo de frecuencia (%)", 0.0, 5.0, 0.1)
    num_ngramas = st.slider("Número de n-gramas a mostrar", 5, 50, 10)
    
    st.divider()
    st.markdown("### 📊 Estadísticas")
    if 'historial' not in st.session_state:
        st.session_state.historial = []
    
    if st.button("💾 Guardar análisis actual", use_container_width=True):
        st.session_state.historial.append({
            'idioma': idioma_seleccionado,
            'texto': st.session_state.get('texto_actual', ''),
            'frecuencias': st.session_state.get('df_frecuencias', None)
        })
        st.success("Análisis guardado en historial!")
    
    if st.button("🧹 Limpiar historial", use_container_width=True):
        st.session_state.historial = []
        st.success("Historial limpiado!")
    
    if st.session_state.historial:
        st.markdown(f"📦 **Análisis guardados:** {len(st.session_state.historial)}")

# Contenido principal
st.title("🔍 CriptoAnalizador Pro - Análisis Estadístico de Textos")
st.caption("Herramienta profesional para criptoanálisis y estudio de frecuencias lingüísticas")

tab1, tab2, tab3 = st.tabs(["📝 Análisis Principal", "🆚 Comparativo", "ℹ️ Acerca de"])

with tab1:
    # Ingreso de texto
    col1, col2 = st.columns([3, 1])
    with col1:
        texto_entrada = st.text_area("**Ingresá tu texto:**", height=200, 
                                    placeholder="Pegá el texto a analizar aquí...",
                                    help="Texto a analizar. Puede ser plano o cifrado")
    with col2:
        st.markdown("**O cargá desde archivo**")
        archivo_subido = st.file_uploader("Subir archivo de texto (.txt)", type="txt", label_visibility="collapsed")
        if archivo_subido:
            texto_entrada = StringIO(archivo_subido.getvalue().decode("utf-8")).read()
        
        st.divider()
        st.markdown("**Preprocesamiento**")
        eliminar_no_alfabeticos = st.checkbox("Eliminar caracteres no alfabéticos", value=True)
        conservar_espacios = st.checkbox("Conservar espacios", value=True)
    
    if not texto_entrada:
        st.info("👋 ¡Bienvenido! Por favor, ingresá un texto o cargá un archivo para comenzar el análisis.")
        st.stop()
    
    # Normalización y preprocesamiento
    try:
        texto_mayus = texto_entrada.upper()
        if eliminar_no_alfabeticos:
            if conservar_espacios:
                texto_mayus = re.sub(r'[^A-ZÑ\s]', '', texto_mayus)
            else:
                texto_mayus = re.sub(r'[^A-ZÑ]', '', texto_mayus)
        texto_mayus = normalizar_texto(texto_mayus)
        st.session_state.texto_actual = texto_mayus
    except Exception as e:
        st.error(f"Error en procesamiento de texto: {str(e)}")
        st.stop()
    
    # Cifrado César
    st.subheader("🔐 Cifrado César", help="Aplica un cifrado/descifrado César al texto")
    cesar_col1, cesar_col2, cesar_col3 = st.columns([1, 2, 1])
    
    with cesar_col1:
        aplicar_cesar = st.checkbox("Aplicar cifrado César", value=False)
    
    with cesar_col2:
        if aplicar_cesar:
            modo = st.radio("Modo", ["Cifrar", "Descifrar"], horizontal=True)
            desplazamiento = st.slider("Desplazamiento", 1, 26, 3)
    
    with cesar_col3:
        auto_descifrar = st.checkbox("Autodescifrar", value=False, 
                                    help="Intenta descifrar automáticamente usando análisis de frecuencias")
    
    if auto_descifrar:
        with st.spinner("Buscando mejor desplazamiento..."):
            mejor_despl = descifrar_cesar_auto(texto_mayus, FRECUENCIAS_IDIOMAS[idioma_seleccionado])
        st.success(f"Desplazamiento probable: {mejor_despl}")
        texto_mayus = aplicar_cifrado_cesar(texto_mayus, -mejor_despl)
        desplazamiento = mejor_despl
        aplicar_cesar = True
        modo = "Descifrar"
    
    if aplicar_cesar:
        operacion = desplazamiento if modo == "Cifrar" else -desplazamiento
        texto_mayus = aplicar_cifrado_cesar(texto_mayus, operacion)
        st.text_area("**Texto transformado:**", value=texto_mayus, height=150)
    
    # Procesar letras
    alfabeto = string.ascii_uppercase + 'Ñ'
    solo_letras = ''.join([c for c in texto_mayus if c in alfabeto])
    total_caracteres = len(solo_letras)
    
    if total_caracteres < 50:
        st.warning("⚠️ El texto es muy corto para un análisis significativo. Se recomiendan al menos 100 caracteres.")
    
    conteo = Counter(solo_letras)
    total = sum(conteo.values()) if total_caracteres > 0 else 1
    
    frecuencia_calculada = {
        letra: (conteo.get(letra, 0) / total) * 100 for letra in FRECUENCIAS_IDIOMAS[idioma_seleccionado].keys()
    }
    
    # Calcular índice de coincidencia
    ic = indice_coincidencia(solo_letras)
    ic_esperado = 0.0775 if idioma_seleccionado == "Español" else 0.0667
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Letra': list(FRECUENCIAS_IDIOMAS[idioma_seleccionado].keys()),
        'Frec. Esperada (%)': list(FRECUENCIAS_IDIOMAS[idioma_seleccionado].values()),
        'Frec. Observada (%)': [frecuencia_calculada.get(l, 0) for l in FRECUENCIAS_IDIOMAS[idioma_seleccionado].keys()]
    })
    
    # Filtrar por umbral mínimo
    df_filtrado = df[df['Frec. Observada (%)'] > umbral_frecuencia]
    st.session_state.df_frecuencias = df
    
    # Estadísticas básicas
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    col_stats1.metric("Total caracteres", f"{total_caracteres:,}")
    col_stats2.metric("Letras únicas", len(conteo))
    col_stats3.metric("Índice de coincidencia", f"{ic:.4f}", 
                     f"{(ic/ic_esperado*100-100):+.1f}% vs esperado")
    col_stats4.metric("Entropía aproximada", f"{np.log2(len(conteo)):.2f} bits" if conteo else "N/A")
    
    # Gráfico de frecuencias con Plotly
    st.subheader("📈 Análisis de Frecuencias")
    fig = go.Figure(data=[
        go.Bar(name='Observada', x=df['Letra'], y=df['Frec. Observada (%)'],
               marker_color='#1f77b4', opacity=0.8),
        go.Scatter(name='Esperada', x=df['Letra'], y=df['Frec. Esperada (%)'],
                  mode='lines+markers', line=dict(color='firebrick', width=2))
    ])
    
    fig.update_layout(
        barmode='group',
        title='Distribución de Frecuencias de Letras',
        xaxis_title="Letra",
        yaxis_title="Frecuencia (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de datos
    st.dataframe(df.style.format({'Frec. Esperada (%)': '{:.2f}', 
                                'Frec. Observada (%)': '{:.2f}'}).background_gradient(
        subset=['Frec. Observada (%)'], cmap='Blues'), 
        height=400, use_container_width=True)
    
    # Botones de exportación
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", data=csv, 
                      file_name="frecuencias_lenguaje.csv", mime="text/csv",
                      use_container_width=True)
    
    # Análisis de n-gramas
    st.subheader("🔍 Análisis de N-Gramas")
    
    if st.checkbox("Mostrar bigramas y trigramas"):
        col_bg, col_tg, col_wc = st.columns([2, 2, 3])
        
        with col_bg:
            st.markdown("**🔸 Top Bigramas**")
            bigramas = obtener_ngramas(solo_letras, 2, num_ngramas)
            for b, f in bigramas:
                st.code(f"{b}: {f} ocurrencias ({f/(total_caracteres-1)*100:.2f}%)")
        
        with col_tg:
            st.markdown("**🔹 Top Trigramas**")
            trigramas = obtener_ngramas(solo_letras, 3, num_ngramas)
            for t, f in trigramas:
                st.code(f"{t}: {f} ocurrencias ({f/(total_caracteres-2)*100:.2f}%)")
        
        with col_wc:
            st.markdown("**☁️ Nube de Palabras de Bigramas**")
            if bigramas:
                wc = generar_wordcloud(bigramas)
                if wc:
                    st.image(wc.to_array(), use_column_width=True)
                else:
                    st.warning("No hay suficientes bigramas para generar nube")
            else:
                st.info("No hay bigramas disponibles")
    
    # Análisis avanzado
    with st.expander("🧠 Análisis Avanzado"):
        st.markdown("**Correlación de frecuencias**")
        
        # Calcular correlación
        correlacion = np.corrcoef(
            list(FRECUENCIAS_IDIOMAS[idioma_seleccionado].values()),
            df['Frec. Observada (%)']
        )[0, 1]
        
        st.metric("Correlación con idioma de referencia", f"{correlacion:.4f}")
        
        # Gráfico de dispersión (CORRECCIÓN APLICADA AQUÍ)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=df['Frec. Esperada (%)'],
            y=df['Frec. Observada (%)'],
            mode='markers+text',
            text=df['Letra'],
            marker=dict(size=12, color='royalblue')
        ))  # Paréntesis corregido
        
        # Línea de referencia
        fig_scatter.add_trace(go.Scatter(
            x=[0, max(df['Frec. Esperada (%)'])],
            y=[0, max(df['Frec. Esperada (%)'])],
            mode='lines',
            line=dict(color='firebrick', dash='dash'),
            name='Referencia'
        ))
        
        fig_scatter.update_layout(
            title='Frecuencia Observada vs Esperada',
            xaxis_title='Frecuencia Esperada (%)',
            yaxis_title='Frecuencia Observada (%)',
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("🆚 Análisis Comparativo entre Textos")
    
    if not st.session_state.historial:
        st.info("No hay análisis guardados en el historial. Realiza un análisis primero y guárdalo.")
        st.stop()
    
    st.markdown("### Seleccionar análisis para comparar")
    historial_opciones = [f"Análisis {i+1} ({len(item['texto'])} chars, {item['idioma']})" 
                         for i, item in enumerate(st.session_state.historial)]
    
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        seleccion1 = st.selectbox("Primer análisis", historial_opciones, index=len(historial_opciones)-1)
        idx1 = historial_opciones.index(seleccion1)
        df1 = st.session_state.historial[idx1]['frecuencias']
    
    with col_comp2:
        seleccion2 = st.selectbox("Segundo análisis", historial_opciones, index=max(0, len(historial_opciones)-2))
        idx2 = historial_opciones.index(seleccion2)
        df2 = st.session_state.historial[idx2]['frecuencias']
    
    # Combinar datos
    df_comparativo = pd.merge(
        df1, df2, 
        on='Letra', 
        suffixes=(f' ({seleccion1})', f' ({seleccion2})')
    )
    
    # Gráfico comparativo
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=df_comparativo['Letra'],
        y=df_comparativo['Frec. Observada (%)_x'],
        name=seleccion1,
        marker_color='#1f77b4'
    ))
    fig_comp.add_trace(go.Bar(
        x=df_comparativo['Letra'],
        y=df_comparativo['Frec. Observada (%)_y'],
        name=seleccion2,
        marker_color='#ff7f0e'
    ))
    
    fig_comp.update_layout(
        barmode='group',
        title='Comparación de Frecuencias',
        xaxis_title="Letra",
        yaxis_title="Frecuencia Observada (%)",
        height=500
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Tabla comparativa
    st.dataframe(df_comparativo.style.format({
        'Frec. Observada (%)_x': '{:.2f}', 
        'Frec. Observada (%)_y': '{:.2f}'
    }).background_gradient(), height=400, use_container_width=True)

with tab3:
    st.header("ℹ️ Acerca de CriptoAnalizador Pro")
    st.markdown("""
    **CriptoAnalizador Pro** es una herramienta avanzada para el análisis estadístico de textos 
    con aplicaciones en criptografía, lingüística y procesamiento de lenguaje natural.
    
    ### Características principales:
    - Análisis de frecuencias de letras (monogramas)
    - Detección de bigramas y trigramas más comunes
    - Cifrado/descifrado César interactivo
    - Autodescifrado mediante análisis estadístico
    - Comparación entre múltiples textos
    - Visualizaciones profesionales e interactivas
    
    ### Métricas calculadas:
    - **Índice de coincidencia**: Mide la probabilidad de que dos letras aleatorias sean iguales
    - **Correlación de frecuencias**: Compara distribución con idioma de referencia
    - **Entropía aproximada**: Mide la aleatoriedad del texto
    
    ### Idiomas soportados:
    - Español, Inglés, Francés
    
    ### Tecnologías utilizadas:
    - Python 3.10+
    - Streamlit (interfaz web)
    - Plotly (visualizaciones interactivas)
    - Pandas (procesamiento de datos)
    
    *Desarrollado por [Tu Nombre] | [Año Actual]*
    """)
    
    st.divider()
    st.markdown("### 📚 Recursos de aprendizaje")
    st.markdown("""
    - [Criptografía clásica - Wikipedia](https://es.wikipedia.org/wiki/Criptograf%C3%ADa_cl%C3%A1sica)
    - [Análisis de frecuencias aplicado a criptoanálisis](https://www.nku.edu/~christensen/1402%20frequency%20analysis.pdf)
    - [Fundamentos matemáticos de la criptografía](https://math.uchicago.edu/~may/REU2020/REUPapers/Adhikari,Anish.pdf)
    """)