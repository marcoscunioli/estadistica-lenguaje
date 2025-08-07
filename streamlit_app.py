import streamlit as st
import string
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from io import StringIO

# --- CONSTANTES Y CONFIGURACIÓN ---

# Frecuencias de letras estándar para el idioma español.
# Fuente: Análisis de textos extensos en español.
FRECUENCIAS_ESPANOL = {
    'A': 12.53, 'B': 1.42, 'C': 4.68, 'D': 5.86, 'E': 13.68,
    'F': 0.69, 'G': 1.01, 'H': 0.70, 'I': 6.25, 'J': 0.44,
    'K': 0.01, 'L': 4.97, 'M': 3.15, 'N': 6.71, 'Ñ': 0.31,
    'O': 8.68, 'P': 2.51, 'Q': 0.88, 'R': 6.87, 'S': 7.98,
    'T': 4.63, 'U': 3.93, 'V': 0.90, 'W': 0.02, 'X': 0.22,
    'Y': 0.90, 'Z': 0.52
}

# Alfabeto español utilizado para el análisis y cifrado.
ALFABETO_ES = string.ascii_uppercase + 'Ñ'

# --- FUNCIONES DE PROCESAMIENTO DE TEXTO Y CRIPTOGRAFÍA ---

def limpiar_texto(texto: str) -> str:
    """
    Normaliza el texto a mayúsculas y filtra solo los caracteres del alfabeto español.

    Args:
        texto: El texto de entrada.

    Returns:
        El texto limpio y en mayúsculas.
    """
    return "".join([char for char in texto.upper() if char in ALFABETO_ES])

def aplicar_cifrado_cesar(texto: str, desplazamiento: int) -> str:
    """
    Aplica el cifrado César a un texto dado.

    Args:
        texto: El texto a cifrar (puede contener cualquier caracter).
        desplazamiento: El número de posiciones a desplazar.

    Returns:
        El texto cifrado.
    """
    resultado = ""
    texto_mayus = texto.upper()
    for char in texto_mayus:
        if char in ALFABETO_ES:
            idx = ALFABETO_ES.index(char)
            nuevo_idx = (idx + desplazamiento) % len(ALFABETO_ES)
            resultado += ALFABETO_ES[nuevo_idx]
        else:
            resultado += char  # Mantiene caracteres no alfabéticos
    return resultado

def obtener_n_gramas(texto_limpio: str) -> tuple:
    """
    Calcula los 10 bigramas y trigramas más comunes en un texto.

    Args:
        texto_limpio: El texto ya procesado (solo letras).

    Returns:
        Una tupla con las listas de los 10 bigramas y trigramas más comunes.
    """
    bigramas = Counter(texto_limpio[i:i+2] for i in range(len(texto_limpio) - 1))
    trigramas = Counter(texto_limpio[i:i+3] for i in range(len(texto_limpio) - 2))
    return bigramas.most_common(10), trigramas.most_common(10)

# --- FUNCIONES DE CRIPTOANÁLISIS ---

def calcular_ic(texto_limpio: str) -> float:
    """
    Calcula el Índice de Coincidencia (IC) de un texto.

    Args:
        texto_limpio: El texto ya procesado (solo letras).

    Returns:
        El valor del Índice de Coincidencia.
    """
    n = len(texto_limpio)
    if n < 2:
        return 0.0
    
    conteo = Counter(texto_limpio)
    numerador = sum(f * (f - 1) for f in conteo.values())
    denominador = n * (n - 1)
    
    return numerador / denominador if denominador > 0 else 0.0

def calcular_chi_cuadrado(texto_limpio: str) -> float:
    """
    Calcula la puntuación de Chi-cuadrado del texto comparando sus frecuencias
    con las del español estándar. Un valor más bajo indica una mayor similitud.

    Args:
        texto_limpio: El texto ya procesado (solo letras).

    Returns:
        La puntuación de Chi-cuadrado.
    """
    n = len(texto_limpio)
    if n == 0:
        return float('inf')

    conteo_observado = Counter(texto_limpio)
    chi_cuadrado = 0.0

    for letra in ALFABETO_ES:
        frecuencia_esperada = FRECUENCIAS_ESPANOL.get(letra, 0) / 100.0
        conteo_esperado = frecuencia_esperada * n
        conteo_observado_letra = conteo_observado.get(letra, 0)
        
        diferencia = conteo_observado_letra - conteo_esperado
        chi_cuadrado += (diferencia ** 2) / (conteo_esperado if conteo_esperado > 0 else 1)

    return chi_cuadrado

def crack_cesar(texto_cifrado: str) -> dict:
    """
    Intenta descifrar un texto con cifrado César probando todos los desplazamientos
    y eligiendo el que produce el menor valor de Chi-cuadrado.

    Args:
        texto_cifrado: El texto cifrado a analizar.

    Returns:
        Un diccionario con el mejor desplazamiento, el texto descifrado y la puntuación.
    """
    texto_limpio_cifrado = limpiar_texto(texto_cifrado)
    mejores_resultados = {
        'desplazamiento': -1,
        'puntuacion': float('inf'),
        'texto': ''
    }

    for d in range(len(ALFABETO_ES)):
        texto_prueba = aplicar_cifrado_cesar(texto_limpio_cifrado, -d) # Descifrar es desplazar en negativo
        puntuacion = calcular_chi_cuadrado(texto_prueba)
        
        if puntuacion < mejores_resultados['puntuacion']:
            mejores_resultados['puntuacion'] = puntuacion
            mejores_resultados['desplazamiento'] = d
            mejores_resultados['texto'] = aplicar_cifrado_cesar(texto_cifrado, -d)
            
    return mejores_resultados

# --- INTERFAZ DE USUARIO (STREAMLIT) ---

def main():
    """
    Función principal que construye y ejecuta la aplicación Streamlit.
    """
    st.set_page_config(page_title="Criptoanálisis Pro", layout="wide")
    st.title("🕵️‍♂️ Herramienta de Criptoanálisis Profesional")
    st.markdown("Analiza textos, aplica cifrados y utiliza herramientas estadísticas para descifrarlos.")

    # --- BARRA LATERAL PARA ENTRADAS ---
    with st.sidebar:
        st.header("📥 Ingreso del Texto")
        texto_entrada = st.text_area("Ingresá texto manualmente:", height=150, key="texto_manual")
        
        archivo_subido = st.file_uploader("O cargar archivo de texto (.txt):", type="txt")
        if archivo_subido:
            try:
                texto_entrada = StringIO(archivo_subido.getvalue().decode("utf-8")).read()
                st.success("Archivo cargado correctamente.")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
                return

        st.header("🔐 Opciones de Cifrado")
        aplicar_cesar = st.checkbox("Aplicar cifrado César", value=False)
        desplazamiento = 0
        if aplicar_cesar:
            desplazamiento = st.slider("Desplazamiento César", 1, len(ALFABETO_ES) - 1, 3)

    if not texto_entrada:
        st.info("Por favor, ingresa un texto o sube un archivo para comenzar el análisis.")
        return

    # --- PROCESAMIENTO PRINCIPAL ---
    texto_original = texto_entrada
    texto_procesado = aplicar_cifrado_cesar(texto_original, desplazamiento) if aplicar_cesar else texto_original
    texto_limpio = limpiar_texto(texto_procesado)

    if not texto_limpio:
        st.warning("El texto no contiene letras del alfabeto español para analizar.")
        return

    # --- PESTAÑAS PARA MOSTRAR RESULTADOS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis de Frecuencias", "🔎 N-gramas", "🛠️ Herramientas de Criptoanálisis", "📄 Texto Procesado"])

    with tab4:
        st.subheader("Texto Original")
        st.text_area("", value=texto_original, height=150, key="original")
        if aplicar_cesar:
            st.subheader("Texto Cifrado con César")
            st.text_area("", value=texto_procesado, height=150, key="cifrado")

    with tab1:
        st.subheader("Frecuencia de Letras")
        conteo = Counter(texto_limpio)
        total_letras = len(texto_limpio)
        frecuencia_calculada = {letra: (conteo.get(letra, 0) / total_letras) * 100 for letra in ALFABETO_ES}

        df = pd.DataFrame({
            'Letra': list(FRECUENCIAS_ESPANOL.keys()),
            'Esperada Español (%)': list(FRECUENCIAS_ESPANOL.values()),
            'Observada en Texto (%)': [frecuencia_calculada.get(l, 0) for l in FRECUENCIAS_ESPANOL.keys()]
        })

        # Gráfico de Frecuencias
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(x='Letra', y='Observada en Texto (%)', kind='bar', ax=ax, label='Observada', alpha=0.7)
        df.plot(x='Letra', y='Esperada Español (%)', kind='line', ax=ax, color='red', marker='o', linestyle='--', label='Español Estándar')
        ax.set_title("Comparación de Frecuencias de Letras", fontsize=16)
        ax.set_xlabel("Letra")
        ax.set_ylabel("Frecuencia (%)")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig)

        # Tabla de Frecuencias
        st.dataframe(df.style.format({'Esperada Español (%)': '{:.2f}', 'Observada en Texto (%)': '{:.2f}'}))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar CSV", data=csv, file_name="frecuencias.csv", mime="text/csv")

    with tab2:
        st.subheader("Análisis de N-gramas")
        bigramas, trigramas = obtener_n_gramas(texto_limpio)
        col1, col2 = st.columns(2)
        with col1:
            st.write("🔸 **Top 10 Bigramas**")
            df_bi = pd.DataFrame(bigramas, columns=['Bigrama', 'Frecuencia'])
            st.dataframe(df_bi)
        with col2:
            st.write("🔹 **Top 10 Trigramas**")
            df_tri = pd.DataFrame(trigramas, columns=['Trigrama', 'Frecuencia'])
            st.dataframe(df_tri)

    with tab3:
        st.subheader("Índice de Coincidencia (IC)")
        ic_calculado = calcular_ic(texto_limpio)
        st.metric("IC del Texto", f"{ic_calculado:.4f}")
        st.info("""
        **Interpretación del IC:**
        - **Español estándar:** ~0.0778
        - **Texto aleatorio (o cifrado robusto):** ~0.0385 (1/27)
        Un IC cercano a 0.0778 sugiere que el texto está en español o cifrado con un cifrado de sustitución simple (como César). Un IC bajo sugiere un cifrado más complejo (polialfabético) o texto aleatorio.
        """)

        st.subheader("Descifrador Automático de César")
        if st.button("🔍 Intentar Descifrar César"):
            with st.spinner("Analizando todos los desplazamientos..."):
                resultado_crack = crack_cesar(texto_procesado)
                st.success(f"¡Análisis completo! El desplazamiento más probable es **{resultado_crack['desplazamiento']}**.")
                st.write(f"Puntuación Chi-cuadrado (menor es mejor): `{resultado_crack['puntuacion']:.2f}`")
                st.text_area("Texto Probablemente Descifrado:", value=resultado_crack['texto'], height=200)

if __name__ == "__main__":
    main()
