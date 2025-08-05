import streamlit as st
import string
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from io import StringIO

# Frecuencias típicas del español
frecuencias_espanol = {
    'A': 12.53, 'B': 1.42, 'C': 4.68, 'D': 5.86, 'E': 13.68,
    'F': 0.69, 'G': 1.01, 'H': 0.70, 'I': 6.25, 'J': 0.44,
    'K': 0.00, 'L': 4.97, 'M': 3.15, 'N': 6.71, 'Ñ': 0.31,
    'O': 8.68, 'P': 2.51, 'Q': 0.88, 'R': 6.87, 'S': 7.98,
    'T': 4.63, 'U': 3.93, 'V': 0.90, 'W': 0.02, 'X': 0.22,
    'Y': 0.90, 'Z': 0.52
}

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

def obtener_bigrama_trigrama(texto_limpio):
    bigramas = Counter()
    trigramas = Counter()
    for i in range(len(texto_limpio)-1):
        bigramas[texto_limpio[i:i+2]] += 1
    for i in range(len(texto_limpio)-2):
        trigramas[texto_limpio[i:i+3]] += 1
    return bigramas.most_common(10), trigramas.most_common(10)

# Título de la App
st.title("🕵️‍♂️ Estadísticas del Lenguaje - Criptoanálisis interactivo")

st.markdown("""
Analiza la frecuencia de letras, bigramas y trigramas en un texto. Aplicá opcionalmente un **cifrado César** para observar el impacto en el análisis.
""")

# Ingreso de texto
st.subheader("📥 Ingreso del Texto")
texto_entrada = st.text_area("Ingresá manualmente el texto o usá la opción de archivo más abajo", height=150)

archivo_subido = st.file_uploader("O cargar archivo de texto (.txt)", type="txt")
if archivo_subido:
    texto_entrada = StringIO(archivo_subido.getvalue().decode("utf-8")).read()

if texto_entrada:
    texto_mayus = texto_entrada.upper()

    # Cifrado César (opcional)
    st.subheader("🔐 Cifrado César (Opcional)")
    aplicar_cesar = st.checkbox("Aplicar cifrado César al texto", value=False)
    desplazamiento = st.slider("Desplazamiento César", 1, 26, 3) if aplicar_cesar else 0
    if aplicar_cesar:
        texto_mayus = aplicar_cifrado_cesar(texto_mayus, desplazamiento)
        st.text_area("🔄 Texto cifrado con César:", value=texto_mayus, height=150)

    # Procesar letras
    alfabeto = string.ascii_uppercase + 'Ñ'
    solo_letras = [c for c in texto_mayus if c in alfabeto]
    conteo = Counter(solo_letras)
    total = sum(conteo.values())

    frecuencia_calculada = {
        letra: (conteo.get(letra, 0) / total) * 100 for letra in frecuencias_espanol.keys()
    }

    df = pd.DataFrame({
        'Letra': list(frecuencias_espanol.keys()),
        'Esperada (%)': list(frecuencias_espanol.values()),
        'Observada (%)': [frecuencia_calculada.get(l, 0) for l in frecuencias_espanol.keys()]
    })

    # Tabla
    st.subheader("📊 Frecuencias de Letras")
    st.dataframe(df.style.format({'Esperada (%)': '{:.2f}', 'Observada (%)': '{:.2f}'}))

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", data=csv, file_name="frecuencias_lenguaje.csv", mime="text/csv")

    # Gráfico
    st.subheader("📈 Gráfico de Frecuencias")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['Letra'], df['Observada (%)'], label='Texto Analizado', alpha=0.7)
    ax.plot(df['Letra'], df['Esperada (%)'], color='red', marker='o', linestyle='--', label='Frecuencia Español')
    ax.set_xlabel("Letra")
    ax.set_ylabel("Frecuencia (%)")
    ax.set_title("Frecuencias Observadas vs Esperadas")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Bigrama y Trigrama
    st.subheader("🔎 Bigrama y Trigrama más comunes")
    texto_limpio = ''.join(solo_letras)
    bigramas, trigramas = obtener_bigrama_trigrama(texto_limpio)

    col1, col2 = st.columns(2)
    with col1:
        st.write("🔸 Top 10 Bigramas")
        for b, f in bigramas:
            st.write(f"{b}: {f} veces")
    with col2:
        st.write("🔹 Top 10 Trigramas")
        for t, f in trigramas:
            st.write(f"{t}: {f} veces")
