import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Predicción de Ingresos", layout="centered")
st.title("💰 Predicción de Ingresos Estimados")

st.markdown("""
Esta aplicación permite **estimar el ingreso mensual** de un cliente
a partir de sus características.
El objetivo es apoyar a aseguradoras en la **segmentación de clientes**
y en el diseño de **ofertas personalizadas**.
""")

# Cargar modelo
try:
    with open("modelo_ingresos.pkl", "rb") as file:
        modelo = pickle.load(file)
except:
    st.error("❌ No se encontró 'modelo_ingresos.pkl'. Coloque el archivo en la misma carpeta del app.")
    st.stop()

# Panel lateral
st.sidebar.header("📋 Ingrese los datos del cliente")

edad = st.sidebar.number_input("Edad del cliente", min_value=18, max_value=100, value=30)
anios_dir = st.sidebar.number_input("Años viviendo en la dirección", min_value=0, max_value=80, value=5)
gasto_auto = st.sidebar.number_input(
    "Gasto en auto (mensual)",
    min_value=0.0,
    max_value=500.0,
    value=50.0,
    step=0.1
)
anios_empleo = st.sidebar.number_input("Años de empleo", min_value=0, max_value=60, value=3)
anios_residen = st.sidebar.number_input("Años de residencia", min_value=0, max_value=80, value=5)

try:
    dataset = pd.read_csv("dataset_segmentado.csv")
except:
    st.error("❌ No se encontró 'dataset_segmentado.csv'. Coloque este archivo junto al app.")
    st.stop()

# Botón de predicción
if st.button("🔍 Predecir ingreso"):
    entrada = pd.DataFrame({
        "edad": [edad],
        "AniosDireccion": [anios_dir],
        "Gastocoche": [gasto_auto],
        "Aniosempleo": [anios_empleo],
        "Aniosresiden": [anios_residen]
    })

    # 1. Obtenemos la predicción cruda del modelo
    ingreso_pred_raw = modelo.predict(entrada)[0]

    # 2. Hacemos la conversión SOLO para mostrar en pantalla (Visualización)
    # Multiplicamos por 1000 (miles) y dividimos entre 12 (meses)
    ingreso_mensual_real = (ingreso_pred_raw * 1000) / 12

    st.success(f"💰 **Ingreso estimado:** S/ {ingreso_mensual_real:,.2f} / mes")

    # 3. Para el segmento usamos el valor crudo ('ingreso_pred_raw')
    # Esto es vital para que coincida con la escala de tu CSV original
    if ingreso_pred_raw < dataset['ingres_pred'].quantile(0.20):
        segmento = "Muy Bajo"
    elif ingreso_pred_raw < dataset['ingres_pred'].quantile(0.40):
        segmento = "Bajo"
    elif ingreso_pred_raw < dataset['ingres_pred'].quantile(0.60):
        segmento = "Medio"
    elif ingreso_pred_raw < dataset['ingres_pred'].quantile(0.80):
        segmento = "Alto"
    else:
        segmento = "Muy Alto"

    st.info(f"📊 Segmento del cliente: **{segmento}**")
