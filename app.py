import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

from sympy.physics.units import years

# --- Configuración de la página ---
st.set_page_config(page_title="Predicción de Acciones", layout="centered")

# --- Estilo visual ---
st.markdown("""
    <style>
    .stApp {
        background-color: white;
        color: black;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #00b4d8;
    }
    .stButton>button {
        background-color: #00b4d8;
        color: black;
        border-radius: 10px;
        height: 40px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0090b0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Título ---
st.title("Consultar precio estimado de una acción")
st.write("Selecciona una acción y una fecha para conocer el precio estimado según el modelo de regresión lineal.")

# --- Inputs principales ---
opcion = st.selectbox(
    "Selecciona una acción:",
    ("AAPL", "MSFT", "TSLA", "GOOG", "AMZN")
)
fecha_input = st.date_input("Selecciona una fecha futura:", min_value=datetime.now() + timedelta(days=1))

buscar = st.button(" Consultar precio")

# --- Solo ejecutar cuando el usuario haga clic ---
if buscar:
    # --- Descargar datos ---
    st.subheader(f" Datos históricos de {opcion}")
    data = yf.download(opcion, start=datetime.now() + timedelta(days=-365*4), end=datetime.now() + timedelta(days=-1))

    if data.empty:
        st.error("No se pudieron obtener datos para esta acción. Intenta con otra.")
        st.stop()

    st.dataframe(data.tail())

    # --- Gráfico del precio histórico ---
    st.line_chart(data['Close'], use_container_width=True)

    # --- Preparar datos ---
    data['Dias'] = np.arange(len(data))
    X = data[['Dias']]
    y = data['Close']
    y_cota_inferior = data['Low']
    y_cota_superior = data['High']

    #
    test_size = 0.15

    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train_cota_inferior, X_test_cota_inferior, y_train_cota_inferior, y_test_cota_inferior = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train_cota_superior, X_test_cota_superior, y_train_cota_superior, y_test_cota_superior = train_test_split(X, y, test_size=test_size, shuffle=False)

    # --- Entrenar modelo ---
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Cota
    modelo_cota_inferior = LinearRegression()
    modelo_cota_inferior.fit(X_train_cota_inferior, y_train_cota_inferior)
    y_pred_cota_inferior = modelo_cota_inferior.predict(X_test_cota_inferior)
    modelo_cota_superior = LinearRegression()
    modelo_cota_superior.fit(X_train_cota_superior, y_train_cota_superior)
    y_pred_cota_superior = modelo_cota_superior.predict(X_test_cota_superior)

    # --- Gráfico de predicciones ---
    dias_futuros = np.arange(len(data), len(data) + 30).reshape(-1, 1)
    predicciones = modelo.predict(dias_futuros)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['Dias'], data['Close'], label='Histórico', color='#00b4d8')
    ax.plot(X_test, y_pred, label='Predicción actual', color='#ff4b4b')
    ax.plot(X_test_cota_inferior, y_pred_cota_inferior, label='Límite estimado inferior', color='#2768F5')
    ax.plot(X_test_cota_superior, y_pred_cota_superior, label='Límite estimado superior', color='#6827F5')
    ax.plot(dias_futuros, predicciones, label='Futuro estimado (30 días)', color='#06d6a0')
    ax.set_title(f'Predicción del precio de {opcion}', color='black', fontsize=14)
    ax.set_xlabel('Días', color='black')
    ax.set_ylabel('Precio (USD)', color='black')
    ax.legend()
    st.pyplot(fig)

    # --- Predicción personalizada ---
    ultimo_dia = data.index[-1].date()
    dias_diferencia = (fecha_input - ultimo_dia).days

    if dias_diferencia < 0:
        st.warning("Esa fecha ya pasó. Solo se pueden predecir fechas futuras.")
    else:
        dia_futuro = np.array([[len(data) + dias_diferencia]])
        precio_estimado = float(modelo.predict(dia_futuro)[0])
        precio_actual = float(data['Close'].iloc[-1])

        # --- Calcular variación porcentual ---
        cambio = ((precio_estimado - precio_actual) / precio_actual) * 100

        if cambio > 0:
            tendencia = f"Se espera un <strong>aumento de {cambio:.2f}%</strong> respecto al precio actual."
            color = "green"
        elif cambio < 0:
            tendencia = f"Se espera una <strong>disminución de {abs(cambio):.2f}%</strong> respecto al precio actual."
            color = "red"
        else:
            tendencia = "🔹 Se espera que el precio se mantenga igual."
            color = "gray"

        r2_formatted = f'{r2_score(y_test, y_pred):.2f}'
        precio_actual_formatted = f'{precio_actual:.2f}'
        precio_estimado_formatted = f'{precio_estimado:.2f}'

        # --- Mostrar resultado ---
        st.markdown(f"""
        <h2>Resultado de la predicción</h2>
        <p>
        <strong>Presición:</strong> {r2_formatted}<br>
        <strong>Acción:</strong> {opcion}<br>
        <strong>Fecha seleccionada:</strong> {fecha_input.strftime('%Y-%m-%d')}<br>
        <strong>Precio actual:</strong> $""" + precio_actual_formatted + """ USD<br>
        <strong>Precio estimado para esa fecha:</strong> $""" + precio_estimado_formatted + f""" USD<br>
        </p>
        <br>
        <p style="color: {color}">{tendencia}</p>
        """, unsafe_allow_html=True)


