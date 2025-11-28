# app.py (versión revisada y robusta)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
from io import BytesIO

# ----------------- Configuración de la página -----------------
st.set_page_config(page_title="Predicción Futurista de Acciones", layout="wide")

# ----------------- Estilos (tema claro, glass) -----------------
st.markdown(
    """
    <style>
    body {
        background: #f2f5f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background: #f2f5f9;
        color: #0a0a0a;
    }
    h1, h2, h3, h4 {
        color: #0a2540;
        font-weight: 700;
        letter-spacing: 0.6px;
    }
    .glass {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(9px);
        -webkit-backdrop-filter: blur(9px);
        border-radius: 16px;
        border: 1px solid rgba(200,200,200,0.4);
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00a8e8, #0077b6);
        color: white;
        padding: 0.6rem 1.0rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.03);
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Header con canvas (usando components.html) -----------------
header_html = """
<div style="text-align:center; margin-bottom:18px;">
  <h1>Predicción Futurista de Precios de Acciones</h1>
  <p style="font-size:15px; color:#345;">Interfaz limpia, con visual y opciones de predicción configurables.</p>
  <canvas id="orbCanvas" width="220" height="220" style="border-radius:50%; display:block; margin:12px auto; background:transparent;"></canvas>
</div>

<script>
(function(){
  const canvas = document.getElementById('orbCanvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  let angle = 0;

  function render() {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);

    // background radial glow
    const gradBg = ctx.createRadialGradient(w/2, h/2, 10, w/2, h/2, 120);
    gradBg.addColorStop(0, 'rgba(0,168,232,0.14)');
    gradBg.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = gradBg;
    ctx.fillRect(0,0,w,h);

    // main orbit
    const r = 70;
    const cx = w/2 + Math.cos(angle*0.6) * 6;
    const cy = h/2 + Math.sin(angle*0.8) * 6;

    const grad = ctx.createLinearGradient(0,0,w,h);
    grad.addColorStop(0, '#00a8e8');
    grad.addColorStop(1, '#0077b6');

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 3;
    ctx.stroke();

    // inner rotating spokes
    for(let i=0;i<7;i++){
      const a = angle + i*(Math.PI*2/7);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(a)*(r-12), cy + Math.sin(a)*(r-12));
      ctx.strokeStyle = 'rgba(0,120,180,0.16)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // small rotating dot
    ctx.beginPath();
    ctx.arc(cx + Math.cos(angle)*r, cy + Math.sin(angle)*r, 5, 0, Math.PI*2);
    ctx.fillStyle = '#00a8e8';
    ctx.fill();

    angle += 0.02;
    requestAnimationFrame(render);
  }

  render();
})();
</script>
"""

components.html(header_html, height=270)

# ----------------- Sidebar: configuración -----------------
st.sidebar.title("Configuración de la Predicción")

accion = st.sidebar.selectbox("Selecciona la acción", ("AAPL", "MSFT", "TSLA", "GOOG", "AMZN"))

# asegurar min_value es date (no datetime)
min_fecha = (datetime.now() + timedelta(days=1)).date()
fecha_obj = st.sidebar.date_input("Fecha objetivo", min_value=min_fecha)

# ventana de predicción configurable para el gráfico
pred_window = st.sidebar.slider("Días a predecir en el gráfico", 7, 180, 30, step=1)

# años de historial
lookback_years = st.sidebar.slider("Años de historial", 1, 8, 4, step=1)

btn_run = st.sidebar.button("Consultar y predecir")

# ----------------- Sección de perfil (opcional) -----------------
with st.sidebar:
    st.markdown("---")
    st.markdown("**Proyecto:** Predicción de precios de acciones")
    st.markdown("**Autor:** Santiago Roa diego alejandro orjuela ")
    st.markdown("**Modelo base:** Regresión Lineal")
    st.markdown("---")

# ----------------- Main layout -----------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Resumen")
    st.markdown(f"- Acción seleccionada: **{accion}**")
    st.markdown(f"- Fecha objetivo: **{fecha_obj}**")
    st.markdown(f"- Ventana gráfica (días): **{pred_window}**")
    st.markdown(f"- Historial (años): **{lookback_years}**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Métricas (al ejecutar)")
    st.markdown("- Precisión (R²): —")
    st.markdown("- Precio actual: —")
    st.markdown("- Precio estimado: —")
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Consulta y Resultados")
    st.markdown("Configura la predicción en la barra lateral y presiona **Consultar y predecir**.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Lógica principal -----------------
if btn_run:
    # descargar datos con try/except
    start_date = datetime.now() - timedelta(days=lookback_years * 365)
    try:
        data = yf.download(accion, start=start_date, end=datetime.now())
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        data = pd.DataFrame()

    if data.empty:
        st.error("No se encontraron datos para la acción solicitada.")
    else:
        # limpieza básica
        data = data.dropna().copy()
        data.reset_index(inplace=True)  # facilita manipulación con fechas
        data['Dias'] = np.arange(len(data))
        X = data[['Dias']]
        y = data['Close']

        # partición y entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
        modelo = LinearRegression().fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # métricas
        # r2_score espera vectores; aseguramos que sean 1D
        try:
            r2 = float(r2_score(y_test.values, np.asarray(y_pred).reshape(-1)))
        except Exception:
            r2 = None

        precio_actual = float(data['Close'].iloc[-1])

        # preparar predicciones para gráfico según ventana pred_window
        dias_futuros = np.arange(len(data), len(data) + pred_window).reshape(-1, 1)
        pred_fut = modelo.predict(dias_futuros)
        # for plotting, crear índice de fechas futuras
        last_date = data['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(pred_window)]

        # gráfico principal
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader(f"Histórico y predicción — {accion}")

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(data['Date'], data['Close'], label='Histórico', linewidth=1.6)
        ax.plot(data['Date'].iloc[-len(X_test):], y_pred, linestyle='--', label='Predicción (test)', linewidth=1.4)
        ax.plot(future_dates, pred_fut, label=f'Estimación {pred_window} días', linewidth=1.6)
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio (USD)')
        ax.legend()
        ax.grid(alpha=0.08)
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # PREDICCIÓN PARA LA FECHA SELECCIONADA
        ultimo_dia = data['Date'].iloc[-1].date()
        dias_delta = (fecha_obj - ultimo_dia).days

        if dias_delta < 0:
            st.warning("La fecha seleccionada ya pasó. Elige una fecha futura.")
        else:
            # construir input para el modelo y asegurarnos de sacar un float
            dia_futuro = np.array([[len(data) + dias_delta]])
            precio_estimado_raw = modelo.predict(dia_futuro.reshape(-1, 1))
            # precio_estimado_raw puede ser array -> extraer float
            try:
                precio_estimado = float(np.squeeze(precio_estimado_raw))
            except Exception:
                # fallback por si algo raro sucede
                precio_estimado = float(precio_estimado_raw[0]) if hasattr(precio_estimado_raw, '__iter__') else float(precio_estimado_raw)

            cambio = float(((precio_estimado - precio_actual) / precio_actual) * 100.0)

            color = "#008c3b" if cambio > 0 else "#b30000" if cambio < 0 else "gray"
            tendencia = (
                f"Aumento de {cambio:.2f}%" if cambio > 0 else
                f"Disminución de {abs(cambio):.2f}%" if cambio < 0 else
                "Se mantiene estable"
            )

            # Mostrar tarjetas de resultado
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.subheader("Resultado personalizado")
            st.markdown(f"- Acción: **{accion}**")
            st.markdown(f"- Fecha seleccionada: **{fecha_obj.strftime('%Y-%m-%d')}**")
            st.markdown(f"- Precio actual (último cierre): **${precio_actual:.2f}**")
            st.markdown(f"- Precio estimado: **${precio_estimado:.2f}**")
            st.markdown(f"- Cambio estimado: **{cambio:.2f}%** ({tendencia})")
            if r2 is not None:
                st.markdown(f"- Precisión estadística (R²): **{r2:.4f}**")
            else:
                st.markdown(f"- Precisión estadística (R²): —")
            st.markdown("</div>", unsafe_allow_html=True)

        # Mostrar tabla con datos y permitir descarga CSV
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Datos históricos (descarga)")
        st.dataframe(data.set_index('Date').tail(200))

        # preparar CSV para descarga
        csv_bytes = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar histórico (CSV)",
            data=csv_bytes,
            file_name=f"{accion}_historico.csv",
            mime="text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)
