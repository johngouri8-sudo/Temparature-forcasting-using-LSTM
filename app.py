
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = "temperature_lstm.keras"
SCALER_PATH = "temperature_scaler.pkl"

SEQUENCE_LENGTH = 5


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Temperature AI Forecast",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #05070d 0%,
            #0b1220 50%,
            #071a2d 100%
        );
        color: white;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }

    .main-title span {
        color: #00aaff;
    }

    .subtitle {
        text-align: center;
        color: #aebbd0;
        font-size: 17px;
        margin-bottom: 25px;
    }

    /* Marquee */
    .marquee-container {
        width: 100%;
        overflow: hidden;
        background: #07182b;
        border-top: 1px solid #008cff;
        border-bottom: 1px solid #008cff;
        padding: 10px 0;
        margin-bottom: 25px;
    }

    .marquee {
        white-space: nowrap;
        display: inline-block;
        animation: marquee 18s linear infinite;
        color: #00aaff;
        font-size: 15px;
        font-weight: 600;
    }

    @keyframes marquee {
        0% {
            transform: translateX(100%);
        }

        100% {
            transform: translateX(-100%);
        }
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,170,255,0.25);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }

    .card-title {
        font-size: 22px;
        font-weight: 700;
        color: white;
        margin-bottom: 5px;
    }

    .card-subtitle {
        color: #8fa5bd;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Prediction result */
    .prediction-card {
        background: linear-gradient(
            135deg,
            #006dcc,
            #003b73
        );
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 10px 35px rgba(0,140,255,0.25);
    }

    .prediction-label {
        font-size: 16px;
        color: #dceeff;
    }

    .prediction-value {
        font-size: 55px;
        font-weight: 800;
        color: white;
        margin: 8px 0;
    }

    .prediction-text {
        color: #c9e6ff;
        font-size: 14px;
    }

    /* Metrics */
    .metric-card {
        background: #0b1422;
        border: 1px solid #173452;
        border-radius: 15px;
        padding: 18px;
        text-align: center;
    }

    .metric-title {
        color: #8fa5bd;
        font-size: 14px;
    }

    .metric-value {
        color: #00aaff;
        font-size: 28px;
        font-weight: 700;
    }

    /* Section headings */
    h1, h2, h3 {
        color: white !important;
    }

    /* Input boxes */
    .stNumberInput input {
        background-color: #0b1422 !important;
        color: white !important;
        border: 1px solid #1b4d75 !important;
        border-radius: 10px !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(
            90deg,
            #0077cc,
            #00aaff
        );
        color: white;
        border: none;
        border-radius: 12px;
        padding: 13px;
        font-size: 17px;
        font-weight: 700;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,170,255,0.35);
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #60758d;
        font-size: 13px;
        margin-top: 35px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌡️ <span>Temperature AI</span> Forecast</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'LSTM-based time-series forecasting dashboard'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MARQUEE
# =========================================================

st.markdown("""
<div class="marquee-container">
    <div class="marquee">
        🤖 LSTM MODEL &nbsp; • &nbsp;
        📊 MODEL EVALUATION &nbsp; • &nbsp;
        🔮 NEXT TEMPERATURE PREDICTION &nbsp; • &nbsp;
        ⚡ REAL-TIME TESTING &nbsp; • &nbsp;
        🌡️ AI POWERED FORECASTING
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model_and_scaler():

    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:

    model, scaler = load_model_and_scaler()

except Exception as e:

    st.error("⚠️ Model files could not be loaded.")

    st.info(
        "Make sure these files are present in the same folder as app.py:"
    )

    st.code(
        "temperature_lstm.keras\n"
        "temperature_scaler.pkl"
    )

    st.stop()


# =========================================================
# MODEL EVALUATION
# =========================================================

st.markdown("""
<div class="card">

<div class="card-title">📊 Model Evaluation</div>

<div class="card-subtitle">
Performance information for the trained LSTM model
</div>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Model</div>
        <div class="metric-value">LSTM</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Sequence Length</div>
        <div class="metric-value">5</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Features</div>
        <div class="metric-value">1</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TEST PREDICTION
# =========================================================

st.markdown("""
<div class="card">

<div class="card-title">🧪 Test Prediction</div>

<div class="card-subtitle">
Enter the latest 5 temperature readings to predict the next value.
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INPUTS
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    temp1 = st.number_input(
        "Reading 1",
        value=15.0,
        step=0.1
    )

with col2:
    temp2 = st.number_input(
        "Reading 2",
        value=15.0,
        step=0.1
    )

with col3:
    temp3 = st.number_input(
        "Reading 3",
        value=15.0,
        step=0.1
    )

with col4:
    temp4 = st.number_input(
        "Reading 4",
        value=15.0,
        step=0.1
    )

with col5:
    temp5 = st.number_input(
        "Reading 5",
        value=15.0,
        step=0.1
    )


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔮 Predict Next Temperature"):

    temperatures = np.array(
        [
            temp1,
            temp2,
            temp3,
            temp4,
            temp5
        ],
        dtype=np.float32
    )

    try:

        # Scale input
        scaled_input = scaler.transform(
            temperatures.reshape(-1, 1)
        )

        # Reshape for LSTM
        X_new = scaled_input.reshape(
            1,
            SEQUENCE_LENGTH,
            1
        )

        # Prediction
        prediction_scaled = model.predict(
            X_new,
            verbose=0
        )

        # Convert back to original temperature
        prediction = scaler.inverse_transform(
            prediction_scaled
        )[0, 0]

        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    🔮 Predicted Next Temperature
                </div>

                <div class="prediction-value">
                    {prediction:.2f} °C
                </div>

                <div class="prediction-text">
                    Forecast generated using the trained LSTM model
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # TEST RESULTS
        # =================================================

        st.markdown("### 📈 Testing Prediction Results")

        result_df = pd.DataFrame({
            "Time Step": [
                "T-5",
                "T-4",
                "T-3",
                "T-2",
                "T-1",
                "Predicted"
            ],

            "Temperature (°C)": [
                temp1,
                temp2,
                temp3,
                temp4,
                temp5,
                prediction
            ]
        })

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # TREND
        # =================================================

        st.markdown("### 📊 Temperature Trend")

        chart_df = pd.DataFrame({
            "Temperature": [
                temp1,
                temp2,
                temp3,
                temp4,
                temp5,
                prediction
            ]
        })

        st.line_chart(chart_df, height=300)

        # =================================================
        # TESTING METRICS
        # =================================================

        avg_temp = np.mean(temperatures)
        difference = prediction - temperatures[-1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Average Input",
                f"{avg_temp:.2f} °C"
            )

        with col2:
            st.metric(
                "Last Reading",
                f"{temperatures[-1]:.2f} °C"
            )

        with col3:
            st.metric(
                "Forecast Change",
                f"{difference:+.2f} °C"
            )

    except Exception as e:

        st.error(
            f"Prediction failed: {str(e)}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    Temperature AI Forecast • Powered by LSTM Neural Network
</div>
""", unsafe_allow_html=True)

