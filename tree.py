import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="Smart ML Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
    color: white;
}

#MainMenu {
    visibility: hidden;
}

footer, header {
    visibility: hidden;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(
        135deg,
        rgba(59,130,246,0.25),
        rgba(139,92,246,0.20)
    );
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 45px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    color: #cbd5e1;
}

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.18);
    backdrop-filter: blur(10px);
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
}

label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

.stNumberInput input {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;
    font-size: 18px;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    box-shadow: 0 8px 25px rgba(59,130,246,0.35);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(139,92,246,0.45);
}

.prediction {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    background: linear-gradient(
        135deg,
        rgba(34,197,94,0.18),
        rgba(16,185,129,0.12)
    );
    border: 1px solid rgba(34,197,94,0.30);
    margin-top: 25px;
}

.prediction-icon {
    font-size: 50px;
}

.prediction-title {
    font-size: 28px;
    font-weight: 800;
    margin: 10px 0;
}

.prediction-text {
    font-size: 17px;
    color: #cbd5e1;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #111827);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.info-box {
    padding: 18px;
    border-radius: 15px;
    background: rgba(59,130,246,0.10);
    border: 1px solid rgba(59,130,246,0.20);
    color: #dbeafe;
}

.custom-footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open("decision_tree_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception:
    st.error("❌ Model file could not be loaded.")
    st.info("Make sure decision_tree_model.pkl is in the same folder as app.py")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 🤖 ML Dashboard")
    st.markdown("---")

    st.markdown("""
### About

This application uses a **Machine Learning Classification Model**
to predict the class based on user-provided feature values.
""")

    st.markdown("---")
    st.markdown("### ⚙️ Model")
    st.success("Model Loaded")

    st.markdown("""
**Algorithm:**  
Decision Tree Classifier

**Mode:**  
Single Prediction
""")

    st.markdown("---")
    st.caption("🚀 Built with Python & Streamlit")

# Hero
st.markdown("""
<div class="hero">
    <h1>🤖 Smart ML Classifier</h1>
    <p>Intelligent classification powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Model status
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="🧠 Model", value="Decision Tree")

with col2:
    st.metric(label="⚡ Prediction", value="Real-Time")

with col3:
    st.metric(label="🔐 Status", value="Active")

st.markdown("<br>", unsafe_allow_html=True)

# Input section
st.markdown("""
<div class="card">
    <div class="card-title">📊 Enter Feature Values</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌿 Sepal Features")

    sepal_length = st.number_input(
        "Sepal Length",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

with col2:
    st.markdown("### 🌸 Petal Features")

    petal_length = st.number_input(
        "Petal Length",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

# Predict button
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Class"):
    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    try:
        prediction = model.predict(input_data)
        predicted_class = prediction[0]

        if hasattr(model, "classes_"):
            predicted_label = predicted_class
        else:
            predicted_label = predicted_class

        st.markdown(f"""
<div class="prediction">
    <div class="prediction-icon">🎯</div>
    <div class="prediction-title">Prediction Result</div>
    <div class="prediction-text">The predicted class is</div>
    <h2>{predicted_label}</h2>
</div>
""", unsafe_allow_html=True)

        st.balloons()

    except Exception as e:
        st.error(f"Prediction error: {e}")

# Information
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>💡 How it works</strong><br><br>
Enter the four feature values above and click
<strong>Predict Class</strong>. The trained machine-learning
model will process the input and return the predicted class.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="custom-footer">
    Made with ❤️ using Python, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)