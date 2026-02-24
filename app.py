# -----------------------------
# Import Required Libraries
# -----------------------------
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import uuid
import pytz

# -----------------------------
# MongoDB Connection
# -----------------------------
uri = "mongodb+srv://diabetes_user:Diabetes%40123@diabetescluster.oxegep6.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))
db = client["diabetes_app"]

users_collection = db["registered_users"]
predictions_collection = db["predictions"]

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------
if "registered" not in st.session_state:
    st.session_state.registered = False

if "patient_info" not in st.session_state:
    st.session_state.patient_info = {}

if "show_success" not in st.session_state:
    st.session_state.show_success = False


# =====================================================
# REGISTRATION PAGE
# =====================================================
def registration_page():

    st.title("📝 Patient Registration")

    with st.form("registration_form"):

        name = st.text_input("Full Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email Address")
        address = st.text_area("Address")

        submit = st.form_submit_button("Register")

        if submit:
            if all([name, phone, email, address]):

                ist = pytz.timezone("Asia/Kolkata")
                current_time = datetime.now(ist)

                patient_id = "PAT" + str(uuid.uuid4().int)[:6]

                user_data = {
                    "patient_id": patient_id,
                    "name": name,
                    "phone": phone,
                    "email": email,
                    "address": address,
                    "gender": "Not Selected",
                    "created_at": current_time.strftime("%d-%m-%Y %I:%M:%S %p")
                }

                users_collection.insert_one(user_data)

                st.session_state.patient_info = user_data
                st.session_state.registered = True
                st.session_state.show_success = True
                st.rerun()

            else:
                st.error("❌ Please fill all fields")


# =====================================================
# MODEL LOADING
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler_svm.pkl")
    return model, scaler


# =====================================================
# MAIN PREDICTION PAGE
# =====================================================
def prediction_page():

    model, scaler = load_model()
    info = st.session_state.patient_info

    # ---------------- Sidebar ----------------
    st.sidebar.markdown("## Patient Profile")
    st.sidebar.markdown(f"**Name:** {info['name']}")
    st.sidebar.markdown(f"**Phone:** {info['phone']}")
    st.sidebar.markdown(f"**Email:** {info['email']}")

    st.sidebar.markdown("### Medical Inputs")

    age = st.sidebar.number_input("Age", 21, 100, 30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 0) if gender == "Female" else 0
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    bp = st.sidebar.slider("Blood Pressure", 0, 130, 70)
    skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 80)
    bmi = st.sidebar.number_input("BMI", 10.0, 70.0, 25.0)
    dpf = st.sidebar.slider("DPF", 0.0, 2.5, 0.5)

    predict_btn = st.sidebar.button("Predict", use_container_width=True)

    st.title("🩺 Diabetes Prediction System")

    if predict_btn:

        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_std = scaler.transform(input_data)

        prob = model.predict_proba(input_std)[0]
        prob_positive = prob[1] * 100

        if prob_positive < 30:
            risk_label = "Low Risk"
        elif prob_positive < 70:
            risk_label = "Moderate Risk"
        else:
            risk_label = "High Risk"

        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)

        # ---------------- SAVE TO MONGODB ----------------
        prediction_doc = {
            "patient_id": info["patient_id"],
            "name": info["name"],
            "phone": info["phone"],
            "email": info["email"],
            "gender": gender,
            "age": age,
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": bp,
            "skin_thickness": skin,
            "insulin": insulin,
            "bmi": bmi,
            "dpf": dpf,
            "risk_label": risk_label,
            "risk_percentage": round(prob_positive, 2),
            "created_at": now.strftime("%d-%m-%Y %H:%M:%S")
        }

        predictions_collection.insert_one(prediction_doc)

        # ---------------- PDF GENERATION ----------------
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Diabetes Prediction Report", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph(f"<b>Patient ID:</b> {info['patient_id']}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Name:</b> {info['name']}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Risk Level:</b> {risk_label}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Risk %:</b> {prob_positive:.2f}%", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        medical_table = [
            ["Age", age],
            ["Gender", gender],
            ["Pregnancies", pregnancies],
            ["Glucose", glucose],
            ["Blood Pressure", bp],
            ["Skin Thickness", skin],
            ["Insulin", insulin],
            ["BMI", bmi],
            ["DPF", dpf]
        ]

        table = Table(medical_table, colWidths=[3 * inch, 3 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 1, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)
        ]))

        elements.append(Paragraph("Medical Inputs", styles["Heading2"]))
        elements.append(table)

        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()

        st.download_button(
            "📄 Download Full Medical Report",
            pdf,
            "diabetes_report.pdf",
            "application/pdf"
        )


# =====================================================
# Navigation
# =====================================================
if not st.session_state.registered:
    registration_page()
else:
    prediction_page()
