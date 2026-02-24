# -----------------------------
# Import Required Libraries
# -----------------------------
#from anyio import current_time
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# ✅ ADDED FOR PDF GENERATION ONLY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO

# -----------------------------
# MongoDB Connection
# -----------------------------

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# -----------------------------
# MongoDB Connection
# -----------------------------
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import uuid
import pytz

#  my string
uri = "mongodb+srv://project00067:Project123@cluster0.vzzvdti.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create MongoDB Client
client = MongoClient(uri, server_api=ServerApi('1'))

# Create Database
db = client["diabetes_app"]

# Create Collection
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

    st.markdown("Please register to access the Diabetes Prediction System")

    with st.form("registration_form"):

        name = st.text_input("Full Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email Address")
        address = st.text_area("Address")

        submit = st.form_submit_button("Register")

        if submit:

            if name and phone and email and address:
                ist = pytz.timezone("Asia/Kolkata")
                
                current_time = datetime.now(ist)

                patient_id = "PAT" + str(uuid.uuid4().int)[:6]

               
                user_data ={
                    "_id": patient_id,
                    "name": name,
                    "phone": phone,
                    "email": email,
                    "address": address,
                    "gender": "Not Selected",
                    "created_at": current_time.strftime("%d-%m-%Y %I:%M:%S %p")

                                           } 
                users_collection.insert_one(user_data)
                st.session_state.patient_info=user_data

                st.session_state.registered = True
                st.session_state.show_success = True

                st.rerun()

            else:
                st.error("❌ Please fill all fields")


# =====================================================
# MAIN PREDICTION PAGE
# =====================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("scaler_svm.pkl")
        return model, scaler
    except:
        st.error("⚠️ Model files not found!")
        st.stop()

def prediction_page():
    model, scaler = load_model()

    # -----------------------------
    # Sidebar Styling (FIXED)
    # -----------------------------
    st.markdown("""
    <style>

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1a3a, #08122b);
        padding: 20px;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: white !important;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #1e2a44 !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: white !important;
    }

    section[data-testid="stSidebar"] input {
        background-color: #1e2a44 !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] button {
        background-color: #3b4b63 !important;
        color: white !important;
        border-radius: 8px !important;
        height: 45px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Load Model
    # -----------------------------
   

    # -----------------------------
    # Sidebar
    # -----------------------------
    st.sidebar.markdown("# Patient Profile")

    info = st.session_state.patient_info

    st.sidebar.markdown(f"**Name:** {info.get('name','')}")
    st.sidebar.markdown(f"**Phone:** {info.get('phone','')}")
    st.sidebar.markdown(f"**Email:** {info.get('email','')}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Medical Inputs")

    age = st.sidebar.slider("Age", 21, 100, 30)

    gender = st.sidebar.selectbox(
        "Gender",
        ["Male", "Female"]
    )
   

# Pregnancy input only for female
    if gender == "Female":
        pregnancies = st.sidebar.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=0
        )
    else:
        pregnancies = 0

    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    bp = st.sidebar.slider("Blood Pressure", 0, 130, 70)
    skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 80)
    bmi = st.sidebar.number_input("BMI", 10.0, 70.0, 25.0)
    dpf = st.sidebar.slider("DPF", 0.0, 2.5, 0.5)

    st.sidebar.markdown("---")

    predict_btn = st.sidebar.button("Predict", use_container_width=True)

    if st.sidebar.button("Logout"):
        st.session_state.registered = False
        st.rerun()

    # -----------------------------
    # Main Title
    # -----------------------------
    st.title("🩺 Diabetes Prediction System")
    st.markdown("AI-Powered Diabetes Risk Assessment Tool")

    if st.session_state.show_success:
        st.success("✅ Registration Successful!")
        st.session_state.show_success = False

    # -----------------------------
    # About System
    # -----------------------------
    st.markdown("""
    ### 📋 About This System

    This Diabetes Prediction System is an AI-powered medical risk assessment tool designed to estimate the likelihood of diabetes based on key health parameters such as glucose level, BMI, blood pressure, age, and family history.

    The system uses a trained Machine Learning model to analyze patterns in medical data and provide an instant risk classification (Low, Moderate, or High). In addition to prediction, it also highlights potential risk factors, positive health indicators, and personalized recommendations to support preventive care.
    ### 🧭 How to Use

    • Enter patient details in sidebar  
    • Click Predict  
    • View results and recommendations  
    """)

    # -----------------------------
    # Prediction
    # -----------------------------
    if predict_btn:
         #update gender in mongodb
         users_collection.update_one(
              {"_id":info["_id"]},
                {"$set":{"gender":gender}}
        ) 
         
         input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
         input_std = scaler.transform(input_data)
         prediction = model.predict(input_std)[0]
         probability = model.predict_proba(input_std)[0]

         prob_negative = probability[0] * 100
         prob_positive = probability[1] * 100

        # ✅ ADD HERE (SAVE TO DATABASE)
        # -----------------------------
        # Determine Risk Label

         if prob_positive<30:
            risk_label="Low Risk"
         elif prob_positive<70:
            risk_label="Moderate Risk"
         else:
            risk_label="High Risk"

        # Indian Time
         ist = pytz.timezone('Asia/Kolkata')
         current_time = datetime.now(ist)    

        # Save prediction to MongoDB
         prediction_data = {
    "patient_id": info["_id"],
    "patient_name": info["name"],
    "age": age,
    "gender": gender,
    "glucose": glucose,
    "blood_pressure": bp,
    "bmi": bmi,
    "prediction": risk_label,
    "probability": round(prob_positive, 2),
    "created_at": current_time.strftime("%d-%m-%Y %H:%M:%S")
}

                 
         predictions_collection.insert_one(prediction_data)

      
         st.markdown("---")
         st.header("Prediction Results")

         col1, col2 = st.columns([2, 1])

         with col1:

            if prob_positive < 30:
                st.success("✅ LOW RISK - Diabetes Unlikely")
            elif prob_positive < 70:
                st.warning("⚠️ MODERATE RISK - Possible Diabetes")
            else:
                st.error("❌ HIGH RISK - Diabetes Likely")

            st.subheader("Probability Breakdown")

            c1, c2 = st.columns(2)
            c1.metric("Non-Diabetic", f"{prob_negative:.1f}%")
            c2.metric("Diabetic", f"{prob_positive:.1f}%")

         with col2:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_positive,
                number={"suffix": "%"},
                title={"text": "Risk Level"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 30], "color": "green"},
                        {"range": [30, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "red"}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

 
        # -----------------------------
        # Risk Factor Analysis
        # -----------------------------
         st.markdown("---")
         st.subheader("Risk Factor Analysis")

         risk_factors = []
         positive_factors = []

         if glucose > 125:
            risk_factors.append("High Glucose Level (>125 mg/dL)")
         elif glucose < 100:
             positive_factors.append("Normal Glucose Level")

         if bmi > 30:
            risk_factors.append("High BMI (Obesity)")
         elif 18.5 <= bmi <= 24.9:
            positive_factors.append("Healthy BMI")

         if age > 45:
            risk_factors.append("Age above 45")

         if bp > 80:
            risk_factors.append("High Blood Pressure")
         elif 60 <= bp <= 80:
            positive_factors.append("Normal Blood Pressure")

         if dpf > 0.5:
            risk_factors.append("Higher Genetic Risk")

         if risk_factors:
            st.warning("Identified Risk Factors:")
            for factor in risk_factors:
                st.markdown(f"- {factor}")

         if positive_factors:
            st.success("Positive Health Indicators:")
            for factor in positive_factors:
                st.markdown(f"- {factor}")

        # -----------------------------
        # Recommendations
        # -----------------------------
         st.markdown("---")
         st.subheader("Recommendations")

         if prob_positive >= 70:
            st.error("""
- Consult a healthcare professional immediately
- Get complete diabetes screening
- Monitor blood sugar regularly
- Improve diet and physical activity
""")
         elif prob_positive >= 30:
            st.warning("""
- Maintain healthy diet
- Increase physical activity
- Monitor glucose periodically
""")
         else:
            st.success("""
- Continue healthy lifestyle
- Exercise regularly
- Routine health check-ups
""")
            
           # -----------------------------
        # COMPLETE PROFESSIONAL PDF REPORT
        # -----------------------------
         buffer = BytesIO()
         doc = SimpleDocTemplate(buffer)
         elements = []
         styles = getSampleStyleSheet()
 
        # Title
         elements.append(Paragraph("Diabetes Prediction Report", styles["Heading1"]))
         elements.append(Spacer(1, 0.3 * inch))

        # Risk Level
         if prob_positive < 30:
            risk_level = "LOW RISK - Diabetes Unlikely"
         elif prob_positive < 70:
            risk_level = "MODERATE RISK - Possible Diabetes"
         else:
            risk_level = "HIGH RISK - Diabetes Likely"

         elements.append(Paragraph(f"<b>Overall Risk Level:</b> {risk_level}", styles["Normal"]))
         elements.append(Paragraph(f"<b>Risk Percentage:</b> {prob_positive:.1f}%", styles["Normal"]))
         elements.append(Spacer(1, 0.3 * inch))

        # Patient Info Table
         patient_table = [
            ["Name", info.get("name","")],
            ["Age", str(age)],
            ["Gender", gender],
            ["Phone", info.get("phone","")],
            ["Email", info.get("email","")],
        ]

         table = Table(patient_table, colWidths=[2.2*inch, 3*inch])
         table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 1, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ]))

         elements.append(Paragraph("Patient Information", styles["Heading2"]))
         elements.append(Spacer(1, 0.2 * inch))
         elements.append(table)
         elements.append(Spacer(1, 0.4 * inch))

        # Risk Factors Section
         elements.append(Paragraph("Risk Factor Analysis", styles["Heading2"]))
         elements.append(Spacer(1, 0.2 * inch))

         if risk_factors:
             elements.append(Paragraph("<b>Identified Risk Factors:</b>", styles["Normal"]))
             elements.append(Spacer(1, 0.1 * inch))
             risk_list = [ListItem(Paragraph(factor, styles["Normal"])) for factor in risk_factors]
             elements.append(ListFlowable(risk_list, bulletType='bullet'))
             elements.append(Spacer(1, 0.3 * inch))

         if positive_factors:
            elements.append(Paragraph("<b>Positive Health Indicators:</b>", styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
            positive_list = [ListItem(Paragraph(factor, styles["Normal"])) for factor in positive_factors]
            elements.append(ListFlowable(positive_list, bulletType='bullet'))
            elements.append(Spacer(1, 0.3 * inch))

        # Recommendations Section
         elements.append(Paragraph("Recommendations", styles["Heading2"]))
         elements.append(Spacer(1, 0.2 * inch))
 
         if prob_positive >= 70:
            recs = [
                "Consult a healthcare professional immediately",
                "Get complete diabetes screening",
                "Monitor blood sugar regularly",
                "Improve diet and physical activity"
            ]
         elif prob_positive >= 30:
            recs = [
                "Maintain healthy diet",
                "Increase physical activity",
                "Monitor glucose periodically"
            ]
         else:
            recs = [
                "Continue healthy lifestyle",
                "Exercise regularly",
                "Routine health check-ups"
            ]

         rec_list = [ListItem(Paragraph(r, styles["Normal"])) for r in recs]
         elements.append(ListFlowable(rec_list, bulletType='bullet'))

         elements.append(Spacer(1, 0.5 * inch))

        # Disclaimer
         elements.append(Paragraph(
            "Medical Disclaimer: This report is AI-generated and does not replace professional medical advice.",
            styles["Normal"]
        ))

         doc.build(elements)
         pdf = buffer.getvalue()
         buffer.close()
 
         st.download_button(
            label="📄 Download Full Medical Report (PDF)",
            data=pdf,
            file_name="diabetes_prediction_report.pdf",
            mime="application/pdf"
        )
         

        # -----------------------------
        # Disclaimer
        # -----------------------------
         st.markdown("---")
         st.warning("""
⚠️ Medical Disclaimer:  
This tool does NOT replace professional medical advice.
""")


# =====================================================
# Navigation
# =====================================================
if not st.session_state.registered:
    registration_page()
else:
    prediction_page()
