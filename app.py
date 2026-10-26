import streamlit as st
import pickle
import json
import numpy as np
import random
import pandas as pd
import os
from datetime import datetime

APPOINTMENT_LOG = "appointment_history/appointments_log.csv"

# Session state setup
for key, default in {
    "otp_sent": False,
    "otp_verified": False,
    "user_email": "",
    "generated_otp": "",
    "appointment_done": False,
    "predicted_disease": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Layout & CSS
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #e6f7ff, #ffffff);
        color: #1a1a1a !important;
        font-weight: 500;
    }

    html, body, [class*="css"] {
        color: #1a1a1a !important;
        font-family: 'Segoe UI', sans-serif;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div,
    .stMultiSelect div,
    input, textarea, select {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        font-weight: bold !important;
        border: 1px solid #000000 !important;
        border-radius: 5px !important;
    }

    button {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        font-weight: bold !important;
        border: 1px solid #000000 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem;
        transition: 0.2s ease-in-out;
    }

    button:hover {
        background-color: #f5f5f5 !important;
        border-color: #000000 !important;
    }

    .appointment-card {
        background-color: #ffffff;
        border: 1px solid #cce7f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .appointment-card span {
        font-weight: bold;
    }

    .stTabs [role="tablist"] {
        background-color: #e6f7ff;
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stDataFrameContainer {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        font-size: 16px !important;
    }

    .element-container .stAlert-success,
    .element-container .stAlert-info {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    .stCodeBlock {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 6px;
    }
    /* Style individual tabs */
.stTabs [role="tablist"] {
    justify-content: space-around !important;
    border-bottom: 2px solid #91d5ff;
}

.stTabs [role="tab"] {
    flex: 1 !important;
    font-weight: bold !important;
    padding: 0.75rem 1.5rem !important;
    border: 1px solid #91d5ff !important;
    border-bottom: none !important;
    background-color: #e6f7ff !important;
    color: #1a1a1a !important;
    text-align: center;
    border-radius: 6px 6px 0 0 !important;
    margin: 0 4px;
}

/* Highlight active tab */
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    border-bottom: 2px solid white !important;
}
/* Style for multiselect dropdown box */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 1px solid #000000 !important;
    border-radius: 6px !important;
    padding: 5px !important;
    color: #1a1a1a !important;
    font-weight: bold !important;
}

/* Style for placeholder and selected text */
div[data-baseweb="select"] * {
    color: #1a1a1a !important;
    font-weight: bold !important;
}

/* Ensure dropdown menu (popup) matches style */
div[data-baseweb="popover"] {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    font-weight: bold !important;
    border: 1px solid #000000 !important;
    border-radius: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.image(
    "https://static.vecteezy.com/system/resources/thumbnails/010/314/736/small_2x/doctors-and-nurses-illustration-free-vector.jpg",
    use_container_width=True,
)
st.title("👨‍⚕️ Smart Symptom Checker & Appointment Assistant")


# Load ML model
with open("model/xgboost_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
label_encoder = data["label_encoder"]
mlb = data["mlb"]

# Load doctor mapping
with open("data/disease_doctor_mapping.json", "r") as f:
    doctor_data = json.load(f)

# LOGIN SECTION
if not st.session_state.otp_verified:
    st.subheader("🔐 Login with Email")
    email_input = st.text_input("Enter your email:")
    if st.button("Send OTP"):
        if email_input:
            st.session_state.generated_otp = str(random.randint(100000, 999999))
            st.session_state.user_email = email_input
            st.session_state.otp_sent = True
            st.markdown(
                f"""
    <div style='
        background-color: #e6f7ff;
        color: #1a1a1a;
        border-left: 5px solid #1890ff;
        padding: 1rem;
        border-radius: 5px;
        font-weight: 600;
        margin-top: 1rem;
    '>
        📧 An OTP has been sent to your email: <strong>{email_input}</strong>. Please check your inbox.
    </div>
    """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
<div style='background-color:#ffffff; color:#1a1a1a; font-family:monospace; padding:10px; border:1px solid #ccc; border-radius:6px; font-weight:bold;'>
{st.session_state.generated_otp}
</div>
""",
                unsafe_allow_html=True,
            )

        else:
            st.warning("Please enter a valid email.")

    if st.session_state.otp_sent:
        otp_input = st.text_input("Enter OTP:")
        if st.button("Verify OTP"):
            if otp_input == st.session_state.generated_otp:
                st.session_state.otp_verified = True
                st.markdown(
                    """
<div style='background-color:#e6f7ff; border:1px solid #91d5ff; padding:12px; border-radius:6px; color:#1a1a1a; font-weight:600;'>
✅ Logged in successfully!
</div>
""",
                    unsafe_allow_html=True,
                )

                st.rerun()
            else:
                st.markdown(
                    """
    <div style='
        background-color: #fff1f0;
        color: #a8071a;
        border-left: 5px solid #ff4d4f;
        padding: 1rem;
        border-radius: 5px;
        font-weight: 600;
        margin-top: 1rem;
    '>
        ❌ Invalid OTP. Try again.
    </div>
    """,
                    unsafe_allow_html=True,
                )


# POST-APPOINTMENT
if st.session_state.otp_verified and st.session_state.appointment_done:
    st.markdown(
        """
    <div style='
        background-color: #e6f7ff;
        color: #1a1a1a;
        border-left: 5px solid #1890ff;
        padding: 1rem;
        border-radius: 5px;
        font-weight: 600;
        margin-top: 1rem;
    '>
        🎉 <strong>Appointment confirmed and saved.</strong>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("📋 Your Previous Appointments:")

    if os.path.exists(APPOINTMENT_LOG):
        df_log = pd.read_csv(APPOINTMENT_LOG)
        user_log = df_log[df_log["Email"] == st.session_state.user_email]
        if not user_log.empty:
            st.dataframe(user_log.sort_values("Timestamp", ascending=False))
        else:
            st.info("No past appointments found.")
    else:
        st.info("No appointment log found.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Search Again"):
            st.session_state.appointment_done = False
            st.session_state.predicted_disease = ""
            st.rerun()
    with col2:
        if st.button("🚪 Log Out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# MAIN APP
if st.session_state.otp_verified and not st.session_state.appointment_done:
    tab1, tab2 = st.tabs(["🩺 Diagnosis", "📖 Appointment History"])

    with tab1:
        st.subheader("🤒 Symptom-Based Diagnosis")
        all_symptoms = mlb.classes_
        selected_symptoms = st.multiselect(
            "Select your symptoms:", sorted(all_symptoms)
        )

        if st.button("Predict Disease"):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                X_input = mlb.transform([selected_symptoms])
                pred_encoded = model.predict(X_input)[0]
                pred_disease = label_encoder.inverse_transform([pred_encoded])[0]
                st.session_state.predicted_disease = pred_disease
                st.markdown(
                    f"""
    <div style='
        background-color: #e6f7ff;
        color: #1a1a1a;
        border-left: 5px solid #1890ff;
        padding: 1rem;
        border-radius: 5px;
        font-weight: 600;
        margin-top: 1rem;
    '>
        🧾 <strong>Predicted Disease: {pred_disease}</strong>
    </div>
    """,
                    unsafe_allow_html=True,
                )

        if st.session_state.predicted_disease:
            disease = st.session_state.predicted_disease
            appointments = doctor_data.get(disease, [])

            if appointments:
                st.markdown("### 👨‍⚕️ Available Doctors:")
                for i, doc in enumerate(appointments):
                    available = doc["Availability"]
                    color = "green" if available else "red"
                    availability = "✅ Available" if available else "❌ Not Available"

                    st.markdown(
                        f"""
                    <div class="appointment-card">
                        <b>👨‍⚕️ {doc['Doctor_Name']}</b> — {doc['Speciality']}<br>
                        📍 <b>Location:</b> {doc['Location']}<br>
                        📅 <b>Date:</b> {doc['Date']} &nbsp;&nbsp;&nbsp; 🕒 <b>Time:</b> {doc['Time_Slot']}<br>
                        <span style='color:{color};'>{availability}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    if available:
                        if st.button(
                            f"Book Appointment with {doc['Doctor_Name']}",
                            key=f"book_{i}",
                        ):
                            record = {
                                "Email": st.session_state.user_email,
                                "Disease": disease,
                                "Doctor_Name": doc["Doctor_Name"],
                                "Speciality": doc["Speciality"],
                                "Location": doc["Location"],
                                "Date": doc["Date"],
                                "Time_Slot": doc["Time_Slot"],
                                "Timestamp": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                            }
                            df_new = pd.DataFrame([record])
                            if os.path.exists(APPOINTMENT_LOG):
                                df_new.to_csv(
                                    APPOINTMENT_LOG, mode="a", index=False, header=False
                                )
                            else:
                                df_new.to_csv(APPOINTMENT_LOG, index=False)

                            st.session_state.appointment_done = True
                            st.markdown(
                                f"""
    <div style='
        background-color: #e6f7ff;
        color: #1a1a1a;
        border-left: 5px solid #52c41a;
        padding: 1rem;
        border-radius: 5px;
        font-weight: 600;
        margin-top: 1rem;
    '>
        📩 Appointment booked with <strong>{doc['Doctor_Name']}</strong>!<br>
        Confirmation sent to <strong>{st.session_state.user_email}</strong> (simulated).
    </div>
    """,
                                unsafe_allow_html=True,
                            )

                            st.rerun()
            else:
                st.info("No doctors found for this disease.")

    with tab2:
        if os.path.exists(APPOINTMENT_LOG):
            df_log = pd.read_csv(APPOINTMENT_LOG)
            user_log = df_log[df_log["Email"] == st.session_state.user_email]
            if not user_log.empty:
                st.dataframe(user_log.sort_values("Timestamp", ascending=False))
            else:
                st.info("You have not booked any appointments yet.")
        else:
            st.info("No appointment log found.")
