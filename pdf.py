





from typing import List
import streamlit as st
import pdfplumber
# import re
# import tempfile
from gtts import gTTS
from langchain import PromptTemplate
import google.generativeai as genai
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from google.oauth2 import service_account
# from google.cloud import firestore
# from datetime import datetime
# import bcrypt  # For hashing passwords

# # ---------------------- Firestore Setup ----------------------
# def get_firestore_client():
#     firebase_creds = st.secrets["firebase"]
#     creds = service_account.Credentials.from_service_account_info(firebase_creds)
#     db = firestore.Client(credentials=creds, project=firebase_creds["project_id"])
#     return db

# def upload_report_to_firestore(structured_data, summary, doctor_id=None):
#     """Upload a patient report (structured data, summary, doctor_id, timestamp) to Firestore."""
#     try:
#         db = get_firestore_client()
#         doc_data = {
#             "structured_data": structured_data,
#             "summary": summary,
#             "doctor_id": doctor_id,
#             "timestamp": firestore.SERVER_TIMESTAMP  # record submission time
#         }
#         db.collection('Reports').add(doc_data)
#         st.write("Report uploaded successfully")
#         return True
#     except Exception as e:
#         st.write(e)
#         return False

# def save_doctor_to_firestore(name, email, password, qr_url):
#     """Save new doctor information to Firestore in a 'Doctors' collection with a hashed password."""
#     try:
#         db = get_firestore_client()
#         # Hash the password using bcrypt
#         hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
#         doctor_data = {
#             "name": name,
#             "email": email,
#             "password": hashed_pw,  # store hashed password
#             "qr_url": qr_url,
#             "created_at": firestore.SERVER_TIMESTAMP
#         }
#         # Use email as the document ID
#         db.collection('Doctors').document(email).set(doctor_data)
#         st.write("Doctor registered successfully")
#         return doctor_data
#     except Exception as e:
#         st.write(e)
#         return None

# def get_doctor_from_firestore(email):
#     """Retrieve a doctor's document from Firestore using their email."""
#     try:
#         db = get_firestore_client()
#         doc = db.collection('Doctors').document(email).get()
#         if doc.exists:
#             return doc.to_dict()
#         else:
#             return None
#     except Exception as e:
#         st.write(e)
#         return None

# # ---------------------- Global Functions ----------------------
@st.cache_data
def cached_get_pdf_text(doc):
    return get_pdf_text(doc)

@st.cache_data
def cached_summarize_lab_report(lab_data_str: str) -> str:
    
    return summarize_lab_report(lab_data_str)

def get_pdf_text(doc):
    full_text = ""
    if doc.type == "application/pdf":
        with pdfplumber.open(doc) as pdf_file:
            for page in pdf_file.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
    else:  # assume TXT file
        full_text = doc.getvalue().decode("utf-8")
    
    lines = full_text.splitlines()

    def is_test_line(line: str) -> bool:
        pattern = re.compile(
            r'^(?P<test_name>.+?)\s+'
            r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
        )
        return bool(pattern.search(line))

    def group_test_chunks(lines: List[str]) -> List[List[str]]:
        chunks = []
        current_chunk = []
        for line in lines:
            if is_test_line(line):
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [line]
            else:
                if current_chunk:
                    current_chunk.append(line)
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def parse_test_info(first_line: str):
        main_test_pattern = re.compile(
            r'^(?P<test_name>.+?)\s+'
            r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
        )
        match = main_test_pattern.search(first_line)
        if not match:
            return None, None
        return match.groupdict(), match

    def extract_ranges_with_labels(chunk: List[str]) -> List[dict]:
        range_pattern = re.compile(
            r'(?P<label>[A-Za-z\s()]+):?\s*'
            r'(?:(?P<operator><|>|<=|>=)?\s*(?P<lower>\d+(?:\.\d+)?))?'
            r'(?:\s*(?:-|–|to)\s*(?P<upper>\d+(?:\.\d+)?))?\s*(?P<unit>[^\s]+)?'
        )
        remaining_text = " ".join(chunk).strip()
        matches = range_pattern.finditer(remaining_text)
        labeled_ranges = []
        for match in matches:
            label_data = {
                "label": (match.group("label") or "").strip(),
                "lower": float(match.group("lower")) if match.group("lower") else None,
                "upper": float(match.group("upper")) if match.group("upper") else None,
                "operator": match.group("operator"),
                "unit": match.group("unit")
            }
            labeled_ranges.append(label_data)
        return labeled_ranges

    def filter_ranges(test_data: dict) -> dict:
        test_name = test_data.get("test_name", "").strip().lower()
        try:
            result_value = float(test_data.get("result", "0"))
        except ValueError:
            result_value = None
        valid_ranges = []
        for r in test_data.get("ranges_with_labels", []):
            label = (r.get("label") or "").strip().lower()
            if label == test_name and result_value is not None and r.get("lower") == result_value:
                continue
            if r.get("lower") is None and r.get("upper") is None:
                continue
            valid_ranges.append(r)
        test_data["ranges_with_labels"] = valid_ranges
        return test_data

    test_chunks = group_test_chunks(lines)
    extracted_data = []
    for chunk in test_chunks:
        first_line = chunk[0]
        test_info, _ = parse_test_info(first_line)
        if not test_info:
            continue
        ranges_with_labels = extract_ranges_with_labels(chunk)
        test_info["ranges_with_labels"] = ranges_with_labels
        extracted_data.append(test_info)
    extracted_data = [filter_ranges(test) for test in extracted_data]
    return full_text, extracted_data

def format_lab_data_for_prompt(extracted_data: List[dict]) -> str:
    formatted = ""
    for test in extracted_data:
        test_name = test.get("test_name", "")
        result = test.get("result", "")
        test_unit = test.get("unit", "")
        ranges_with_labels = test.get("ranges_with_labels", [])
        range_str_list = []
        for range_item in ranges_with_labels:
            label = range_item.get("label", "")
            lower = range_item.get("lower")
            upper = range_item.get("upper")
            operator = range_item.get("operator", "")
            ref_unit = range_item.get("unit", test_unit)
            unit_str = f" {ref_unit}" if test_unit == ref_unit and ref_unit else ""
            if lower is not None and upper is not None:
                if label:
                    range_str_list.append(f"{label}: {lower}-{upper}{unit_str}")
                else:
                    range_str_list.append(f"{lower}-{upper}{unit_str}")
            elif operator and lower is not None:
                if label:
                    range_str_list.append(f"{label}: {operator} {lower}{unit_str}")
                else:
                    range_str_list.append(f"{operator} {lower}{unit_str}")
        formatted_ranges_str = "; ".join(range_str_list)
        formatted += f"- {test_name}: {result} {test_unit} (Reference Range: {formatted_ranges_str})\n"
    return formatted

def summarize_lab_report(lab_data_str: str) -> str:
    genai.configure(api_key="AIzaSyB_V0B3ttXMYLn-4md_jEq_PdDRz7BJ0tM")
    prompt_template = PromptTemplate(
        input_variables=["lab_data"],
        template=(
            "You are a medical assistant with expertise in lab report analysis. Below is structured lab report data "
            "in bullet-point format, where each entry is a medically relevant lab test result. Please ignore any extraneous information such as addresses, administrative details, guidelines, or commentary that are not actual lab test results.\n\n"
            "Each test entry includes the test name, numeric result, unit, and reference intervals "
            "which may or may not have labels (e.g., Low (desirable): <200 mg/dL, Moderate (borderline): 200–239 mg/dL, High: ≥240 mg/dL).\n\n"
            "For each test:\n"
            "1. If labeled intervals are present: Determine which labeled interval the numeric result falls into and use that label as the classification.\n"
            "2. If NO labels are present but there is a reference range: Compare the value to the range and classify as:\n"
            "   - 'Normal' if the value is within the reference range\n"
            "   - 'Abnormal - Low' if the value is below the reference range\n"
            "   - 'Abnormal - High' if the value is above the reference range\n"
            "3. If neither labels nor ranges are present: Classify as 'No reference range available'\n\n"
            "Output a final summary in a bullet list format:\n"
            "• [Test Name]: [Numeric Value] - [Classification]\n\n"
            "Do not include any intermediate steps or additional text.\n\n"
            "Lab Report Data:\n{lab_data}\n\n"
            "Final Summary:"
        )
    )
    compiled_prompt = prompt_template.format(lab_data=lab_data_str)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(compiled_prompt)
    summary = response.text
    return summary

def text_to_speech(text: str) -> str:
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        audio_file = f.name
    return audio_file

def parse_summary_to_dataframe(summary_text):
    lines = summary_text.strip().split('\n')
    data = []
    for line in lines:
        if not line or line.strip() in ['•', '-']:
            continue
        line = line.strip().lstrip('•').lstrip('-').strip()
        parts = line.split(':')
        if len(parts) >= 2:
            test_name = parts[0].strip()
            value_status = parts[1].strip()
            value_status_parts = value_status.split('-')
            if len(value_status_parts) >= 2:
                value = value_status_parts[0].strip()
                status = '-'.join(value_status_parts[1:]).strip()
                data.append({
                    'Test': test_name,
                    'Value': value,
                    'Status': status
                })
    return pd.DataFrame(data)

def categorize_tests(df):
    categories = {
        'Complete Blood Count': [
            'Hemoglobin', 'RBC', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW-CV',
            'Total Leucocyte Count', 'Neutrophils', 'Lymphocytes', 'Monocytes',
            'Eosinophils', 'Absolute Neutrophil Count', 'Absolute Lymphocyte Count',
            'Absolute Monocyte Count', 'Absolute Eosinophil Count', 'Absolute Basophil Count',
            'Platelet Count', 'MPV', 'PDW', 'Erythrocyte Sedimentation Rate'
        ],
        'Lipid Profile': [
            'Cholesterol - Total', 'Triglycerides', 'Cholesterol - HDL', 
            'Cholesterol - LDL', 'Cholesterol- VLDL', 'Cholesterol : HDL Cholesterol',
            'LDL : HDL Cholesterol', 'Non HDL Cholesterol'
        ],
        'Glucose Metabolism': [
            'Glucose - Fasting', 'Glucose', 'HbA1c', 'Insulin'
        ],
        'Liver Function': [
            'Bilirubin', 'ALT', 'AST', 'ALP', 'GGT', 'Protein - Total', 'Albumin'
        ],
        'Kidney Function': [
            'Urea', 'Creatinine', 'Uric Acid', 'eGFR'
        ],
        'Electrolytes': [
            'Sodium', 'Potassium', 'Chloride', 'Calcium', 'Phosphorus', 'Magnesium'
        ],
        'Other Tests': []
    }
    df['Category'] = 'Other Tests'
    for category, tests in categories.items():
        for test in tests:
            df.loc[df['Test'].str.contains(test, case=False), 'Category'] = category
    return df

def create_gauge_chart(value, min_val, max_val, status):
    try:
        value = float(value)
        fig, ax = plt.subplots(figsize=(3, 0.5))
        is_dark_mode = plt.rcParams["axes.facecolor"] == "#1e1e1e"
        if is_dark_mode:
            fig.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            text_color = '#e0e0e0'
            bar_color = '#4a5568'
        else:
            fig.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            text_color = '#333333'
            bar_color = 'lightgray'
        if 'normal' in status.lower():
            color = '#4ade80' if is_dark_mode else 'green'
        elif 'high' in status.lower() or 'risk' in status.lower():
            color = '#f87171' if is_dark_mode else 'red'
        elif 'intermediate' in status.lower() or 'borderline' in status.lower():
            color = '#fbbf24' if is_dark_mode else 'orange'
        else:
            color = '#60a5fa' if is_dark_mode else 'blue'
        if min_val is None and max_val is None:
            min_val = value * 0.5
            max_val = value * 1.5
        elif min_val is None:
            min_val = max_val * 0.5
        elif max_val is None:
            max_val = min_val * 1.5
        plot_value = max(min_val, min(value, max_val))
        ax.barh(0, max_val - min_val, left=min_val, height=0.3, color=bar_color)
        ax.barh(0, plot_value - min_val, left=min_val, height=0.3, color=color)
        ax.plot(plot_value, 0, 'o', color=text_color, markersize=8)
        ax.set_xlim(min_val * 0.9, max_val * 1.1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        ax.text(min_val, -0.4, f'{min_val}', ha='center', fontsize=8, color=text_color)
        ax.text(max_val, -0.4, f'{max_val}', ha='center', fontsize=8, color=text_color)
        ax.text(plot_value, 0.4, f'{value}', ha='center', fontsize=9, fontweight='bold', color=text_color)
        return fig
    except Exception as e:
        st.write(e)
        return None

def script(summary):
    test_results = "Here are the detailed test results:\n" + summary
    genai.configure(api_key="AIzaSyB_V0B3ttXMYLn-4md_jEq_PdDRz7BJ0tM")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        f"Summarize the following test results and create a brief script of how a lab assistant will quickly tell the doctor about this lab result of a patient: {test_results}"
    )
    script_text = response.text
    return script_text

# # ---------------------- URL Query Parameters ----------------------
# params = st.experimental_get_query_params()
# # If role=doctor, show doctor interface; otherwise default to patient.
# role = params.get("role", ["patient"])[0]
# # For patient uploads, the doctor_id should be passed via the URL.
# doctor_id_param = params.get("doctor_id", [None])[0]

# # ---------------------- Session State Initialization ----------------------
# if 'summary' not in st.session_state:
#     st.session_state.summary = None
# if 'api_called' not in st.session_state:
#     st.session_state.api_called = False
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'lab_data_str' not in st.session_state:
#     st.session_state.lab_data_str = None
# if 'extracted_data' not in st.session_state:
#     st.session_state.extracted_data = None
# if 'doctor_logged_in' not in st.session_state:
#     st.session_state.doctor_logged_in = False
# if 'doctor_info' not in st.session_state:
#     st.session_state.doctor_info = None

# # ---------------------- Sidebar ----------------------
# with st.sidebar:
#     st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
#     st.title("Lab Report Analyzer")
#     st.markdown("---")
#     st.markdown("### For Doctors")
#     st.markdown("This portal allows doctors to create accounts, log in, and view patient lab reports with detailed analysis.")
#     st.markdown("---")
#     if role == "doctor":
#         if not st.session_state.doctor_logged_in:
#             st.subheader("Doctor Login / Registration")
#             option = st.radio("Select Option", ["Login", "Register"])
#             if option == "Login":
#                 email = st.text_input("Email", key="login_email")
#                 password = st.text_input("Password", type="password", key="login_password")
#                 if st.button("Login"):
#                     doctor = get_doctor_from_firestore(email)
#                     if doctor is None:
#                         st.error("Doctor not found. Please register first.")
#                     elif not bcrypt.checkpw(password.encode('utf-8'), doctor.get("password").encode('utf-8')):
#                         st.error("Incorrect password.")
#                     else:
#                         st.session_state.doctor_logged_in = True
#                         st.session_state.doctor_info = doctor
#                         st.success(f"Logged in as {doctor.get('name')}")
#                         st.experimental_rerun()
#             else:  # Registration
#                 reg_name = st.text_input("Name", key="reg_name")
#                 reg_email = st.text_input("Email", key="reg_email")
#                 reg_password = st.text_input("Password", type="password", key="reg_password")
#                 if st.button("Register"):
#                     base_url = "https://clinicassist.streamlit.app/"  # Update to your actual app URL.
#                     qr_data = f"{base_url}/?doctor_id={reg_email}"
#                     qr_url = f"https://api.qrserver.com/v1/create-qr-code/?data={qr_data}&size=150x150"
#                     doctor = save_doctor_to_firestore(reg_name, reg_email, reg_password, qr_url)
#                     if doctor:
#                         st.session_state.doctor_logged_in = True
#                         st.session_state.doctor_info = doctor
#                         st.success(f"Registered and logged in as {reg_name}")
#                         st.experimental_rerun()
#         else:
#             st.success(f"Logged in as {st.session_state.doctor_info.get('name')}")
#     st.markdown("---")
#     if st.button("Clear Analysis"):
#         st.session_state.summary = None
#         st.session_state.api_called = False
#         st.session_state.df = None
#         st.session_state.lab_data_str = None
#         st.session_state.extracted_data = None
#         st.experimental_rerun()
#     st.caption("© 2025 Medical Lab Analyzer")

# # ---------------------- Main Content ----------------------
# if role == "doctor":
#     # ------------------ Doctor Interface ------------------
#     st.header("Doctor Dashboard")
#     st.subheader("Patient Lab Reports Queue")
#     db = get_firestore_client()
#     # Query reports for the logged-in doctor, sorted by submission time (earlier first)
#     reports_query = db.collection("Reports")\
#         .where("doctor_id", "==", st.session_state.doctor_info.get("email"))\
#         .order_by("timestamp", direction=firestore.Query.ASCENDING)
#     reports = reports_query.stream()
#     any_reports = False
#     for report in reports:
#         any_reports = True
#         report_data = report.to_dict()
#         st.markdown(f"**Report ID:** {report.id}")
#         submitted_at = report_data.get("timestamp")
#         if submitted_at:
#             try:
#                 submitted_str = submitted_at.strftime("%Y-%m-%d %H:%M:%S")
#             except Exception:
#                 submitted_str = str(submitted_at)
#         else:
#             submitted_str = "N/A"
#         st.markdown(f"**Submitted:** {submitted_str}")
#         st.markdown(f"**Summary:** {report_data.get('summary', 'No summary available')}")
#         if st.button("View Detailed Report", key=report.id):
#             st.write("### Detailed Report")
#             df = parse_summary_to_dataframe(report_data.get("summary", ""))
#             df = categorize_tests(df)
#             tab1, tab2, tab3 = st.tabs(["📋 Summary View", "📊 Detailed Analysis", "📝 Raw Data"])
#             with tab1:
#                 st.dataframe(df, use_container_width=True)
#             with tab2:
#                 for idx, row in df.iterrows():
#                     st.markdown(f"**{row['Test']}**: {row['Value']} – {row['Status']}")
#                     try:
#                         if "high" in row['Status'].lower():
#                             min_val = float(row['Value']) * 0.7
#                             max_val = float(row['Value']) * 0.9
#                         elif "low" in row['Status'].lower():
#                             min_val = float(row['Value']) * 1.1
#                             max_val = float(row['Value']) * 1.3
#                         else:
#                             min_val = float(row['Value']) * 0.8
#                             max_val = float(row['Value']) * 1.2
#                         gauge_chart = create_gauge_chart(row['Value'], min_val, max_val, row['Status'])
#                         if gauge_chart:
#                             st.pyplot(gauge_chart)
#                     except Exception as e:
#                         st.write(e)
#                     st.markdown("---")
#             with tab3:
#                 st.dataframe(df, use_container_width=True)
#         st.markdown("---")
#     if not any_reports:
#         st.info("No reports available yet.")
#     st.subheader("Your Patient Submission QR Code")
#     qr_url = st.session_state.doctor_info.get("qr_url")
#     if qr_url:
#         st.image(qr_url)
#         st.markdown(f"Share this URL with patients: {qr_url.split('=')[1]}")
#     else:
#         st.info("QR code not available.")
# else:
#     # ------------------ Patient Interface ------------------
#     st.header("Patient Lab Report Submission")
#     st.markdown("""
#     **Instructions:**  
#     Please visit the URL provided by your doctor (or scan their QR code) so that your lab report is sent to the correct doctor. Then, upload your lab report document (PDF or TXT) below.
#     """)
#     # For patients, we use the doctor_id from the URL query parameter.
#     doctor_id = doctor_id_param
#     if not doctor_id:
#         st.error("Doctor ID not found. Please ensure you are using the correct link provided by your doctor.")
#     else:
#         uploaded_file = st.file_uploader("Upload Your Lab Report", type=["pdf", "txt"])
#         if uploaded_file is not None:
#             if not st.session_state.extracted_data:
#                 with st.spinner("Processing your file..."):
#                     full_text, extracted_data = cached_get_pdf_text(uploaded_file)
#                     st.session_state.extracted_data = extracted_data
#                     lab_data_str = format_lab_data_for_prompt(extracted_data)
#                     st.session_state.lab_data_str = lab_data_str
#             # Patients do not see the extracted text; they only see instructions.
#             if st.button("Submit Lab Report"):
#                 with st.spinner("Analyzing and submitting your lab report..."):
#                     summary = cached_summarize_lab_report(st.session_state.lab_data_str)
#                     st.session_state.summary = summary
#                     upload_report_to_firestore(st.session_state.lab_data_str, summary, doctor_id)
#                     st.success("Your lab report has been submitted successfully!\nPlease contact your doctor for further details.")




### -------- NEW CODE -------######




import uuid
import tempfile
import re
from datetime import datetime
import bcrypt
from supabase import create_client, Client
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- SUPABASE INITIALIZATION ----------------------
supabase_url = "https://yiatqptwrhrofodxvoey.supabase.co"  #st.secrets["SUPABASE"]["SUPABASE_URL"]
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlpYXRxcHR3cmhyb2ZvZHh2b2V5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQyMjAyNjQsImV4cCI6MjA1OTc5NjI2NH0.wqlViqrGv0PO2tZ5l23fxliBbaWDSJo6KBDBXRqm4DI" #st.secrets["SUPABASE"]["SUPABASE_ANON_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# ---------------------- HELPER FUNCTIONS ----------------------
# def get_pdf_text(pdf_file) -> tuple[str, list]:
#     """Extract text from PDF using pdfplumber and split it into test entries."""
#     full_text = ""
#     with pdfplumber.open(pdf_file) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 full_text += page_text + "\n"
#     lines = full_text.splitlines()

#     def is_test_line(line: str) -> bool:
#         pattern = re.compile(r'^(?P<test_name>.+?)\s+(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?')
#         return bool(pattern.search(line))

#     def group_test_chunks(lines: list) -> list:
#         chunks = []
#         current_chunk = []
#         for line in lines:
#             if is_test_line(line):
#                 if current_chunk:
#                     chunks.append(current_chunk)
#                 current_chunk = [line]
#             else:
#                 if current_chunk:
#                     current_chunk.append(line)
#         if current_chunk:
#             chunks.append(current_chunk)
#         return chunks

#     def parse_test_info(first_line: str) -> tuple[dict, any]:
#         pattern = re.compile(r'^(?P<test_name>.+?)\s+(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?')
#         match = pattern.search(first_line)
#         if not match:
#             return None, None
#         return match.groupdict(), match

#     def extract_ranges(chunk: list) -> list:
#         # This function can be improved; for now, it extracts any numeric ranges
#         pattern = re.compile(r'(?P<label>[A-Za-z\s()]+):?\s*(?P<range>\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?)')
#         text = " ".join(chunk)
#         matches = pattern.finditer(text)
#         ranges = []
#         for m in matches:
#             ranges.append(m.groupdict())
#         return ranges

#     chunks = group_test_chunks(lines)
#     extracted = []
#     for chunk in chunks:
#         info, _ = parse_test_info(chunk[0])
#         if info:
#             info["ranges"] = extract_ranges(chunk)
#             extracted.append(info)
#     return full_text, extracted

# def format_lab_data_for_prompt(extracted_data: list) -> str:
#     formatted = ""
#     for test in extracted_data:
#         test_name = test.get("test_name", "")
#         result = test.get("result", "")
#         unit = test.get("unit", "")
#         ranges = test.get("ranges", [])
#         ranges_str = "; ".join([f"{r['label'].strip()}: {r['range']}" for r in ranges]) if ranges else "N/A"
#         formatted += f"- {test_name}: {result} {unit} (Ranges: {ranges_str})\n"
#     return formatted

# def summarize_lab_report(lab_data_str: str) -> str:
#     """
#     Dummy summarization function.
#     Replace this with your LLM API call (e.g., Gemini, GPT) as needed.
#     """
#     # For now, we return a dummy summary.
#     return "Dummy summary: All test values are within normal ranges."

def create_gauge_chart(value: float, min_val: float, max_val: float, status: str):
    """Creates a simple gauge chart using matplotlib."""
    fig, ax = plt.subplots(figsize=(3, 0.5))
    # For simplicity, just show a horizontal bar.
    ax.barh(0, max_val - min_val, left=min_val, height=0.3, color="lightgray")
    ax.barh(0, value - min_val, left=min_val, height=0.3, color="green" if status.lower()=="normal" else "red")
    ax.set_xlim(min_val, max_val)
    ax.axis("off")
    return fig

def upload_file_to_storage(file_obj, folder="lab-reports"):
    file_name = f"{uuid.uuid4()}.pdf"
    res = supabase.storage.from_(folder).upload(file_name, file_obj.getvalue())
    public_url = supabase.storage.from_(folder).get_public_url(file_name)
    return public_url

# ---------------------- DOCTOR AUTHENTICATION ----------------------
# def doctor_signup(name: str, email: str, password: str):
#     """Sign up a new doctor using Supabase Auth and store info in 'users' table."""
#     auth_res = supabase.auth.sign_up({"email": email, "password": password})
#     if auth_res.user:
#         user_data = {
#             "id": auth_res.user.id,
#             "name": name,
#             "email": email,
#             "role": "doctor"
#         }
#         supabase.table("users").insert(user_data).execute()
#         return auth_res.user
#     else:
#         st.error("Sign up failed.")
#         return None
def doctor_signup(name: str, email: str, password: str):
    # Sign up the user
    auth_res = supabase.auth.sign_up({"email": email, "password": password})
    if auth_res.user:
        # Immediately sign in to ensure the JWT is set
        sign_in_res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if sign_in_res.user:
            user_id = sign_in_res.user.id
            user_data = {
                "id": user_id,  # This must equal auth.uid() in your RLS policy
                "name": name,
                "email": email,
                "role": "doctor"
            }
            # Now insert the new user record
            supabase.table("users").insert(user_data).execute()
            return sign_in_res.user
        else:
            st.error("Sign in after sign up failed.")
            return None
    else:
        st.error("Sign up failed.")
        return None

def doctor_login(email: str, password: str):
    """Log in a doctor using Supabase Auth."""
    auth_res = supabase.auth.sign_in_with_password({"email": email, "password": password})
    if auth_res.user:
        data = supabase.table("users").select("*").eq("id", auth_res.user.id).execute()
        if data.data:
            return data.data[0]
        else:
            st.error("Doctor record not found.")
            return None
    else:
        st.error("Login failed.")
        return None

# ---------------------- PATIENT SUBMISSION PAGE ----------------------
def patient_submission_page(doctor_id: str):
    st.title("Submit Your Lab Report")
    st.markdown("Please enter your name and upload your lab report (PDF).")
    with st.form("patient_form"):
        patient_name = st.text_input("Your Name")
        uploaded_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])
        submitted = st.form_submit_button("Submit Report")
        if submitted:
            if not patient_name or not uploaded_file:
                st.error("Please provide your name and upload a file.")
            else:
                st.write("This is entering the summary phase")
                full_text, extracted_data = get_pdf_text(uploaded_file)
                lab_data_str = format_lab_data_for_prompt(extracted_data)
                st.write("Extracted data:", lab_data_str)
                summary = summarize_lab_report(lab_data_str)
                st.write("Summary done")
                file_url = upload_file_to_storage(uploaded_file)
                # Omit created_at so default now() is used
                st.write("Done with summary phase, uploading to Supabase")
                report_data = {
                    "id": str(uuid.uuid4()),
                    "doctor_id": doctor_id,
                    "patient_name": patient_name,
                    "file_url": file_url,
                    "summary": summary
                }
                supabase.table("lab_reports").insert(report_data).execute()
                st.success("Your lab report has been submitted successfully!")
                st.markdown("### Summary")
                st.write(summary)

# ---------------------- DOCTOR DASHBOARD ----------------------
# def doctor_dashboard(doctor: dict):
#     st.title("Doctor Dashboard")
#     st.markdown(f"Welcome, Dr. {doctor['name']}!")
#     st.markdown("Below is the queue of lab reports submitted by your patients (oldest first).")
#     res = supabase.table("lab_reports").select("*").eq("doctor_id", doctor["email"]).order("created_at", desc=False).execute()
#     reports = res.data
#     if not reports:
#         st.info("No lab reports submitted yet.")
#     else:
#         for report in reports:
#             st.markdown(f"**Patient:** {report['patient_name']} | **Submitted:** {report['created_at']}")
#             st.markdown(f"**Summary:** {report['summary'][:100]}...")
#             if st.button("View Details", key=report["id"]):
#                 st.subheader("Detailed Report")
#                 st.write("**Patient Name:**", report["patient_name"])
#                 st.write("**File URL:**", report["file_url"])
#                 st.write("**Full Summary:**", report["summary"])
#                 st.markdown("**Additional Insights:**")
#                 # Here, you can add further processing (e.g., generating charts)
#                 st.pyplot(create_gauge_chart(70, 50, 100, "Normal"))
#             st.markdown("---")

def doctor_dashboard(doctor: dict):
    st.header("Doctor Dashboard")
    st.markdown(f"Welcome, Dr. {doctor['name']}!")
    
    # Generate and display the doctor's patient submission URL.
    base_url = st.secrets["APP"]["APP_URL"]  # Set this in your secrets, e.g., "https://yourappurl.com"
    custom_url = f"{base_url}/?doctor_id={doctor['email']}"
    st.markdown("### Your Patient Submission URL:")
    st.code(custom_url)
    
    st.subheader("Patient Lab Reports Queue")
    res = supabase.table("lab_reports").select("*").eq("doctor_id", doctor["email"]).order("created_at", desc=False).execute()
    reports = res.data
    any_reports = False
    if not reports:
        st.info("No lab reports submitted yet.")
    else:
        for report in reports:
            any_reports = True
            st.markdown(f"**Report ID:** {report['id']}")
            submitted_at = report.get("created_at", "N/A")
            try:
                submitted_str = datetime.fromisoformat(submitted_at).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                submitted_str = submitted_at
            st.markdown(f"**Submitted:** {submitted_str}")
            st.markdown(f"**Summary:** {report.get('summary', 'No summary available')}")
            if st.button("View Detailed Report", key=report["id"]):
                st.write("### Detailed Report")
                # Parse the summary into a DataFrame for better visualization.
                df = parse_summary_to_dataframe(report.get("summary", ""))
                df = categorize_tests(df)
                tab1, tab2, tab3 = st.tabs(["📋 Summary View", "📊 Detailed Analysis", "📝 Raw Data"])
                with tab1:
                    st.dataframe(df, use_container_width=True)
                with tab2:
                    for idx, row in df.iterrows():
                        st.markdown(f"**{row['Test']}**: {row['Value']} – {row['Status']}")
                        try:
                            if "high" in row['Status'].lower():
                                min_val = float(row['Value']) * 0.7
                                max_val = float(row['Value']) * 0.9
                            elif "low" in row['Status'].lower():
                                min_val = float(row['Value']) * 1.1
                                max_val = float(row['Value']) * 1.3
                            else:
                                min_val = float(row['Value']) * 0.8
                                max_val = float(row['Value']) * 1.2
                            gauge_chart = create_gauge_chart(float(row['Value']), min_val, max_val, row['Status'])
                            if gauge_chart:
                                st.pyplot(gauge_chart)
                        except Exception as e:
                            st.write(e)
                        st.markdown("---")
                with tab3:
                    st.dataframe(df, use_container_width=True)
                # Generate a script (text) from the full summary.
                script_text = script(report.get("summary", ""))
                st.markdown("**Script Summary (to be read aloud):**")
                st.write(script_text)
                if st.button("Play Script", key=f"play_{report['id']}"):
                    audio_path = text_to_speech(script_text)
                    st.audio(audio_path, format="audio/mp3")
            st.markdown("---")
    st.subheader("Your Patient Submission QR Code")
    qr_url = doctor.get("qr_url")
    if qr_url:
        st.image(qr_url)
        st.markdown(f"Share this URL with patients: {qr_url.split('=')[1]}")
    else:
        st.info("QR code not available.")

# ---------------------- MAIN APP ROUTING ----------------------
def main():
    query_params = st.query_params
    # If the URL has a "doctor_id" query parameter, show the patient submission page.

    if "doctor_id" in query_params:
        doctor_id = query_params["doctor_id"]
        patient_submission_page(doctor_id)
    else:
        # Otherwise, show doctor authentication and dashboard.
        if "doctor" not in st.session_state:
            st.sidebar.title("Doctor Authentication")
            auth_mode = st.sidebar.radio("Select Option", ["Log In", "Sign Up"])
            if auth_mode == "Sign Up":
                name = st.sidebar.text_input("Full Name")
                email = st.sidebar.text_input("Email")
                password = st.sidebar.text_input("Password", type="password")
                if st.sidebar.button("Sign Up"):
                    user = doctor_signup(name, email, password)
                    if user:
                        st.session_state.doctor = {"id": user.id, "name": name, "email": email}
                        st.success("Signed up successfully!")
            else:  # Log In
                email = st.sidebar.text_input("Email", key="login_email")
                password = st.sidebar.text_input("Password", type="password", key="login_password")
                if st.sidebar.button("Log In"):
                    doctor_data = doctor_login(email, password)
                    if doctor_data:
                        st.session_state.doctor = doctor_data
                        st.success(f"Welcome, Dr. {doctor_data['name']}!")
        else:
            # Show the doctor dashboard if logged in
            doctor_dashboard(st.session_state.doctor)

if __name__ == "__main__":
    main()
