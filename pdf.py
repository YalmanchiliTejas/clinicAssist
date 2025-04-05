# # from typing import List
# # import streamlit as st
# # import pdfplumber
# # import re
# # import tempfile
# # from gtts import gTTS
# # from langchain import PromptTemplate
# # import google.generativeai as genai
# # import os

# # # Configure Gemini 2.5 Pro API with your API key
# # genai.configure(api_key="AIzaSyB_V0B3ttXMYLn-4md_jEq_PdDRz7BJ0tM")

# # def get_pdf_text(pdf_doc):
# #     full_text = ""
    
# #     with pdfplumber.open(pdf_doc) as pdf_file:
# #         for page in pdf_file.pages:
# #             page_text = page.extract_text()
# #             if page_text:
# #                 full_text += page_text + "\n"
    
# #     # Split the extracted text into lines.
# #     lines = full_text.splitlines()
    
# #     # Helper function: Identify lines that likely begin a new test entry.
# #     def is_test_line(line: str) -> bool:
# #         pattern = re.compile(
# #             r'^(?P<test_name>.+?)\s+'
# #             r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
# #         )
# #         return bool(pattern.search(line))
    
# #     # Helper function: Group lines into test chunks.
# #     def group_test_chunks(lines: List[str]) -> List[List[str]]:
# #         chunks = []
# #         current_chunk = []
# #         for line in lines:
# #             if is_test_line(line):
# #                 if current_chunk:
# #                     chunks.append(current_chunk)
# #                 current_chunk = [line]
# #             else:
# #                 if current_chunk:
# #                     current_chunk.append(line)
# #         if current_chunk:
# #             chunks.append(current_chunk)
# #         return chunks

# #     # Parse test information from a test line.
# #     def parse_test_info(first_line: str):
# #         main_test_pattern = re.compile(
# #             r'^(?P<test_name>.+?)\s+'
# #             r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
# #         )
# #         match = main_test_pattern.search(first_line)
# #         if not match:
# #             return None, None
# #         return match.groupdict(), match

# #     # Extract labeled ranges within each test chunk.
# #     def extract_ranges_with_labels(chunk: List[str]) -> List[dict]:
# #         range_pattern = re.compile(
# #             r'(?P<label>[A-Za-z\s()]+):?\s*'
# #             r'(?:(?P<operator><|>|<=|>=)?\s*(?P<lower>\d+(?:\.\d+)?))?'
# #             r'(?:\s*(?:-|–|to)\s*(?P<upper>\d+(?:\.\d+)?))?\s*(?P<unit>[^\s]+)?'
# #         )
        
# #         remaining_text = "\n".join(chunk)
# #         matches = range_pattern.finditer(remaining_text)
        
# #         labeled_ranges = []
# #         for match in matches:
# #             label_data = {
# #                 "label": match.group("label").strip(),
# #                 "lower": float(match.group("lower")) if match.group("lower") else None,
# #                 "upper": float(match.group("upper")) if match.group("upper") else None,
# #                 "operator": match.group("operator"),
# #                 "unit": match.group("unit")
# #             }
# #             labeled_ranges.append(label_data)
        
# #         return labeled_ranges
    
# #     test_chunks = group_test_chunks(lines)
# #     extracted_data = []
# #     for chunk in test_chunks:
# #         first_line = chunk[0]
# #         test_info, _ = parse_test_info(first_line)
# #         if not test_info:
# #             continue
# #         ranges_with_labels = extract_ranges_with_labels(chunk)
# #         test_info["ranges_with_labels"] = ranges_with_labels
# #         extracted_data.append(test_info)
# #     return full_text, extracted_data

# # def format_lab_data_for_prompt(extracted_data: List[dict]) -> str:
# #     formatted = ""
# #     for test in extracted_data:
# #         test_name = test.get("test_name", "")
# #         result = test.get("result", "")
# #         unit = test.get("unit", "")
        
# #         ranges_with_labels = test.get("ranges_with_labels", [])
        
# #         range_str_list = []
# #         for range_item in ranges_with_labels:
# #             label = range_item.get("label", "")
# #             lower = range_item.get("lower", "")
# #             upper = range_item.get("upper", "")
# #             operator = range_item.get("operator", "")
# #             unit_range = range_item.get("unit", unit)  # Default to the same unit as the result
            
# #             if lower and upper:
# #                 range_str_list.append(f"{label}: {lower}-{upper} {unit_range}")
# #             elif operator and lower:
# #                 range_str_list.append(f"{label}: {operator} {lower} {unit_range}")
        
# #         formatted_ranges_str = "; ".join(range_str_list)
        
# #         formatted += f"- {test_name}: {result} {unit} (Ranges: {formatted_ranges_str})\n"
    
# #     return formatted

# # def summarize_lab_report(lab_data_str: str) -> str:
# #     """
# #     Uses the Gemini 2.5 Pro API to generate a summary of the lab report.
# #     It builds a prompt based on the structured lab data and sends it to the Gemini model,
# #     now including multiple labeled intervals.
# #     """
# #     prompt_template = PromptTemplate(
# #         input_variables=["lab_data"],
# #         template=(
# #             "You are a medical assistant with expertise in lab report analysis. Below is structured lab report data "
# #             "in bullet-point format, where each test entry includes the test name, numeric result, unit, and multiple "
# #             "labeled reference intervals (e.g., Low (desirable): <200 mg/dL, Moderate (borderline): 200–239 mg/dL, High: ≥240 mg/dL).\n\n"
# #             "For each test, determine which labeled interval the numeric result falls into, and then output a final summary "
# #             "in a bullet list where each bullet contains only the test name, the numeric result, and the matching classification label.\n\n"
# #             "Output format:\n"
# #             "• [Test Name]: [Numeric Value] - [Classification]\n\n"
# #             "Do not include any intermediate steps or additional text.\n\n"
# #             "Lab Report Data:\n{lab_data}\n\n"
# #             "Final Summary:"
# #         )
# #     )
    
# #     compiled_prompt = prompt_template.format(lab_data=lab_data_str)
    
# #     # Call Gemini 2.5 Pro API to generate the summary.
# #     model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
# #     response = model.generate_content(compiled_prompt)
# #     summary = response.text
# #     return summary

# # def text_to_speech(text: str) -> str:
# #     """
# #     Converts the provided text to speech using gTTS.
# #     Returns the path to the generated MP3 file.
# #     """
# #     tts = gTTS(text=text, lang="en")
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
# #         tts.save(f.name)
# #         audio_file = f.name
# #     return audio_file

# # # ------------------- Streamlit App -------------------

# # st.title("PDF Lab Report Summarizer with Text-to-Speech")

# # uploaded_file = st.file_uploader("Upload a PDF lab report", type="pdf")

# # if uploaded_file is not None:
# #     with st.spinner("Extracting text and summarizing the lab report..."):
# #         _, extracted_data = get_pdf_text(uploaded_file)
# #         lab_data_str = format_lab_data_for_prompt(extracted_data)
# #         summary = summarize_lab_report(lab_data_str)
    
# #     st.subheader("Summary")
# #     st.write(summary)
    
# #     if st.button("Speak Summary"):
# #         audio_path = text_to_speech(summary)
# #         st.audio(audio_path, format="audio/mp3")
# #         # Optionally, you can remove the temporary file afterward:
# #         # os.remove(audio_path)
# from typing import List
# import streamlit as st
# import pdfplumber
# import re
# import tempfile
# from gtts import gTTS
# from langchain import PromptTemplate
# import google.generativeai as genai
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Page configuration
# st.set_page_config(
#     page_title="Medical Lab Report Analyzer",
#     page_icon="🩺",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main {
#         padding: 1rem;
#     }
#     .normal {
#         color: #0f5132;
#         background-color: #d1e7dd;
#         padding: 0.2rem 0.5rem;
#         border-radius: 0.25rem;
#         font-weight: 500;
#     }
#     .abnormal {
#         color: #842029;
#         background-color: #f8d7da;
#         padding: 0.2rem 0.5rem;
#         border-radius: 0.25rem;
#         font-weight: 500;
#     }
#     .intermediate {
#         color: #664d03;
#         background-color: #fff3cd;
#         padding: 0.2rem 0.5rem;
#         border-radius: 0.25rem;
#         font-weight: 500;
#     }
#     .report-header {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#         border-left: 5px solid #0d6efd;
#     }
#     .category-header {
#         background-color: #e9ecef;
#         padding: 0.5rem 1rem;
#         border-radius: 0.25rem;
#         margin: 1rem 0 0.5rem 0;
#         font-weight: 600;
#     }
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 2px;
#     }
#     .stTabs [data-baseweb="tab"] {
#         height: 50px;
#         white-space: pre-wrap;
#         background-color: #f8f9fa;
#         border-radius: 4px 4px 0 0;
#         gap: 1px;
#         padding-top: 10px;
#         padding-bottom: 10px;
#     }
#     .stTabs [aria-selected="true"] {
#         background-color: #e9ecef;
#         border-bottom: 2px solid #0d6efd;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Configure Gemini 2.5 Pro API with your API key
# genai.configure(api_key="AIzaSyB_V0B3ttXMYLn-4md_jEq_PdDRz7BJ0tM")

# # def get_pdf_text(pdf_doc):
# #     full_text = ""
    
# #     with pdfplumber.open(pdf_doc) as pdf_file:
# #         for page in pdf_file.pages:
# #             page_text = page.extract_text()
# #             if page_text:
# #                 full_text += page_text + "\n"
    
# #     # Split the extracted text into lines.
# #     lines = full_text.splitlines()
    
# #     # Helper function: Identify lines that likely begin a new test entry.
# #     def is_test_line(line: str) -> bool:
# #         pattern = re.compile(
# #             r'^(?P<test_name>.+?)\s+'
# #             r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
# #         )
# #         return bool(pattern.search(line))
    
# #     # Helper function: Group lines into test chunks.
# #     def group_test_chunks(lines: List[str]) -> List[List[str]]:
# #         chunks = []
# #         current_chunk = []
# #         for line in lines:
# #             if is_test_line(line):
# #                 if current_chunk:
# #                     chunks.append(current_chunk)
# #                 current_chunk = [line]
# #             else:
# #                 if current_chunk:
# #                     current_chunk.append(line)
# #         if current_chunk:
# #             chunks.append(current_chunk)
# #         return chunks

# #     # Parse test information from a test line.
# #     def parse_test_info(first_line: str):
# #         main_test_pattern = re.compile(
# #             r'^(?P<test_name>.+?)\s+'
# #             r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
# #         )
# #         match = main_test_pattern.search(first_line)
# #         if not match:
# #             return None, None
# #         return match.groupdict(), match

# #     # Extract labeled ranges within each test chunk.
# #     def extract_ranges_with_labels(chunk: List[str]) -> List[dict]:
# #         range_pattern = re.compile(
# #             r'(?P<label>[A-Za-z\s()]+):?\s*'
# #             r'(?:(?P<operator><|>|<=|>=)?\s*(?P<lower>\d+(?:\.\d+)?))?'
# #             r'(?:\s*(?:-|–|to)\s*(?P<upper>\d+(?:\.\d+)?))?\s*(?P<unit>[^\s]+)?'
# #         )
        
# #         remaining_text = "\n".join(chunk)
# #         matches = range_pattern.finditer(remaining_text)
        
# #         labeled_ranges = []
# #         for match in matches:
# #             label_data = {
# #                 "label": match.group("label").strip(),
# #                 "lower": float(match.group("lower")) if match.group("lower") else None,
# #                 "upper": float(match.group("upper")) if match.group("upper") else None,
# #                 "operator": match.group("operator"),
# #                 "unit": match.group("unit")
# #             }
# #             labeled_ranges.append(label_data)
        
# #         return labeled_ranges
    
# #     test_chunks = group_test_chunks(lines)
# #     extracted_data = []
# #     for chunk in test_chunks:
# #         first_line = chunk[0]
# #         test_info, _ = parse_test_info(first_line)
# #         if not test_info:
# #             continue
# #         ranges_with_labels = extract_ranges_with_labels(chunk)
# #         test_info["ranges_with_labels"] = ranges_with_labels
# #         extracted_data.append(test_info)
# #     return full_text, extracted_data

# # def format_lab_data_for_prompt(extracted_data: List[dict]) -> str:
# #     formatted = ""
# #     for test in extracted_data:
# #         test_name = test.get("test_name", "")
# #         result = test.get("result", "")
# #         unit = test.get("unit", "")
        
# #         ranges_with_labels = test.get("ranges_with_labels", [])
        
# #         range_str_list = []
# #         for range_item in ranges_with_labels:
# #             label = range_item.get("label", "")
# #             lower = range_item.get("lower", "")
# #             upper = range_item.get("upper", "")
# #             operator = range_item.get("operator", "")
# #             unit_range = range_item.get("unit", unit)  # Default to the same unit as the result
            
# #             if lower and upper:
# #                 range_str_list.append(f"{label}: {lower}-{upper} {unit_range}")
# #             elif operator and lower:
# #                 range_str_list.append(f"{label}: {operator} {lower} {unit_range}")
        
# #         formatted_ranges_str = "; ".join(range_str_list)
        
# #         formatted += f"- {test_name}: {result} {unit} (Ranges: {formatted_ranges_str})\n"
    
# #     return formatted

# # def summarize_lab_report(lab_data_str: str) -> str:
# #     """
# #     Uses the Gemini 2.5 Pro API to generate a summary of the lab report.
# #     It builds a prompt based on the structured lab data and sends it to the Gemini model,
# #     now including multiple labeled intervals.
# #     """
# #     prompt_template = PromptTemplate(
# #         input_variables=["lab_data"],
# #         template=(
# #             "You are a medical assistant with expertise in lab report analysis. Below is structured lab report data "
# #             "in bullet-point format, where each test entry includes the test name, numeric result, unit, and multiple "
# #             "labeled reference intervals (e.g., Low (desirable): <200 mg/dL, Moderate (borderline): 200–239 mg/dL, High: ≥240 mg/dL).\n\n"
# #             "For each test, determine which labeled interval the numeric result falls into, and then output a final summary "
# #             "in a bullet list where each bullet contains only the test name, the numeric result, and the matching classification label.\n\n"
# #             "Output format:\n"
# #             "• [Test Name]: [Numeric Value] - [Classification]\n\n"
# #             "Do not include any intermediate steps or additional text.\n\n"
# #             "Lab Report Data:\n{lab_data}\n\n"
# #             "Final Summary:"
# #         )
# #     )
    
# #     compiled_prompt = prompt_template.format(lab_data=lab_data_str)
    
# #     # Call Gemini 2.5 Pro API to generate the summary.
# #     model = genai.GenerativeModel("gemini-2.0-flash")
# #     response = model.generate_content(compiled_prompt)
# #     summary = response.text
# #     return summary
# def get_pdf_text(pdf_doc):
#     full_text = ""
    
#     with pdfplumber.open(pdf_doc) as pdf_file:
#         for page in pdf_file.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 full_text += page_text + "\n"
    
#     # Split the extracted text into lines.
#     lines = full_text.splitlines()
    
#     # Helper function: Identify lines that likely begin a new test entry.
#     def is_test_line(line: str) -> bool:
#         pattern = re.compile(
#             r'^(?P<test_name>.+?)\s+'
#             r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
#         )
#         return bool(pattern.search(line))
    
#     # Helper function: Group lines into test chunks.
#     def group_test_chunks(lines: List[str]) -> List[List[str]]:
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

#     # Parse test information from a test line.
#     def parse_test_info(first_line: str):
#         main_test_pattern = re.compile(
#             r'^(?P<test_name>.+?)\s+'
#             r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
#         )
#         match = main_test_pattern.search(first_line)
#         if not match:
#             return None, None
#         return match.groupdict(), match

#     # Extract labeled ranges within each test chunk.
#     def extract_ranges_with_labels(chunk: List[str]) -> List[dict]:
#         range_pattern = re.compile(
#             r'(?P<label>[A-Za-z\s()]+):?\s*'
#             r'(?:(?P<operator><|>|<=|>=)?\s*(?P<lower>\d+(?:\.\d+)?))?'
#             r'(?:\s*(?:-|–|to)\s*(?P<upper>\d+(?:\.\d+)?))?\s*(?P<unit>[^\s]+)?'
#         )
#         # range_pattern = re.compile(
#         #     r'(?:'                                   # Non-capturing group for the optional label
#         #     r'(?P<label>[A-Za-z\s()/-]+):\s*'       # Label: letters, spaces, parentheses or dashes followed by a colon and optional spaces
#         #     r')?'
#         #     r'(?P<operator><|>|<=|>=)?\s*'            # Optional operator
#         #     r'(?P<lower>\d+(?:\.\d+)?)'               # Lower numeric value
#         #     r'(?:\s*(?:-|–|to)\s*(?P<upper>\d+(?:\.\d+)?))?'  # Optional separator and upper value
#         #     r'\s*(?P<unit>[^\s]+)'                    # Unit (one or more non-space characters)
#         # )
     
        
#         # remaining_text = "\n".join(chunk)
#         remaining_text = " ".join(chunk).strip()
#         matches = range_pattern.finditer(remaining_text)
        
#         labeled_ranges = []
#         for match in matches:
#             label_data = {
#                 "label": (match.group("label") or "").strip(),
#                 "lower": float(match.group("lower")) if match.group("lower") else None,
#                 "upper": float(match.group("upper")) if match.group("upper") else None,
#                 "operator": match.group("operator"),
#                 "unit": match.group("unit")
#             }
#             labeled_ranges.append(label_data)
        
#         return labeled_ranges
#     def filter_ranges(test_data: dict) -> dict:
#         test_name = test_data.get("test_name", "").strip().lower()
#         try:
#             result_value = float(test_data.get("result", "0"))
#         except ValueError:
#             result_value = None
#         valid_ranges = []
#         for r in test_data.get("ranges_with_labels", []):
#             label = (r.get("label") or "").strip().lower()
#             # Skip if the label duplicates the test name and the lower value equals the test result.
#             if label == test_name and result_value is not None and r.get("lower") == result_value:
#                 continue
#             # Skip if no numeric bounds are available.
#             if r.get("lower") is None and r.get("upper") is None:
#                 continue
#             valid_ranges.append(r)
#         test_data["ranges_with_labels"] = valid_ranges
#         return test_data
    
#     test_chunks = group_test_chunks(lines)
#     extracted_data = []
#     for chunk in test_chunks:
#         first_line = chunk[0]
#         test_info, _ = parse_test_info(first_line)
#         if not test_info:
#             continue
#         ranges_with_labels = extract_ranges_with_labels(chunk)
#         test_info["ranges_with_labels"] = ranges_with_labels
#         extracted_data.append(test_info)
#     extracted_data = [filter_ranges(test) for test in extracted_data]
#     print("Extracted Data:", extracted_data, flush=True)
#     return full_text, extracted_data

# # def format_lab_data_for_prompt(extracted_data: List[dict]) -> str:
# #     formatted = ""
# #     for test in extracted_data:
# #         test_name = test.get("test_name", "")
# #         result = test.get("result", "")
# #         unit = test.get("unit", "")
        
# #         ranges_with_labels = test.get("ranges_with_labels", [])
        
# #         range_str_list = []
# #         for range_item in ranges_with_labels:
# #             label = range_item.get("label", "")
# #             lower = range_item.get("lower", "")
# #             upper = range_item.get("upper", "")
# #             operator = range_item.get("operator", "")
# #             unit_range = range_item.get("unit", unit)  # Default to the same unit as the result
            
# #             if lower and upper:
# #                 range_str_list.append(f"{label}: {lower}-{upper} {unit_range}")
# #             elif operator and lower:
# #                 range_str_list.append(f"{label}: {operator} {lower} {unit_range}")
        
# #         formatted_ranges_str = "; ".join(range_str_list)
        
# #         formatted += f"- {test_name}: {result} {unit} (Ranges: {formatted_ranges_str})\n"
    
# #     return formatted

# def format_lab_data_for_prompt(extracted_data: List[dict]) -> str:
#     formatted = ""
#     for test in extracted_data:
#         test_name = test.get("test_name", "")
#         result = test.get("result", "")
#         test_unit = test.get("unit", "")
        
#         ranges_with_labels = test.get("ranges_with_labels", [])
        
#         range_str_list = []
#         for range_item in ranges_with_labels:
#             label = range_item.get("label", "")
#             lower = range_item.get("lower")
#             upper = range_item.get("upper")
#             operator = range_item.get("operator", "")
#             # Get the unit from the reference range; default to the test unit if not provided.
#             ref_unit = range_item.get("unit", test_unit)
            
#             # If the test unit and the reference range unit differ, omit the reference unit.
#             if test_unit != ref_unit:
#                 unit_str = ""
#             else:
#                 unit_str = f" {ref_unit}" if ref_unit else ""
            
#             # Format the range string based on available numbers.
#             if lower is not None and upper is not None:
#                 if label:
#                     range_str_list.append(f"{label}: {lower}-{upper}{unit_str}")
#                 else:
#                     range_str_list.append(f"{lower}-{upper}{unit_str}")
#             elif operator and lower is not None:
#                 if label:
#                     range_str_list.append(f"{label}: {operator} {lower}{unit_str}")
#                 else:
#                     range_str_list.append(f"{operator} {lower}{unit_str}")
        
#         formatted_ranges_str = "; ".join(range_str_list)
#         formatted += f"- {test_name}: {result} {test_unit} (Reference Range: {formatted_ranges_str})\n"
    
#     return formatted

# def summarize_lab_report(lab_data_str: str) -> str:
#     """
#     Uses the Gemini 2.5 Pro API to generate a summary of the lab report.
#     It builds a prompt based on the structured lab data and sends it to the Gemini model,
#     now including multiple labeled intervals.
#     """
#     prompt_template = PromptTemplate(
#         input_variables=["lab_data"],
#         template=(
#             # "You are a medical assistant with expertise in lab report analysis. Below is structured lab report data "
#             # "in bullet-point format, where each test entry includes the test name, numeric result, unit, and multiple "
#             # "labeled reference intervals (e.g., Low (desirable): <200 mg/dL, Moderate (borderline): 200–239 mg/dL, High: ≥240 mg/dL).\n\n"
#             # "For each test, determine which labeled interval the numeric result falls into, and then output a final summary "
#             # "in a bullet list where each bullet contains only the test name, the numeric result, and the matching classification label.\n\n"
#             # "Output format:\n"
#             # "• [Test Name]: [Numeric Value] - [Classification]\n\n"
#             # "Do not include any intermediate steps or additional text.\n\n"
#             # "Lab Report Data:\n{lab_data}\n\n"
#             # "Final Summary:"
#             "You are a medical assistant with expertise in lab report analysis. Below is structured lab report data "
#             "in bullet-point format, where each entry is a medically relevant lab test result. Please ignore any extraneous information such as addresses, administrative details, guidelines, or commentary that are not actual lab test results.\n\n"
#             "Each test entry includes the test name, numeric result, unit, and reference intervals "
#             "which may or may not have labels (e.g., Low (desirable): <200 mg/dL, Moderate (borderline): 200–239 mg/dL, High: ≥240 mg/dL).\n\n"
#             "For each test:\n"
#             "1. If labeled intervals are present: Determine which labeled interval the numeric result falls into and use that label as the classification.\n"
#             "2. If NO labels are present but there is a reference range: Compare the value to the range and classify as:\n"
#             "   - 'Normal' if the value is within the reference range\n"
#             "   - 'Abnormal - Low' if the value is below the reference range\n"
#             "   - 'Abnormal - High' if the value is above the reference range\n"
#             "3. If neither labels nor ranges are present: Classify as 'No reference range available'\n\n"
#             "Output a final summary in a bullet list format:\n"
#             "• [Test Name]: [Numeric Value] - [Classification]\n\n"
#             "Do not include any intermediate steps or additional text.\n\n"
#             "Lab Report Data:\n{lab_data}\n\n"
#             "Final Summary:"

#         )
#     )
    
#     compiled_prompt = prompt_template.format(lab_data=lab_data_str)
    
#     # Call Gemini 2.5 Pro API to generate the summary.
#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(compiled_prompt)
#     summary = response.text
#     print("done:", summary)
#     return summary

# def text_to_speech(text: str) -> str:
#     """
#     Converts the provided text to speech using gTTS.
#     Returns the path to the generated MP3 file.
#     """
#     tts = gTTS(text=text, lang="en")
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
#         tts.save(f.name)
#         audio_file = f.name
#     return audio_file

# def parse_summary_to_dataframe(summary_text):
#     """Parse the summary text into a structured DataFrame"""
#     lines = summary_text.strip().split('\n')
#     data = []
    
#     for line in lines:
#         # Skip empty lines or bullet points without content
#         if not line or line.strip() in ['•', '-']:
#             continue
            
#         # Remove bullet points if present
#         line = line.strip().lstrip('•').lstrip('-').strip()
        
#         # Extract test name, value, and status
#         parts = line.split(':')
#         if len(parts) >= 2:
#             test_name = parts[0].strip()
#             value_status = parts[1].strip()
            
#             # Split value and status
#             value_status_parts = value_status.split('-')
#             if len(value_status_parts) >= 2:
#                 value = value_status_parts[0].strip()
#                 status = '-'.join(value_status_parts[1:]).strip()
                
#                 # Add to data
#                 data.append({
#                     'Test': test_name,
#                     'Value': value,
#                     'Status': status
#                 })
    
#     return pd.DataFrame(data)

# def categorize_tests(df):
#     """Categorize tests into groups for better organization"""
#     # Define categories and their associated tests
#     categories = {
#         'Complete Blood Count': [
#             'Hemoglobin', 'RBC', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW-CV',
#             'Total Leucocyte Count', 'Neutrophils', 'Lymphocytes', 'Monocytes',
#             'Eosinophils', 'Absolute Neutrophil Count', 'Absolute Lymphocyte Count',
#             'Absolute Monocyte Count', 'Absolute Eosinophil Count', 'Absolute Basophil Count',
#             'Platelet Count', 'MPV', 'PDW', 'Erythrocyte Sedimentation Rate'
#         ],
#         'Lipid Profile': [
#             'Cholesterol - Total', 'Triglycerides', 'Cholesterol - HDL', 
#             'Cholesterol - LDL', 'Cholesterol- VLDL', 'Cholesterol : HDL Cholesterol',
#             'LDL : HDL Cholesterol', 'Non HDL Cholesterol'
#         ],
#         'Glucose Metabolism': [
#             'Glucose - Fasting', 'Glucose', 'HbA1c', 'Insulin'
#         ],
#         'Liver Function': [
#             'Bilirubin', 'ALT', 'AST', 'ALP', 'GGT', 'Protein - Total', 'Albumin'
#         ],
#         'Kidney Function': [
#             'Urea', 'Creatinine', 'Uric Acid', 'eGFR'
#         ],
#         'Electrolytes': [
#             'Sodium', 'Potassium', 'Chloride', 'Calcium', 'Phosphorus', 'Magnesium'
#         ],
#         'Other Tests': []  # For tests that don't fit in other categories
#     }
    
#     # Add a category column to the DataFrame
#     df['Category'] = 'Other Tests'
    
#     # Assign categories based on test names
#     for category, tests in categories.items():
#         for test in tests:
#             df.loc[df['Test'].str.contains(test, case=False), 'Category'] = category
    
#     return df

# def create_gauge_chart(value, min_val, max_val, status):
#     """Create a gauge chart to visualize where a value falls in its reference range"""
#     try:
#         value = float(value)
        
#         # Set colors based on status
#         if 'normal' in status.lower():
#             color = 'green'
#         elif 'high' in status.lower() or 'risk' in status.lower():
#             color = 'red'
#         elif 'intermediate' in status.lower() or 'borderline' in status.lower():
#             color = 'orange'
#         else:
#             color = 'blue'
        
#         # Create the gauge chart
#         fig, ax = plt.subplots(figsize=(3, 0.5))
        
#         # Determine range
#         if min_val is None and max_val is None:
#             min_val = value * 0.5
#             max_val = value * 1.5
#         elif min_val is None:
#             min_val = max_val * 0.5
#         elif max_val is None:
#             max_val = min_val * 1.5
            
#         # Ensure value is within the range for visualization
#         plot_value = max(min_val, min(value, max_val))
        
#         # Create a horizontal bar
#         ax.barh(0, max_val - min_val, left=min_val, height=0.3, color='lightgray')
#         ax.barh(0, plot_value - min_val, left=min_val, height=0.3, color=color)
        
#         # Add a marker for the actual value
#         ax.plot(plot_value, 0, 'o', color='black', markersize=8)
        
#         # Set limits and remove axes
#         ax.set_xlim(min_val * 0.9, max_val * 1.1)
#         ax.set_ylim(-0.5, 0.5)
#         ax.axis('off')
        
#         # Add min and max values as text
#         ax.text(min_val, -0.4, f'{min_val}', ha='center', fontsize=8)
#         ax.text(max_val, -0.4, f'{max_val}', ha='center', fontsize=8)
#         ax.text(plot_value, 0.4, f'{value}', ha='center', fontsize=9, fontweight='bold')
        
#         return fig
#     except:
#         return None

# def script(summary):
#     test_results = "Here are the detailed test results:\n" + summary
#     # Call Gemini 2.5 Pro API to generate the summary.
#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(f"Summarize the following test results and create a brief script of how a lab assistant will quickly tell the doctor about this lab result of a patient: {test_results}")
#     script = response.text
#     print("Generated Script:", script)
#     return summary
# # ------------------- Streamlit App -------------------

# # Sidebar
# with st.sidebar:
#     st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
#     st.title("Lab Report Analyzer")
#     st.markdown("---")
#     st.markdown("### Quick Analysis for Busy Doctors")
#     st.markdown("""
#     This tool helps you quickly analyze patient lab reports by:
#     - Highlighting abnormal values
#     - Categorizing results by test type
#     - Providing visual indicators
#     - Offering audio summaries
#     """)
#     st.markdown("---")
    
#     # Filter options
#     st.subheader("Filter Options")
#     show_only_abnormal = st.checkbox("Show only abnormal results", value=False)
    
#     # Audio options
#     st.subheader("Audio Options")
#     voice_speed = st.slider("Speech Rate", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
    
#     st.markdown("---")
#     st.caption("© 2025 Medical Lab Analyzer")

# # Main content
# st.markdown('<div class="report-header"><h1>📊 Patient Lab Report Analyzer</h1><p>Upload a patient\'s lab report PDF to get a quick analysis of their results.</p></div>', unsafe_allow_html=True)

# # File uploader in the main area
# uploaded_file = st.file_uploader("Upload a PDF lab report", type="pdf")

# if uploaded_file is not None:
#     with st.spinner("Analyzing lab report..."):
#         _, extracted_data = get_pdf_text(uploaded_file)
#         lab_data_str = format_lab_data_for_prompt(extracted_data)
#         summary = summarize_lab_report(lab_data_str)
#         script(summary)
        
#         # Parse summary into DataFrame
#         df = parse_summary_to_dataframe(summary)
        
#         # Categorize tests
#         df = categorize_tests(df)
        
#         # Filter abnormal results if requested
#         if show_only_abnormal:
#             df = df[~df['Status'].str.contains('Normal', case=False)]
    
#     # Create tabs for different views
#     tab1, tab2, tab3 = st.tabs(["📋 Summary View", "📊 Detailed Analysis", "📝 Raw Data"])
    
#     with tab1:
#         # Summary statistics
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             normal_count = df[df['Status'].str.contains('Normal', case=False)].shape[0]
#             st.metric("Normal Results", normal_count)
#         with col2:
#             abnormal_count = df[~df['Status'].str.contains('Normal', case=False)].shape[0]
#             st.metric("Abnormal Results", abnormal_count, delta=f"{abnormal_count} need attention")
#         with col3:
#             total_count = df.shape[0]
#             st.metric("Total Tests", total_count)
        
#         st.markdown("---")
        
#         # Abnormal results highlight
#         if abnormal_count > 0:
#             st.markdown("### ⚠️ Abnormal Results")
#             abnormal_df = df[~df['Status'].str.contains('Normal', case=False)]
            
#             for _, row in abnormal_df.iterrows():
#                 status_class = "abnormal" if "high" in row['Status'].lower() or "risk" in row['Status'].lower() else "intermediate"
#                 st.markdown(f"""
#                 <div style="margin-bottom: 10px; padding: 10px; border-radius: 5px; border-left: 4px solid {'#842029' if status_class == 'abnormal' else '#664d03'}; background-color: {'#f8d7da' if status_class == 'abnormal' else '#fff3cd'}">
#                     <span style="font-weight: bold;">{row['Test']}:</span> {row['Value']} - <span class="{status_class}">{row['Status']}</span>
#                 </div>
#                 """, unsafe_allow_html=True)
        
#         # Audio summary
#         st.markdown("### 🔊 Audio Summary")
        
#         # Create a focused audio summary for abnormal results
#         if abnormal_count > 0:
#             audio_summary = f"Alert: {abnormal_count} abnormal results detected. "
#             for _, row in abnormal_df.iterrows():
#                 audio_summary += f"{row['Test']} is {row['Value']}, which is {row['Status']}. "
#         else:
#             audio_summary = "All test results are within normal ranges."
        
#         if st.button("Play Audio Summary"):
#             audio_path = text_to_speech(audio_summary)
#             st.audio(audio_path, format="audio/mp3")
    
#     with tab2:
#         # Group by categories
#         categories = df['Category'].unique()
        
#         for category in categories:
#             category_df = df[df['Category'] == category]
            
#             st.markdown(f'<div class="category-header">{category}</div>', unsafe_allow_html=True)
            
#             for _, row in category_df.iterrows():
#                 # Determine status class for styling
#                 if "normal" in row['Status'].lower():
#                     status_class = "normal"
#                 elif "high" in row['Status'].lower() or "risk" in row['Status'].lower():
#                     status_class = "abnormal"
#                 else:
#                     status_class = "intermediate"
                
#                 # Create columns for each test
#                 col1, col2, col3 = st.columns([3, 2, 3])
                
#                 with col1:
#                     st.markdown(f"**{row['Test']}**")
#                     st.markdown(f"Value: {row['Value']}")
#                     st.markdown(f'Status: <span class="{status_class}">{row["Status"]}</span>', unsafe_allow_html=True)
                
#                 with col2:
#                     # Try to extract reference ranges for visualization
#                     try:
#                         # This is a simplified approach - in a real app, you'd parse the actual ranges
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
#                     except:
#                         pass
                
#                 with col3:
#                     # Recommendations based on status
#                     if "normal" not in row['Status'].lower():
#                         st.markdown("**Recommendation:**")
#                         if "high" in row['Status'].lower():
#                             st.markdown("Consider follow-up testing or treatment for elevated levels.")
#                         elif "low" in row['Status'].lower():
#                             st.markdown("Consider follow-up testing or treatment for decreased levels.")
#                         elif "borderline" in row['Status'].lower() or "intermediate" in row['Status'].lower():
#                             st.markdown("Monitor in future tests. Consider lifestyle modifications.")
#                         elif "risk" in row['Status'].lower():
#                             st.markdown("⚠️ Requires attention. Consider appropriate intervention.")
                
#                 st.markdown("---")
    
#     with tab3:
#         # Raw data view
#         st.dataframe(
#             df,
#             column_config={
#                 "Test": "Test Name",
#                 "Value": "Result",
#                 "Status": st.column_config.TextColumn(
#                     "Status",
#                     help="Classification of the test result",
#                     width="medium"
#                 ),
#                 "Category": "Test Category"
#             },
#             use_container_width=True,
#             hide_index=True
#         )
        
#         # Export options
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Export as CSV"):
#                 csv = df.to_csv(index=False)
#                 st.download_button(
#                     label="Download CSV",
#                     data=csv,
#                     file_name="lab_report_analysis.csv",
#                     mime="text/csv"
#                 )
#         with col2:
#             if st.button("Print Report"):
#                 st.info("Preparing printer-friendly version...")
#                 # In a real app, you would generate a printer-friendly version here
# else:
#     # Display a placeholder when no file is uploaded
#     st.info("👆 Upload a patient's lab report PDF to get started")
    
#     # Sample image to show what the app does
#     col1, col2 = st.columns([1, 1])
#     with col1:
#         st.markdown("### How it works:")
#         st.markdown("""
#         1. Upload a patient's lab report PDF
#         2. Our AI analyzes the results
#         3. View a color-coded summary of all test results
#         4. Focus on abnormal values that need attention
#         5. Get audio summaries for quick review
#         """)
#     with col2:
#         st.image("https://img.icons8.com/color/240/000000/health-checkup.png", width=200)







from typing import List
import streamlit as st
import pdfplumber
import re
import tempfile
from gtts import gTTS
from langchain import PromptTemplate
import google.generativeai as genai
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Medical Lab Report Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark mode support
st.markdown("""
<style>
    :root {
        /* Light mode colors */
        --background-main: #ffffff;
        --text-main: #333333;
        --normal-text: #0f5132;
        --normal-bg: #d1e7dd;
        --abnormal-text: #842029;
        --abnormal-bg: #f8d7da;
        --intermediate-text: #664d03;
        --intermediate-bg: #fff3cd;
        --header-bg: #f8f9fa;
        --category-bg: #e9ecef;
        --border-accent: #0d6efd;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            /* Dark mode colors */
            --background-main: #1e1e1e;
            --text-main: #e0e0e0;
            --normal-text: #4ade80;
            --normal-bg: #134a2b;
            --abnormal-text: #f87171;
            --abnormal-bg: #4c1d1d;
            --intermediate-text: #fbbf24;
            --intermediate-bg: #433b10;
            --header-bg: #2d3748;
            --category-bg: #1a202c;
            --border-accent: #3b82f6;
        }
    }

    .main {
        padding: 1rem;
        background-color: var(--background-main);
        color: var(--text-main);
    }
    
    .normal {
        color: var(--normal-text);
        background-color: var(--normal-bg);
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 500;
    }
    
    .abnormal {
        color: var(--abnormal-text);
        background-color: var(--abnormal-bg);
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 500;
    }
    
    .intermediate {
        color: var(--intermediate-text);
        background-color: var(--intermediate-bg);
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 500;
    }
    
    .report-header {
        background-color: var(--header-bg);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid var(--border-accent);
    }
    
    .category-header {
        background-color: var(--category-bg);
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0 0.5rem 0;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--header-bg);
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--category-bg);
        border-bottom: 2px solid var(--border-accent);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching and API call management
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'api_called' not in st.session_state:
    st.session_state.api_called = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'lab_data_str' not in st.session_state:
    st.session_state.lab_data_str = None
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None

# Cache the PDF text extraction to prevent redundant processing
@st.cache_data
def cached_get_pdf_text(pdf_doc):
    return get_pdf_text(pdf_doc)

# Cache the API call to prevent redundant calls
@st.cache_data
def cached_summarize_lab_report(lab_data_str: str) -> str:
    # Configure Gemini API only when actually making the call
    genai.configure(api_key="AIzaSyB_V0B3ttXMYLn-4md_jEq_PdDRz7BJ0tM")
    return summarize_lab_report(lab_data_str)

def get_pdf_text(pdf_doc):
    full_text = ""
    
    with pdfplumber.open(pdf_doc) as pdf_file:
        for page in pdf_file.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    
    # Split the extracted text into lines.
    lines = full_text.splitlines()
    
    # Helper function: Identify lines that likely begin a new test entry.
    def is_test_line(line: str) -> bool:
        pattern = re.compile(
            r'^(?P<test_name>.+?)\s+'
            r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
        )
        return bool(pattern.search(line))
    
    # Helper function: Group lines into test chunks.
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

    # Parse test information from a test line.
    def parse_test_info(first_line: str):
        main_test_pattern = re.compile(
            r'^(?P<test_name>.+?)\s+'
            r'(?P<result>\d+(?:\.\d+)?)(?:\s+(?P<unit>[^\s]+))?'
        )
        match = main_test_pattern.search(first_line)
        if not match:
            return None, None
        return match.groupdict(), match

    # Extract labeled ranges within each test chunk.
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
            # Skip if the label duplicates the test name and the lower value equals the test result.
            if label == test_name and result_value is not None and r.get("lower") == result_value:
                continue
            # Skip if no numeric bounds are available.
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
            # Get the unit from the reference range; default to the test unit if not provided.
            ref_unit = range_item.get("unit", test_unit)
            
            # If the test unit and the reference range unit differ, omit the reference unit.
            if test_unit != ref_unit:
                unit_str = ""
            else:
                unit_str = f" {ref_unit}" if ref_unit else ""
            
            # Format the range string based on available numbers.
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
    """
    Uses the Gemini 2.5 Pro API to generate a summary of the lab report.
    It builds a prompt based on the structured lab data and sends it to the Gemini model,
    now including multiple labeled intervals.
    """
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
    
    # Call Gemini API to generate the summary.
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(compiled_prompt)
    summary = response.text
    return summary

def text_to_speech(text: str) -> str:
    """
    Converts the provided text to speech using gTTS.
    Returns the path to the generated MP3 file.
    """
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        audio_file = f.name
    return audio_file

def parse_summary_to_dataframe(summary_text):
    """Parse the summary text into a structured DataFrame"""
    lines = summary_text.strip().split('\n')
    data = []
    
    for line in lines:
        # Skip empty lines or bullet points without content
        if not line or line.strip() in ['•', '-']:
            continue
            
        # Remove bullet points if present
        line = line.strip().lstrip('•').lstrip('-').strip()
        
        # Extract test name, value, and status
        parts = line.split(':')
        if len(parts) >= 2:
            test_name = parts[0].strip()
            value_status = parts[1].strip()
            
            # Split value and status
            value_status_parts = value_status.split('-')
            if len(value_status_parts) >= 2:
                value = value_status_parts[0].strip()
                status = '-'.join(value_status_parts[1:]).strip()
                
                # Add to data
                data.append({
                    'Test': test_name,
                    'Value': value,
                    'Status': status
                })
    
    return pd.DataFrame(data)

def categorize_tests(df):
    """Categorize tests into groups for better organization"""
    # Define categories and their associated tests
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
        'Other Tests': []  # For tests that don't fit in other categories
    }
    
    # Add a category column to the DataFrame
    df['Category'] = 'Other Tests'
    
    # Assign categories based on test names
    for category, tests in categories.items():
        for test in tests:
            df.loc[df['Test'].str.contains(test, case=False), 'Category'] = category
    
    return df

def create_gauge_chart(value, min_val, max_val, status):
    """Create a gauge chart to visualize where a value falls in its reference range"""
    try:
        value = float(value)
        
        # Create the figure and axis with dark mode detection
        fig, ax = plt.subplots(figsize=(3, 0.5))
        
        # Check if dark mode is active based on figure background
        is_dark_mode = plt.rcParams["axes.facecolor"] == "#1e1e1e"
        
        # Set colors based on theme
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
        
        # Set colors based on status
        if 'normal' in status.lower():
            color = '#4ade80' if is_dark_mode else 'green'
        elif 'high' in status.lower() or 'risk' in status.lower():
            color = '#f87171' if is_dark_mode else 'red'
        elif 'intermediate' in status.lower() or 'borderline' in status.lower():
            color = '#fbbf24' if is_dark_mode else 'orange'
        else:
            color = '#60a5fa' if is_dark_mode else 'blue'
        
        # Determine range
        if min_val is None and max_val is None:
            min_val = value * 0.5
            max_val = value * 1.5
        elif min_val is None:
            min_val = max_val * 0.5
        elif max_val is None:
            max_val = min_val * 1.5
            
        # Ensure value is within the range for visualization
        plot_value = max(min_val, min(value, max_val))
        
        # Create a horizontal bar
        ax.barh(0, max_val - min_val, left=min_val, height=0.3, color=bar_color)
        ax.barh(0, plot_value - min_val, left=min_val, height=0.3, color=color)
        
        # Add a marker for the actual value
        ax.plot(plot_value, 0, 'o', color=text_color, markersize=8)
        
        # Set limits and remove axes
        ax.set_xlim(min_val * 0.9, max_val * 1.1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        
        # Add min and max values as text
        ax.text(min_val, -0.4, f'{min_val}', ha='center', fontsize=8, color=text_color)
        ax.text(max_val, -0.4, f'{max_val}', ha='center', fontsize=8, color=text_color)
        ax.text(plot_value, 0.4, f'{value}', ha='center', fontsize=9, fontweight='bold', color=text_color)
        
        return fig
    except:
        return None

def script(summary):
    """Generate a script for how a lab assistant would explain the results to a doctor"""
    test_results = "Here are the detailed test results:\n" + summary
    # Call Gemini API to generate the script, only when needed
    genai.configure(api_key="AIzaSyB_V0B3ttXMYLn-4md_jEq_PdDRz7BJ0tM")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Summarize the following test results and create a brief script of how a lab assistant will quickly tell the doctor about this lab result of a patient: {test_results}")
    script_text = response.text
    return script_text

# ------------------- Streamlit App -------------------

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.title("Lab Report Analyzer")
    st.markdown("---")
    st.markdown("### Quick Analysis for Busy Doctors")
    st.markdown("""
    This tool helps you quickly analyze patient lab reports by:
    - Highlighting abnormal values
    - Categorizing results by test type
    - Providing visual indicators
    - Offering audio summaries
    """)
    st.markdown("---")
    
    # Filter options
    st.subheader("Filter Options")
    show_only_abnormal = st.checkbox("Show only abnormal results", value=False)
    
    # Audio options
    st.subheader("Audio Options")
    voice_speed = st.slider("Speech Rate", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
    
    st.markdown("---")
    st.caption("© 2025 Medical Lab Analyzer")

# Main content
st.markdown('<div class="report-header"><h1>📊 Patient Lab Report Analyzer</h1><p>Upload a patient\'s lab report PDF to get a quick analysis of their results.</p></div>', unsafe_allow_html=True)

# File uploader in the main area
uploaded_file = st.file_uploader("Upload a PDF lab report", type="pdf")

if uploaded_file is not None:
    # Process the file only once and store the results in session state
    if not st.session_state.extracted_data:
        with st.spinner("Processing file..."):
            _, extracted_data = cached_get_pdf_text(uploaded_file)
            st.session_state.extracted_data = extracted_data
            lab_data_str = format_lab_data_for_prompt(extracted_data)
            st.session_state.lab_data_str = lab_data_str
    
    # Only make API calls when the button is clicked
    if st.button("Generate Analysis") or st.session_state.api_called:
        if not st.session_state.api_called:
            with st.spinner("Analyzing lab report..."):
                # Call the API and store results
                summary = cached_summarize_lab_report(st.session_state.lab_data_str)
                st.session_state.summary = summary
                st.session_state.api_called = True
                
                # Parse and categorize only once
                df = parse_summary_to_dataframe(summary)
                df = categorize_tests(df)
                st.session_state.df = df
        
        # Filter abnormal results if requested
        df = st.session_state.df
        if show_only_abnormal:
            df = df[~df['Status'].str.contains('Normal', case=False)]
    
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Summary View", "📊 Detailed Analysis", "📝 Raw Data"])
        
        with tab1:
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                normal_count = st.session_state.df[st.session_state.df['Status'].str.contains('Normal', case=False)].shape[0]
                st.metric("Normal Results", normal_count)
            with col2:
                abnormal_count = st.session_state.df[~st.session_state.df['Status'].str.contains('Normal', case=False)].shape[0]
                st.metric("Abnormal Results", abnormal_count, delta=f"{abnormal_count} need attention")
            with col3:
                total_count = st.session_state.df.shape[0]
                st.metric("Total Tests", total_count)
            
            st.markdown("---")
            
            # Abnormal results highlight
            if abnormal_count > 0:
                st.markdown("### ⚠️ Abnormal Results")
                abnormal_df = st.session_state.df[~st.session_state.df['Status'].str.contains('Normal', case=False)]
                
                for _, row in abnormal_df.iterrows():
                    status_class = "abnormal" if "high" in row['Status'].lower() or "risk" in row['Status'].lower() else "intermediate"
                    st.markdown(f"""
                    <div style="margin-bottom: 10px; padding: 10px; border-radius: 5px; border-left: 4px solid var(--{status_class}-text); background-color: var(--{status_class}-bg)">
                        <span style="font-weight: bold;">{row['Test']}:</span> {row['Value']} - <span class="{status_class}">{row['Status']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Audio summary
            st.markdown("### 🔊 Audio Summary")
            
            # Create a focused audio summary for abnormal results
            # if abnormal_count > 0:
            #     audio_summary = f"Alert: {abnormal_count} abnormal results detected. "
            #     abnormal_df = st.session_state.df[~st.session_state.df['Status'].str.contains('Normal', case=False)]
            #     for _, row in abnormal_df.iterrows():
            #         audio_summary += f"{row['Test']} is {row['Value']}, which is {row['Status']}. "
            # else:
            #     audio_summary = "All test results are within normal ranges."
            audio_summary = f"This patient's lab report contains {total_count} tests. "

            # Add information about abnormal results if present
            if abnormal_count > 0:
                audio_summary += f"{abnormal_count} tests show abnormal results. "

            # Now include ALL test results in the summary
            audio_summary += "Here is a complete summary of all test results: "
            for _, row in st.session_state.df.iterrows():
                audio_summary += f"{row['Test']} is {row['Value']}, which is {row['Status']}. "
            
            if st.button("Play Audio Summary"):
                audio_path = text_to_speech(audio_summary)
                st.audio(audio_path, format="audio/mp3")
        
        with tab2:
            # Group by categories
            categories = df['Category'].unique()
            
            for category in categories:
                category_df = df[df['Category'] == category]
                
                st.markdown(f'<div class="category-header">{category}</div>', unsafe_allow_html=True)
                
                for _, row in category_df.iterrows():
                    # Determine status class for styling
                    if "normal" in row['Status'].lower():
                        status_class = "normal"
                    elif "high" in row['Status'].lower() or "risk" in row['Status'].lower():
                        status_class = "abnormal"
                    else:
                        status_class = "intermediate"
                    
                    # Create columns for each test
                    col1, col2, col3 = st.columns([3, 2, 3])
                    
                    with col1:
                        st.markdown(f"**{row['Test']}**")
                        st.markdown(f"Value: {row['Value']}")
                        st.markdown(f'Status: <span class="{status_class}">{row["Status"]}</span>', unsafe_allow_html=True)
                    
                    with col2:
                        # Try to extract reference ranges for visualization
                        try:
                            # This is a simplified approach - in a real app, you'd parse the actual ranges
                            if "high" in row['Status'].lower():
                                min_val = float(row['Value']) * 0.7
                                max_val = float(row['Value']) * 0.9
                            elif "low" in row['Status'].lower():
                                min_val = float(row['Value']) * 1.1
                                max_val = float(row['Value']) * 1.3
                            else:
                                min_val = float(row['Value']) * 0.8
                                max_val = float(row['Value']) * 1.2
                            
                            gauge_chart = create_gauge_chart(row['Value'], min_val, max_val, row['Status'])
                            if gauge_chart:
                                st.pyplot(gauge_chart)
                        except:
                            pass
                    
                    with col3:
                        # Recommendations based on status
                        if "normal" not in row['Status'].lower():
                            st.markdown("**Recommendation:**")
                            if "high" in row['Status'].lower():
                                st.markdown("Consider follow-up testing or treatment for elevated levels.")
                            elif "low" in row['Status'].lower():
                                st.markdown("Consider follow-up testing or treatment for decreased levels.")
                            elif "borderline" in row['Status'].lower() or "intermediate" in row['Status'].lower():
                                st.markdown("Monitor in future tests. Consider lifestyle modifications.")
                            elif "risk" in row['Status'].lower():
                                st.markdown("⚠️ Requires attention. Consider appropriate intervention.")
                    
                    st.markdown("---")
        
        with tab3:
            # Raw data view
            st.dataframe(
                df,
                column_config={
                    "Test": "Test Name",
                    "Value": "Result",
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="Classification of the test result",
                        width="medium"
                    ),
                    "Category": "Test Category"
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export as CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="lab_report_analysis.csv",
                        mime="text/csv"
                    )
            with col2:
                if st.button("Print Report"):
                    st.info("Preparing printer-friendly version...")
                    # In a real app, you would generate a printer-friendly version here
    else:
        st.info("👆 Click 'Generate Analysis' to analyze this lab report")
else:
    # Display a placeholder when no file is uploaded
    st.info("👆 Upload a patient's lab report PDF to get started")
    
    # Sample image to show what the app does
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### How it works:")
        st.markdown("""
        1. Upload a patient's lab report PDF
        2. Click "Generate Analysis" to process the report
        3. View a color-coded summary of all test results
        4. Focus on abnormal values that need attention
        5. Get audio summaries for quick review
        """)
    with col2:
        st.image("https://img.icons8.com/color/240/000000/health-checkup.png", width=200)
