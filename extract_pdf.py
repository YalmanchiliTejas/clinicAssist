import pytesseract
from pdf2image import convert_from_path
import tempfile
import fitz
import argparse
import re
def extract_with_ocr(pdf_path):

  full_text = ""

  with tempfile.TemporaryDirectory() as path:
    images= convert_from_path(pdf_path, output_folder=path)

    for image in images:
      text = pytesseract.image_to_string(image)
      full_text += text + "\n"
  return full_text


def extract_medical_report(pdf_path):

  try:

    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):

      page = doc.load_page(page_num)
      text += page.get_text()
    doc.close()

    if len(text.strip()) < 100:
      return extract_with_ocr(pdf_path);
    return text
  except Exception as e:
    print(f"Error with text extraction: {e}")
    return extract_with_ocr(pdf_path);

def clean_lab_report(text):
  lines = text.split("\n")
  cleaned_lines = []
  for line in lines:
    if re.search(r'page \d+|\d+/\d+|confidential|laboratory report', line.lower()):
      continue
    cleaned_lines.append(line)
  cleaned_text = '\n'.join(cleaned_lines)
  cleaned_text = re.sub(r'(\d+)[.,](\d+)', r'\1.\2',cleaned_text)
  return cleaned_text



if __name__ == "__main__":
  

  parser = argparse.ArgumentParser()
  parser.add_argument("-f","--file",required=True)
  args = parser.parse_args()
  text = extract_medical_report(args.file)
  print(clean_lab_report(text), flush=True)
