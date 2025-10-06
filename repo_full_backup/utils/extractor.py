#pdf extraction using pdfplumber
import pdfplumber
from typing import List

def extract_text_from_pdf(file_path: str) -> List[str]:
    extracted_pages = []

   
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            #extracting texting, if empty returning none
            text = page.extract_text() or ""
            
            #removing whitespace
            text = text.replace('\n', ' ').strip()
            
            extracted_pages.append(text)

            print(f"Extracted Page {page_number}: {len(text)} characters")

    print(f"\nTotal Pages Extracted: {len(extracted_pages)}")
    return extracted_pages
