# Import necessary libraries
from paddleocr import PaddleOCR
import pprint

# Initialize the OCR model with German language
ocr = PaddleOCR(use_angle_cls=True, lang='de')

# Path to the image file
image_path = "./MM-MP1044_5.png"

# Perform OCR on the image
result = ocr.ocr(image_path, cls=True)

# Process the OCR result to extract the data
def process_ocr_result(result):
    data = []
    current_entry = {}
    for line in result[0]:
        if isinstance(line, list) and len(line) > 1 and isinstance(line[1], list) and len(line[1]) > 0 and isinstance(line[1][0], str):
            text = line[1][0]
            # Define logic to extract and structure the text into the desired format
            if "Nr." in text:
                if current_entry:
                    data.append(current_entry)
                    current_entry = {}
                current_entry["Nr."] = text.split("Nr.")[-1].strip()
            elif "Artikel" in text:
                current_entry["Artikel"] = text.split("Artikel")[-1].strip()
            elif "B" in text:
                current_entry["B"] = text.split("B")[-1].strip()
            elif "H" in text:
                current_entry["H"] = text.split("H")[-1].strip()
            elif "T" in text:
                current_entry["T"] = text.split("T")[-1].strip()
            elif "Beschreibung" in text:
                current_entry["Beschreibung"] = text.split("Beschreibung")[-1].strip()
            elif "Presie" in text:
                presie_key = f"Presie {len([k for k in current_entry if 'Presie' in k]) + 1} (Kaltschaum + Federkern, Boxspring)"
                current_entry[presie_key] = text.split("Presie")[-1].strip()

    if current_entry:
        data.append(current_entry)
    
    return data

# Process the OCR result
extracted_data = process_ocr_result(result)

# Print the extracted data
pprint.pprint(extracted_data)
