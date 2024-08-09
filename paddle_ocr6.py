from paddleocr import PaddleOCR, PPStructure, draw_structure_result
import cv2
import openpyxl
from openpyxl import Workbook

# Initialize PaddleOCR with the text detection and recognition models
ocr = PaddleOCR(det_model_dir="./models/ppstructure/inference/text_detection/ch_PP-OCRv3_det_infer", 
                rec_model_dir="./models/ppstructure/inference/text_recognition/ch_PP-OCRv3_rec_infer", 
                use_angle_cls=True)

# Initialize PPStructure for table extraction
table_engine = PPStructure(table_model_dir="./models/ppstructure/inference/table_recognition/en_ppocr_mobile_v2.0_table_structure_infer", lang="en")

# Load image
image_path = "./sample/sample.png"
image = cv2.imread(image_path)

# Perform text detection and recognition
text_result = ocr.ocr(image_path, cls=True, det=True, rec=True)

# Perform table structure recognition
table_result = table_engine(image_path)

# Extract table data
table_data = []
for item in table_result:
    if 'html' in item:
        html = item['html']['structure']
        cells = item['html']['cells']
        for row in html['rows']:
            row_data = []
            for cell in row['cells']:
                cell_index = cell['index']
                cell_text = cells[cell_index]['text']
                row_data.append(cell_text)
            table_data.append(row_data)

# Save extracted table data to Excel
excel_file_path = './extracted_table_data.xlsx'
wb = Workbook()
ws = wb.active

for row in table_data:
    ws.append(row)

wb.save(excel_file_path)

# Print extracted table data
for row in table_data:
    print(row)

# Visualize the table structure result
image = cv2.imread(image_path)
im_show = draw_structure_result(image, table_result, font_path="./fonts/german.ttf")
cv2.imwrite('result.jpg', im_show)
