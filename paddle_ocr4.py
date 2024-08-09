from paddleocr import PaddleOCR, draw_ocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

def extract_text_and_table(image_path):
    # Initialize the PaddleOCR instance
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # use angle classifier and English language

    # Run OCR on the image
    result = ocr.ocr(image_path, cls=True)

    # Draw results on the image
    image = cv2.imread(image_path)
    boxes = [res[0] for res in result]
    texts = [res[1][0] for res in result]
    scores = [res[1][1] for res in result]

    # Visualize the results
    for box, text, score in zip(boxes, texts, scores):
        print(f'Text: {text}, Confidence: {score}')
        box = np.array(box).astype(int)
        cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the image with detected text
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = './sample/CM-HU1157_10.png'  # Replace with your image path
    extract_text_and_table(image_path)
