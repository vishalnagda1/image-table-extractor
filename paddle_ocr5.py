import cv2
import numpy as np
import os
from paddleocr import (
    PaddleOCR,
    PPStructure,
    draw_ocr,
    draw_structure_result,
    save_structure_res,
)
from PIL import Image


def improve_image_quality(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Denoising
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Convert to grayscale
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    hist_eq_image = cv2.equalizeHist(gray_image)

    # Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)

    # Sharpening using a kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(denoised_image, -1, kernel)

    # Resize with interpolation
    height, width = denoised_image.shape[:2]
    resized_image = cv2.resize(denoised_image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    # Gamma Correction
    gamma = 1.2
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_image = cv2.LUT(denoised_image, look_up_table)

    return gamma_corrected_image

def enlarge_image(image, scale_factor=2):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the new dimensions
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))

    # Resize the image
    enlarged_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    
    return enlarged_image

# Constants
SAVE_FOLDER = "./output"
IMG_PATH = "./sample/CM-HU1157_10.png"
FONT_PATH = "./fonts/german.ttf"


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def table_detection(image, save_folder):
    table_engine = PPStructure(show_log=True, image_orientation=True, lang="en")
    result = table_engine(image)
    save_structure_res(result, save_folder, "table_result")

    for line in result:
        line.pop("img")
        print(line)

    return result


def draw_table_results(image, result, font_path, save_folder):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    im_show = draw_structure_result(image, result, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(f"{save_folder}/table_detection.jpg")


def text_detection(image, font_path, save_folder):
    ocr = PaddleOCR(use_angle_cls=True, lang="german")
    result = ocr.ocr(image, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # draw ocr result
    result = result[0]
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(f"{save_folder}/ocr_result.jpg")


def main():
    ensure_directory_exists(SAVE_FOLDER)

    # Improve image quality
    improved_image = improve_image_quality(IMG_PATH)
    
    # Enlarge the improved image
    enlarged_image = enlarge_image(improved_image, scale_factor=2)

    # Save the improved and enlarged image for verification
    cv2.imwrite(os.path.join(SAVE_FOLDER, 'improved_enlarged_image.jpg'), enlarged_image)

    # Table Detection
    table_result = table_detection(enlarged_image, SAVE_FOLDER)
    draw_table_results(enlarged_image, table_result, FONT_PATH, SAVE_FOLDER)

    # Text Detection
    text_detection(enlarged_image, FONT_PATH, SAVE_FOLDER)


if __name__ == "__main__":
    main()
