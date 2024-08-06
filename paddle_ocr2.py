import os
import cv2
from paddleocr import PPStructure, draw_structure_result, save_structure_res, draw_ocr
from PIL import Image

def load_image(image_path):
    """Load an image from the specified path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image path {image_path} does not exist.")
    return cv2.imread(image_path)

def detect_table_structure(image, lang='en'):
    """Detect table structure in the image using PaddleOCR."""
    table_engine = PPStructure(show_log=True, image_orientation=True, lang=lang)
    return table_engine(image)

def save_detection_results(results, save_folder, image_basename):
    """Save detection results to the specified folder."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_structure_res(results, save_folder, image_basename.split('.')[0])

def draw_and_save_results(image, results, font_path, output_file):
    """Draw and save the OCR results on the image."""
    im_show = draw_structure_result(image, results, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(output_file)

def draw_and_save_ocr(image, results, font_path, output_file):
    """Draw and save the OCR results on the image."""
    result = results[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(output_file)

def main():
    save_folder = './output'
    img_path = './sample/CM-HU1157_10.png'
    font_path = './fonts/german.ttf'
    
    try:
        img = load_image(img_path)
        result = detect_table_structure(img)
        save_detection_results(result, save_folder, os.path.basename(img_path))
        
        for line in result:
            line.pop('img')
            print(line)
        
        image = Image.open(img_path).convert('RGB')
        draw_and_save_results(image, result, font_path, 'table_detection.jpg')
        draw_and_save_ocr(image, result, font_path, 'ocr.jpg')
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
