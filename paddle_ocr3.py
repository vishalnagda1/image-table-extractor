import os
import cv2
from paddleocr import (
    PaddleOCR,
    PPStructure,
    draw_ocr,
    draw_structure_result,
    save_structure_res,
)
from PIL import Image

# Constants
SAVE_FOLDER = "./output"
IMG_PATH = "./sample/sample-460x460.png"
FONT_PATH = "./fonts/german.ttf"

DET_MODEL_DIR = "./models/ppstructure/inference/text_detection/ch_PP-OCRv3_det_infer"
REC_MODEL_DIR = "./models/ppstructure/inference/text_recognition/ch_PP-OCRv3_rec_infer"
TABLE_MODEL_DIR = "./models/ppstructure/inference/table_recognition/en_ppocr_mobile_v2.0_table_structure_infer"
# LAYOUT_MODEL_DIR = "./models/ppstructure/inference/layout_analysis/picodet_lcnet_x1_0_fgd_layout_cdla_infer"


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def table_detection(image_path, save_folder):
    table_engine = PPStructure(
        show_log=True,
        image_orientation=True,
        lang="en",
        # det_model_dir=DET_MODEL_DIR,
        # rec_model_dir=REC_MODEL_DIR,
        table_model_dir=TABLE_MODEL_DIR,
        # layout_model_dir=LAYOUT_MODEL_DIR,
    )

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    print(f"Image shape: {img.shape}")
    
    result = table_engine(img)
    save_structure_res(result, save_folder, f"{os.path.basename(image_path).split('.')[0]}_3")

    for line in result:
        line.pop("img")
        print(line)

    return result


def draw_table_results(image_path, result, font_path, save_folder):
    image = Image.open(image_path).convert("RGB")
    im_show = draw_structure_result(image, result, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(f"{save_folder}/table_detection3.jpg")


def text_detection(image_path, font_path, save_folder):
    ocr = PaddleOCR(use_angle_cls=True, lang="german", det_model_dir=DET_MODEL_DIR, rec_model_dir=REC_MODEL_DIR)
    result = ocr.ocr(image_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # draw ocr result
    result = result[0]
    image = Image.open(image_path).convert("RGB")
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(f"{save_folder}/ocr_result3.jpg")


def main():
    ensure_directory_exists(SAVE_FOLDER)

    # Table Detection
    table_result = table_detection(IMG_PATH, SAVE_FOLDER)
    draw_table_results(IMG_PATH, table_result, FONT_PATH, SAVE_FOLDER)

    # Text Detection
    text_detection(IMG_PATH, FONT_PATH, SAVE_FOLDER)


if __name__ == "__main__":
    main()
