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

table_engine = PPStructure(show_log=True, image_orientation=True, lang="en")
ocr = PaddleOCR(use_angle_cls=True, lang="german")

save_folder = "./output"
img_path = "./sample/CM-HU1157_10.png"
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split(".")[0])

for line in result:
    line.pop("img")
    print(line)


font_path = "./fonts/german.ttf"  # PaddleOCR下提供字体包
image = Image.open(img_path).convert("RGB")
im_show = draw_structure_result(image, result, font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save(f"{save_folder}/table_detection.jpg")


result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw ocr result
result = result[0]
# image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path="./fonts/german.ttf")
im_show = Image.fromarray(im_show)
im_show.save(f"{save_folder}/ocr_result.jpg")
