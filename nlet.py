# Ref: https://www.youtube.com/watch?v=HZh31OGiQRQ

# image_path = "./sample/MM-PP1013_11.png"
# image_path = "./sample/CM-HU1157_10.png"

from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='german') # need to run only once to download and load model into memory
img_path = "./sample/CM-HU1157_10.png"
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/german.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
