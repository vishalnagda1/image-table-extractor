import os
import cv2
from paddleocr import PaddleOCR, PPStructure, draw_ocr, draw_structure_result, save_structure_res
from PIL import Image
from image_processor import ImageProcessor
from image_enhancer import ImageEnhancer

class OCRProcessor:
    def __init__(self, save_folder, img_path, font_path):
        self.save_folder = save_folder
        self.img_path = img_path
        self.font_path = font_path

        self.det_model_dir = "models/default/PP-OCRv4/det_en/en_PP-OCRv3_det_infer"
        self.rec_model_dir = "models/default/PP-OCRv4/rec_en/en_PP-OCRv4_rec_infer"
        self.table_model_dir = "models/default/PP-StructureV2/table_en/en_ppstructure_mobile_v2.0_SLANet_infer"
        self.layout_model_dir = "models/default/PP-StructureV2/layout_en/picodet_lcnet_x1_0_fgd_layout_infer"

        self.ensure_directory_exists(self.save_folder)

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def table_detection(self):
        table_engine = PPStructure(
            show_log=True,
            image_orientation=True,
            lang="en",
            table_model_dir=self.table_model_dir,
            layout_model_dir=self.layout_model_dir,
        )

        img = cv2.imread(self.img_path)
        if img is None:
            raise ValueError(f"Image at path {self.img_path} could not be loaded.")

        print("IMAGE SHAPE---------------------->", img.shape[1::-1])

        result = table_engine(img)
        save_structure_res(
            result, self.save_folder, f"{os.path.basename(self.img_path).split('.')[0]}_structure"
        )

        for line in result:
            line.pop("img")
            print(line)

        return result

    def draw_table_results(self, result):
        image = Image.open(self.img_path).convert("RGB")
        im_show = draw_structure_result(image, result, font_path=self.font_path)
        im_show = Image.fromarray(im_show)
        output_image_name = f"{os.path.basename(self.img_path).split('.')[0]}_table_detection.jpg"
        im_show.save(f"{self.save_folder}/{output_image_name}")

    def text_detection(self):
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det_model_dir=self.det_model_dir,
            rec_model_dir=self.rec_model_dir,
        )
        result = ocr.ocr(self.img_path, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)

        # draw ocr result
        result = result[0]
        image = Image.open(self.img_path).convert("RGB")
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=self.font_path)
        im_show = Image.fromarray(im_show)
        output_image_name = f"{os.path.basename(self.img_path).split('.')[0]}_ocr_result.jpg"
        im_show.save(f"{self.save_folder}/{output_image_name}")

    def process_image(self):
        # Table Detection
        table_result = self.table_detection()
        self.draw_table_results(table_result)

        # Text Detection
        self.text_detection()


if __name__ == "__main__":

    input_image_path = "sample/sample.png"
    
    image_enhancer = ImageEnhancer(input_image_path)
    enhanced_image_path = image_enhancer.enhance(outscale=2).save("enhanced.png")
    
    img_processor = ImageProcessor(enhanced_image_path)
    intermediate_img_path = img_processor.add_padding(10).make_square().save("processed.png")

    processor = OCRProcessor(
        save_folder="./output",
        img_path=intermediate_img_path,
        font_path="./fonts/german.ttf"
    )
    processor.process_image()
