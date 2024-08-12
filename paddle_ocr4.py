import os
import cv2
from paddleocr import PaddleOCR, PPStructure, draw_ocr, draw_structure_result, save_structure_res
from PIL import Image

class OCRProcessor:
    def __init__(self, save_folder, img_path, font_path):
        self.save_folder = save_folder
        self.img_path = img_path
        self.font_path = font_path

        self.det_model_dir = "models/default/PP-OCRv4/det_en/en_PP-OCRv3_det_infer"
        self.rec_model_dir = "models/default/PP-OCRv4/rec_en/en_PP-OCRv4_rec_infer"
        self.table_model_dir = "models/default/PP-StructureV2/table_en/en_ppstructure_mobile_v2.0_SLANet_infer"
        self.layout_model_dir = "models/default/PP-StructureV2/layout_en/picodet_lcnet_x1_0_fgd_layout_infer"

        self.result_table = None
        self.result_text = None

        self.ensure_directory_exists(self.save_folder)

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def detect_table(self):
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

        self.result_table = table_engine(img)
        save_structure_res(
            self.result_table, self.save_folder, f"{os.path.basename(self.img_path).split('.')[0]}_structure"
        )

        for line in self.result_table:
            line.pop("img")
            print(line)

        return self

    def draw_table_result(self):
        if self.result_table is None:
            raise ValueError("No table result found. Please run detect_table() first.")

        image = Image.open(self.img_path).convert("RGB")
        im_show = draw_structure_result(image, self.result_table, font_path=self.font_path)
        im_show = Image.fromarray(im_show)
        output_image_name = f"{os.path.basename(self.img_path).split('.')[0]}_table_detection.jpg"
        im_show.save(f"{self.save_folder}/{output_image_name}")

        return self

    def detect_text(self):
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det_model_dir=self.det_model_dir,
            rec_model_dir=self.rec_model_dir,
        )
        self.result_text = ocr.ocr(self.img_path, cls=True)
        for idx in range(len(self.result_text)):
            res = self.result_text[idx]
            for line in res:
                print(line)

        return self

    def draw_text_result(self):
        if self.result_text is None:
            raise ValueError("No text result found. Please run detect_text() first.")

        result = self.result_text[0]
        image = Image.open(self.img_path).convert("RGB")
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=self.font_path)
        im_show = Image.fromarray(im_show)
        output_image_name = f"{os.path.basename(self.img_path).split('.')[0]}_ocr_result.jpg"
        im_show.save(f"{self.save_folder}/{output_image_name}")

        return self

    def table_to_excel(self):
        # Example implementation to export the table to an Excel file.
        if self.result_table is None:
            raise ValueError("No table result found. Please run detect_table() first.")

        # Implementation code to convert the table result to an Excel file.

        print("Table data has been exported to an Excel file.")
        return self

    def text_to_string(self, delimiter="\n"):
        """
        Converts the detected text into a single string, joined by the specified delimiter.
        
        :param delimiter: The delimiter to use when joining the detected text. Defaults to newline ('\n').
        :return: The extracted text as a single string.
        """
        if self.result_text is None:
            raise ValueError("No text result found. Please run detect_text() first.")

        extracted_text = delimiter.join([line[1][0] for line in self.result_text[0]])
        print(extracted_text)
        return extracted_text


if __name__ == "__main__":
    input_image_path = "sample/sample.png"

    processor = OCRProcessor(save_folder="./output", img_path=input_image_path, font_path="./fonts/german.ttf")
    table = processor.detect_table().draw_table_result()
    text = processor.detect_text().draw_text_result()
    # table.table_to_excel()
    text_str = text.text_to_string()
