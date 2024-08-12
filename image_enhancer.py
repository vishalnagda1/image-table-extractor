import os

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class ImageEnhancer:
    def __init__(
        self,
        input_image_path,
        output_directory_or_path=None,
        model_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.input_image_path = input_image_path
        self.device = device
        self.model_path = model_path or "RealESRGAN_x4plus.pth"
        self.model = self.load_model()

        # Determine the output directory and optional file name
        if output_directory_or_path:
            if os.path.isdir(output_directory_or_path):
                self.output_directory = output_directory_or_path
                self.output_image_name = None
            else:
                self.output_directory, self.output_image_name = os.path.split(
                    output_directory_or_path
                )
        else:
            self.output_directory = "./output"
            self.output_image_name = None

        # Ensure the output directory exists
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.img = cv2.imread(self.input_image_path, cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError(
                f"Image at path {self.input_image_path} could not be loaded."
            )

    def load_model(self):
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=self.device,
        )
        return upsampler

    def enhance(self, outscale=4):
        """
        Enhances the image using the loaded model.

        :param outscale: The scale factor for the enhancement.
        :return: Self, to allow chaining.
        """
        try:
            self.img, _ = self.model.enhance(self.img, outscale=outscale)
        except RuntimeError as error:
            print(f"Error in enhancing the image: {error}")
        return self

    def save(self, output_image_name=None, format=None):
        """
        Saves the enhanced image with the specified output image name or falls back to the input image name.

        :param output_image_name: Optional. Name to save the image as.
        :param format: Optional. Format to save the image (e.g., 'png', 'jpg').
        :return: The path where the image was saved.
        """
        # Determine the file name and format to use
        if output_image_name:
            file_name = output_image_name
        elif self.output_image_name:
            file_name = self.output_image_name
        else:
            file_name = os.path.basename(self.input_image_path)

        # Add format extension if specified
        if format:
            file_name = f"{os.path.splitext(file_name)[0]}.{format}"

        output_image_path = os.path.join(self.output_directory, file_name)

        # Save the image
        cv2.imwrite(output_image_path, self.img)
        print(f"Enhanced image saved to {output_image_path}")

        return output_image_path


def main(input_image_path, output_directory_or_path=None, model_path=None):
    # Initialize the enhancer
    enhancer = ImageEnhancer(
        input_image_path, output_directory_or_path, model_path=model_path
    )

    # Enhance the image and save
    enhancer.enhance().save()


if __name__ == "__main__":
    input_image_path = "sample/sample.png"
    output_directory_or_path = "output/enhanced.png"
    model_path = "RealESRGAN_x4plus.pth"

    main(input_image_path, output_directory_or_path, model_path)
