import os
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2

class ImageEnhancer:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path or 'RealESRGAN_x4plus.pth'
        self.model = self.load_model()

    def load_model(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(scale=4, model_path=self.model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=True)
        return upsampler

    def enhance_image(self, input_image_path, output_image_path):
        # Load the image
        img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

        # Enhance the image
        try:
            enhanced_img, _ = self.model.enhance(img, outscale=4)
        except RuntimeError as error:
            print(f'Error in enhancing the image: {error}')
            return

        # Save the enhanced image
        cv2.imwrite(output_image_path, enhanced_img)

def main(input_image_path, output_image_path, model_path=None):
    # Initialize the enhancer
    enhancer = ImageEnhancer(model_path=model_path)

    # Enhance the image
    enhancer.enhance_image(input_image_path, output_image_path)
    print(f"Enhanced image saved to {output_image_path}")

if __name__ == "__main__":
    input_image_path = "sample/sample.png"
    # input_image_path = "output/transform1.png"
    output_image_path = "output/enhanced.png"
    model_path = "RealESRGAN_x4plus.pth"
    # model_path = "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth"

    main(input_image_path, output_image_path, model_path)
