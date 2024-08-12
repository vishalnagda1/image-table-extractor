import os
import cv2

class ImageProcessorCV2:
    """
    Processes an image with padding, resizing, grayscale conversion, and more using OpenCV.
    """
    def __init__(self, input_image_path, output_directory_or_path=None):
        self.input_image_path = input_image_path
        
        # Determine the output directory and optional file name
        if output_directory_or_path:
            if os.path.isdir(output_directory_or_path):
                self.output_directory = output_directory_or_path
                self.output_image_name = None
            else:
                self.output_directory, self.output_image_name = os.path.split(output_directory_or_path)
        else:
            self.output_directory = './output'
            self.output_image_name = None
        
        # Ensure the output directory exists
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        self.img = cv2.imread(self.input_image_path)
        if self.img is None:
            raise ValueError(f"Image at path {self.input_image_path} could not be loaded.")

    def add_padding(self, padding=0, color=(0, 0, 0)):
        """
        Adds custom padding to the image.
        
        :param padding: A single integer for uniform padding or a tuple/list of four integers (top, bottom, left, right).
        :param color: The color for the padding (BGR).
        :return: Self, to allow chaining.
        """
        top, bottom, left, right = 0, 0, 0, 0

        if isinstance(padding, (int, float)):
            top = bottom = left = right = padding
        elif isinstance(padding, (tuple, list)) and len(padding) == 4:
            top, bottom, left, right = padding
        else:
            raise ValueError("Padding must be an integer or a list/tuple of four integers.")

        self.img = cv2.copyMakeBorder(
            self.img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=color
        )

        return self

    def make_square(self, color=(0, 0, 0)):
        """
        Automatically adds padding to make the image square.
        
        :param color: The color for the padding (BGR).
        :return: Self, to allow chaining.
        """
        original_height, original_width = self.img.shape[:2]
        new_size = max(original_width, original_height)

        top = (new_size - original_height) // 2
        bottom = new_size - original_height - top
        left = (new_size - original_width) // 2
        right = new_size - original_width - left

        self.img = cv2.copyMakeBorder(
            self.img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=color
        )

        return self

    def resize(self, scale_factor=1):
        """
        Resizes the image by a given scale factor. Can be used for both upscaling and downscaling.
        
        :param scale_factor: The factor by which to resize the image. Greater than 1 for upscaling, less than 1 for downscaling.
        :return: Self, to allow chaining.
        """
        if scale_factor <= 0:
            raise ValueError("Scale factor must be greater than 0.")

        new_width = int(self.img.shape[1] * scale_factor)
        new_height = int(self.img.shape[0] * scale_factor)
        self.img = cv2.resize(self.img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return self

    def resize_to_dimensions(self, width, height):
        """
        Resizes the image to the specified width and height.
        
        :param width: The desired width of the image.
        :param height: The desired height of the image.
        :return: Self, to allow chaining.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be greater than 0.")

        self.img = cv2.resize(self.img, (width, height), interpolation=cv2.INTER_LINEAR)

        return self

    def crop(self, x, y, width, height):
        """
        Crops the image to the specified rectangle.
        
        :param x: The x-coordinate of the top-left corner.
        :param y: The y-coordinate of the top-left corner.
        :param width: The width of the crop area.
        :param height: The height of the crop area.
        :return: Self, to allow chaining.
        """
        self.img = self.img[y:y+height, x:x+width]
        return self

    def rotate(self, angle):
        """
        Rotates the image by the specified angle.
        
        :param angle: The angle (in degrees) to rotate the image.
        :return: Self, to allow chaining.
        """
        height, width = self.img.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.img = cv2.warpAffine(self.img, matrix, (width, height))
        return self

    def blur(self, ksize=(5, 5)):
        """
        Applies a blur effect to the image.
        
        :param ksize: The size of the kernel used for blurring.
        :return: Self, to allow chaining.
        """
        self.img = cv2.GaussianBlur(self.img, ksize, 0)
        return self

    def detect_edges(self, threshold1=100, threshold2=200):
        """
        Detects edges in the image using Canny edge detection.
        
        :param threshold1: First threshold for the hysteresis procedure.
        :param threshold2: Second threshold for the hysteresis procedure.
        :return: Self, to allow chaining.
        """
        self.img = cv2.Canny(self.img, threshold1, threshold2)
        return self

    def adjust_brightness_contrast(self, brightness=0, contrast=0):
        """
        Adjusts the brightness and contrast of the image.
        
        :param brightness: Value to adjust brightness (-255 to 255).
        :param contrast: Value to adjust contrast (-127 to 127).
        :return: Self, to allow chaining.
        """
        self.img = cv2.convertScaleAbs(self.img, alpha=1 + contrast / 127.0, beta=brightness)
        return self

    def invert_colors(self):
        """
        Inverts the colors of the image.
        
        :return: Self, to allow chaining.
        """
        self.img = cv2.bitwise_not(self.img)
        return self

    def to_grayscale(self):
        """
        Converts the image to grayscale.
        
        :return: Self, to allow chaining.
        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        return self

    def get_dimensions(self):
        """
        Returns the current dimensions of the image.
        
        :return: A tuple (width, height) representing the dimensions of the image.
        """
        height, width = self.img.shape[:2]
        return width, height

    def denoise(self, h=10, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21):
        """
        Denoises the image using Non-Local Means Denoising algorithm.
        
        :param h: Parameter regulating filter strength. Big h value perfectly removes noise but also removes image details.
        :param hForColorComponents: Same as h but for color images only.
        :param templateWindowSize: Size in pixels of the window that is used to compute weighted average for a given pixel.
        :param searchWindowSize: Size in pixels of the window used to compute weighted average for a given pixel.
        :return: Self, to allow chaining.
        """
        self.img = cv2.fastNlMeansDenoisingColored(
            self.img, None, h, hForColorComponents, templateWindowSize, searchWindowSize
        )
        return self

    def save(self, output_image_name=None, format=None):
        """
        Saves the processed image with the specified output image name or falls back to the input image name.
        
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
        print(f"Image saved to {output_image_path}")
        
        return output_image_path

if __name__ == "__main__":
    # Example usage:
    input_image_path = "sample/sample.png"
    
    # Case 1: No output directory or image name specified, defaults to './output/sample.png'
    resizer = ImageProcessorCV2(input_image_path)
    output_path = resizer.add_padding(10).make_square().resize(1.5).to_grayscale().save()
    print(f"Final image saved at: {output_path}")
    
    # Get image dimensions
    dimensions = resizer.get_dimensions()
    print(f"Image dimensions: {dimensions}")
    
    # Case 2: Specified output directory without image name, uses input image name in the given directory
    resizer = ImageProcessorCV2(input_image_path, output_directory_or_path="output_images/")
    output_path = resizer.add_padding(10).make_square().resize_to_dimensions(800, 600).to_grayscale().save()
    print(f"Final image saved at: {output_path}")
    print(f"Image dimensions: {resizer.get_dimensions()}")
    
    # Case 3: Specified output directory with image name
    resizer = ImageProcessorCV2(input_image_path, output_directory_or_path="output_images/custom_name.png")
    output_path = resizer.add_padding(10).make_square().resize_to_dimensions(1024, 768).save()
    print(f"Final image saved at: {output_path}")
    print(f"Image dimensions: {resizer.get_dimensions()}")
    
    # Case 4: Specified output directory, then provide a different name at save time
    resizer = ImageProcessorCV2(input_image_path, output_directory_or_path="output_images/")
    output_path = resizer.add_padding(10).make_square().resize(0.75).save("final_image.png")
    print(f"Final image saved at: {output_path}")
    print(f"Image dimensions: {resizer.get_dimensions()}")
    
    # Case 5: Specified output directory with image name, then provide a different name at save time
    resizer = ImageProcessorCV2(input_image_path, output_directory_or_path="output_images/custom_name.png")
    output_path = resizer.add_padding(10).make_square().resize_to_dimensions(640, 480).save("final_image.png")
    print(f"Final image saved at: {output_path}")
    print(f"Image dimensions: {resizer.get_dimensions()}")

    resizer = ImageProcessorCV2(input_image_path)
    output_path = (
        resizer.add_padding(10)
        .make_square()
        .resize_to_dimensions(800, 600)
        .rotate(45)
        .blur((7, 7))
        .denoise()
        .invert_colors()
        .adjust_brightness_contrast(brightness=30, contrast=20)
        .detect_edges(50, 150)
        .save("processed_image.png", format="png")
    )
    print(f"Final image saved at: {output_path}")
    
    # Get image dimensions
    dimensions = resizer.get_dimensions()
    print(f"Image dimensions: {dimensions}")
