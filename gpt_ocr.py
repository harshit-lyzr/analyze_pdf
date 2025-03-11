import os
from pdf2image import convert_from_path
import openai
import base64
from typing import List


class GPT4VisionClient:
    def __init__(self, api_key):
        """
        Initialize the GPT-4 Vision client with the provided API key.
        """
        self.client = openai.OpenAI(api_key=api_key)

    def encode_image_to_base64(self, image_path):
        """
        Encode a local image to a base64 string.

        :param image_path: Path to the local image file.
        :return: Base64-encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path, prompt):
        """
        Analyze the image using the GPT-4 Vision model.

        :param image_path: Path to the local image file.
        :param prompt: Text prompt to guide the analysis.
        :return: Response from the GPT-4 Vision model.
        """
        # Encode the image to base64
        image_base64 = self.encode_image_to_base64(image_path)

        # Create the message payload
        message = {
            'role': 'user',
            'content': [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            ]
        }

        # Send the request to the Chat Completions API
        response = self.client.chat.completions.create(
            model='gpt-4o',
            messages=[message],
            max_tokens=500
        )

        # Extract and return the assistant's response
        return response.choices[0].message.content


def convert_pdf_to_images(pdf_file_path: str) -> List[str]:
    """
    Convert PDF to images and return the list of image paths.

    :param pdf_file_path: Path to the PDF file.
    :return: List of image file paths.
    """
    # Create a folder to save images
    output_folder = "pdf_images"

    # Clean up old images from previous uploads
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_folder, exist_ok=True)

    # Convert PDF to images
    images = convert_from_path(pdf_file_path, dpi=300)

    # Save each page as an image
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    return image_paths


