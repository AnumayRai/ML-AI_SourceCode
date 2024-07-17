import cv2
import pytesseract
from googletrans import Translator

# Update the path to your Tesseract executable if necessary
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

def ocr_extract_text(image_path):
    # Load the image from file
    image = cv2.imread(image_path)

    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the image
    text = pytesseract.image_to_string(gray)

    return text

def translate_text(text, dest_lang='en'):
    # Initialize the translator
    translator = Translator()

    # Translate the text
    translation = translator.translate(text, dest=dest_lang)

    return translation.text

def main():
    # Path to the image containing the text
    image_path = 'path/to/your/image.jpg'

    # Extract text from the image
    extracted_text = ocr_extract_text(image_path)
    print("Extracted Text:")
    print(extracted_text)

    # Translate the extracted text
    translated_text = translate_text(extracted_text)
    print("\nTranslated Text:")
    print(translated_text)

if __name__ == '__main__':
    main()
