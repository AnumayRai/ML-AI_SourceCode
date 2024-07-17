def preprocess_image(image):
    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded

def ocr_extract_text(image_path):
    # Load the image from file
    image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Perform OCR on the preprocessed image
    text = pytesseract.image_to_string(preprocessed_image)

    return text
