# import cv2
# import pytesseract
# import os

# # Set Tesseract CMD path and TESSDATA_PREFIX environment variable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# def draw_bounding_boxes_and_display_crops(image_path, zoom_out_percentage=65):
#     """Draw bounding boxes around each line of text in the image, convert to grayscale, and display the cropped images."""
#     image = cv2.imread(image_path)
    
#     # Calculate the new size to zoom out
#     width = int(image.shape[1] * zoom_out_percentage / 100)
#     height = int(image.shape[0] * zoom_out_percentage / 100)
#     zoom_out_size = (width, height)
    
#     # Resize the image
#     resized_image = cv2.resize(image, zoom_out_size, interpolation=cv2.INTER_AREA)
    
#     # Convert the resized image to grayscale
#     grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
#     # OCR to detect text and get bounding boxes
#     boxes = pytesseract.image_to_data(grayscale_image, lang='eng', output_type=pytesseract.Output.DICT)
    
#     # Iterate over each detected text area and display
#     n_boxes = len(boxes['level'])
#     for i in range(n_boxes):
#         if boxes['level'][i] == 4:  # Level 4 corresponds to line level
#             x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
#             cropped_image = grayscale_image[y:y+h, x:x+w]
#             # Display the cropped image
#             cv2.imshow(f'Cropped {i}', cropped_image)
#             cv2.waitKey(0)  # Wait for a key press to move to the next image

#     cv2.destroyAllWindows()  # Close all the windows when done

# # Replace 'path_to_your_image.jpg' with the path to your actual image file
# image_path = '21.jpg'
# draw_bounding_boxes_and_display_crops(image_path)
from flask import Flask, request, jsonify
import cv2
import pytesseract
import base64
import os
import numpy as np

app = Flask(__name__)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'
def crop_text_regions(image):
    """Detect text regions, crop them, and return base64 encoded values."""

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    boxes = pytesseract.image_to_data(grayscale_image, lang='eng', output_type=pytesseract.Output.DICT)
    

    cropped_images_base64 = []
    

    for i in range(len(boxes['text'])):
        if boxes['level'][i] == 4:  
            x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
            cropped = image[y:y+h, x:x+w]

            _, buffer = cv2.imencode('.jpg', cropped)
            base64_encoded = base64.b64encode(buffer)
            cropped_images_base64.append(base64_encoded.decode())  
    return cropped_images_base64

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    

    cropped_images_base64 = crop_text_regions(image)
    return jsonify({'cropped_images_base64': cropped_images_base64})

if __name__ == '__main__':
    app.run(debug=True)

