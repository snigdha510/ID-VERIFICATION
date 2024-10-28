# Import the necessary modules 
import cv2 # for image processing
import numpy as np
import pytesseract # python library for ocr
from flask import Flask, request, jsonify # web-application instance
from werkzeug.utils import secure_filename # creating a folder to store the uploaded files temporarily
import os
import re # for classification of gov-id type
from PIL import Image # for image processing
from pdf2image import convert_from_path # for pdf type uploaded gov-id
import face_recognition # verification purpose

# Create a Flask web application instance
app = Flask(__name__)

# Allowed the following file extensions for uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}

# Creating the upload folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Setting a clarity score threshold for uploaded gov-id images
CLARITY_THRESHOLD = 100

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess images for better OCR performance
def preprocess_image(image):

    if isinstance(image, str):
        img = cv2.imread(image)

    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError("Image must be either a file path or a numpy array")
    
    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Then applu Gaussian blur to reduce noise in the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Then used adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to extract text from an image using OCR which will further be used for classification of gov-id
def extract_text(image_path):
    preprocessed_img = preprocess_image(image_path)  
    h, w = preprocessed_img.shape  # for getting dimensions of the image

    # Only focus on the relevant part of the image
    x_start, y_start, x_end, y_end = int(w * 0.05), int(h * 0.1), int(w * 0.95), int(h * 0.8)
    cropped_img = preprocessed_img[y_start:y_end, x_start:x_end]  # Crop the image
    custom_config = r'--oem 3 --psm 6'  # Configure Tesseract (python library) OCR
    extracted_text = pytesseract.image_to_string(cropped_img, config=custom_config)  # Perform OCR
    cleaned_text = ' '.join(extracted_text.split())  # Clean the extracted text
    return cleaned_text

# Function to calculate the clarity score of an image
def calculate_clarity_score(image):
    # Ensure the image is a NumPy array 
    if isinstance(image, np.ndarray):
        # Convert the image to grayscale for clarity score calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Image must be a numpy array in BGR format")
    
    # Using laplacian variance to calculate clarity
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

# Function to clean up the OCR text (remove leading/trailing whitespace)
def clean_ocr_text(text):
    return text.strip()

# Function to identify the type of government ID from the image
def identify_gov_id(gov_id_image):
    # Preprocess the image for OCR
    preprocessed_img = preprocess_image(gov_id_image)

    # Defined a bounding box for cropping the ID card
    h, w = preprocessed_img.shape
    x_start, y_start, x_end, y_end = int(w * 0.05), int(h * 0.1), int(w * 0.95), int(h * 0.8)
    cropped_img = preprocessed_img[y_start:y_end, x_start:x_end]  # Crop the image for text extraction

    # Used Tesseract to extract text from the cropped image
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(cropped_img, config=custom_config)
    extracted_text = clean_ocr_text(extracted_text)  # Clean the extracted text

    # print(f"Extracted Text: {extracted_text}")

    extracted_text_lower = extracted_text.lower()  # Convert text to lowercase for pattern matching

    # Defined regular expression patterns for identifying different types of IDs
    pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]'  # PAN Card pattern
    aadhaar_pattern = r'\d{4} \d{4} \d{4}'  # Aadhaar Card pattern
    driving_license_pattern = r'[A-Z]{2}\d{2} \d{4} \d{7}'  # Driving License pattern
    passport_pattern = r'\b[A-Z]{1}[0-9]{7}\b'  # Passport pattern

    # Check the extracted text with the defined patterns to identify the ID type
    if "permanent account number" in extracted_text_lower or re.search(pan_pattern, extracted_text):
        return "PAN Card"
    elif "aadhaar" in extracted_text_lower or re.search(aadhaar_pattern, extracted_text):
        return "Aadhaar Card"
    elif "driving licence" in extracted_text_lower or re.search(driving_license_pattern, extracted_text):
        return "Driving Licence"
    elif re.search(passport_pattern, extracted_text):
        return "Passport"
    elif "p<ind" in extracted_text_lower:  
        return "Passport (MRZ Detected)"
    else:
        return "Unknown ID Type"  # Return unknown if no patterns match

# Endpoint to upload government ID images
@app.route('/upload_id', methods=['POST'])
def upload_id():
    # Check if the file part is present in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']  # Get the uploaded file
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Check if the file is a PDF
        if file.filename.endswith('.pdf'):
            # Save the PDF temporarily
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(pdf_path)
            # Convert PDF to image (only the first page)
            images = convert_from_path(pdf_path)
            gov_id_image = images[0]
            os.remove(pdf_path)  
            
            # Convert the PIL image to a NumPy array
            gov_id_image = np.array(gov_id_image)
            # Convert RGB to BGR format for OpenCV
            gov_id_image = cv2.cvtColor(gov_id_image, cv2.COLOR_RGB2BGR)
        else:
            # Handle image files directly if the file is of jpg, jpeg, png type
            gov_id_image = Image.open(file).convert('RGB') 
            gov_id_image = np.array(gov_id_image)
            gov_id_image = cv2.cvtColor(gov_id_image, cv2.COLOR_RGB2BGR)  

        # Calculate clarity score for the uploaded government ID image
        clarity_score = calculate_clarity_score(gov_id_image)

        # Identify the type of government ID from the image
        id_type = identify_gov_id(gov_id_image)

        # Check if the clarity score meets the threshold
        if clarity_score < CLARITY_THRESHOLD:
            return jsonify({
                "id_type": id_type,
                "clarity_score": clarity_score,
                "message": "Clarity score is below the threshold. Please upload a clearer ID."
            }), 400
        elif id_type == 'Unknown ID Type':
            return jsonify({
                "id_type": id_type,
                "message": "Failed to recognize the uploaded ID. Please try with another ID card and a clear JPG or PNG file."
            }), 400
        else:
            return jsonify({
                "id_type": id_type,
                "clarity_score": clarity_score,
                "message": "ID uploaded successfully."
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400  

# Endpoint to upload selfie images
@app.route('/upload_selfie', methods=['POST'])
def upload_selfie():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Open the image using PIL and convert to RGB
        selfie_image = Image.open(file_path).convert('RGB')

        # Convert the image to a NumPy array for OpenCV
        selfie_cv_image = np.array(selfie_image)
        selfie_cv_image = cv2.cvtColor(selfie_cv_image, cv2.COLOR_RGB2BGR)  # Convert to BGR format

        # Calculate clarity score for the selfie
        clarity_score = calculate_clarity_score(selfie_cv_image)

        # Check clarity score against threshold
        if clarity_score < CLARITY_THRESHOLD:
            return jsonify({
                "clarity_score": clarity_score,
                "message": "Clarity score is below the threshold. Please upload a clearer selfie."
            }), 400
        else:
            return jsonify({
                "clarity_score": clarity_score,
                "message": "Selfie uploaded successfully."
            }), 200

    return jsonify({'error': 'Invalid file type'}), 400

# Endpoint for final verification
@app.route('/verify_faces', methods=['POST'])
def verify_faces():
    if 'gov_id' not in request.files or 'selfie' not in request.files:
        return jsonify({'error': 'Both files are required for verification.'}), 400

    gov_id_file = request.files['gov_id']
    selfie_file = request.files['selfie']

    # Check if both files are allowed
    if not (allowed_file(gov_id_file.filename) and allowed_file(selfie_file.filename)):
        return jsonify({'error': 'Invalid file type for one or both files. Allowed types: jpg, jpeg, png, pdf.'}), 400

    try:
        # Save the government ID and selfie temporarily
        gov_id_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(gov_id_file.filename))
        selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(selfie_file.filename))
        gov_id_file.save(gov_id_path)
        selfie_file.save(selfie_path)

        # Load government ID image, handling PDF conversion if necessary
        if gov_id_file.filename.endswith('.pdf'):
            # Convert PDF to image
            images = convert_from_path(gov_id_path)
            gov_id_image = np.array(images[0])  # Use the first page
        else:
            gov_id_image = face_recognition.load_image_file(gov_id_path)

        # Load selfie image, handling PDF conversion if necessary
        if selfie_file.filename.endswith('.pdf'):
            # Convert PDF to image
            images = convert_from_path(selfie_path)
            selfie_image = np.array(images[0])  # Use the first page
        else:
            selfie_image = face_recognition.load_image_file(selfie_path)

        # Encoding faces for face-matching and verification purpose
        gov_id_encoding = face_recognition.face_encodings(gov_id_image)
        selfie_encoding = face_recognition.face_encodings(selfie_image)

        if len(gov_id_encoding) == 0 or len(selfie_encoding) == 0:
            return jsonify({'error': 'No face detected in one of the images.'}), 400

        # Comparing faces for final verification
        match = face_recognition.compare_faces([gov_id_encoding[0]], selfie_encoding[0])

        if match[0]:
            return jsonify({'verification_status': 'Verification successful.'}), 200
        else:
            return jsonify({'verification_status': 'Verification unsuccessful.'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
# Run the main program
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
