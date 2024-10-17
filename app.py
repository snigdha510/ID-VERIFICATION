from flask import Flask, request, jsonify
import cv2
import pytesseract
from PIL import Image
import os
import re
import numpy as np
import face_recognition
import torch
import torchvision
from torchvision import transforms
from flask import jsonify
import time

app = Flask(__name__)

# Loading a pre-trained Mask R-CNN model from torchvision
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Preprocessing the image for Mask R-CNN
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    return image_tensor

# Running mask R-CNN for segmentation
def segment_image(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        prediction = model([image_tensor])
    
    # Extract masks and bounding boxes
    masks = prediction[0]['masks'].numpy()
    boxes = prediction[0]['boxes'].numpy()

    return masks, boxes

# Usinging segmentation technique for better OCR
def segment_id_card(image_path):
    masks, boxes = segment_image(image_path)
    
    if len(boxes) > 0:
        largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        return largest_box
    else:
        print("No objects detected.")
        return None

def crop_id_image(image_path):
    image = cv2.imread(image_path)
    box = segment_id_card(image_path)
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image
    return image

def clarity_score(image):
    # Converting to grayscale for calculating Laplacian variance
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    normalized_score = min((laplacian_var / 100) * 100, 100)  # Cap at 100
    return normalized_score 

def assess_clarity(score, is_gov_id=True):
    if is_gov_id:
        if score < 20:
            return {"status": "Poor", "acceptable": False, "message": "Government ID image is too blurry. Please retake in better lighting with a steady hand."}
        elif score < 40:
            return {"status": "Fair", "acceptable": False, "message": "Government ID image clarity is below recommended. Please retake for better OCR accuracy."}
        elif score < 60:
            return {"status": "Good", "acceptable": True, "message": "Acceptable clarity, but consider retaking if OCR fails."}
        elif score < 80:
            return {"status": "Very Good", "acceptable": True, "message": "Good image clarity."}
        else:
            return {"status": "Excellent", "acceptable": True, "message": "Excellent image clarity."}
    else: 
        if score < 20:
            return {"status": "Poor", "acceptable": False, "message": "Selfie is too blurry. Please retake in better lighting with a steady hand."}
        elif score < 30:
            return {"status": "Fair", "acceptable": False, "message": "Selfie clarity is below recommended. Please retake for better face matching."}
        elif score < 50:
            return {"status": "Good", "acceptable": True, "message": "Acceptable clarity for face matching."}
        elif score < 70:
            return {"status": "Very Good", "acceptable": True, "message": "Good selfie clarity."}
        else:
            return {"status": "Excellent", "acceptable": True, "message": "Excellent selfie clarity."}


def clean_ocr_text(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9<\s]', '', text)  
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) 
    return cleaned_text

def identify_gov_id(gov_id_image):
    extracted_text = pytesseract.image_to_string(gov_id_image)
    extracted_text = clean_ocr_text(extracted_text)
    extracted_text_lower = extracted_text.lower()

    pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]'
    aadhaar_pattern = r'\d{4} \d{4} \d{4}'
    driving_license_pattern = r'[A-Z]{2}\d{2} \d{4} \d{7}'
    passport_pattern = r'\b[A-Z]{1}[0-9]{7}\b'

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
        return "Unknown ID Type"

def verify_face(selfie_path, gov_id_path):
    selfie_image = face_recognition.load_image_file(selfie_path)
    gov_id_image = face_recognition.load_image_file(gov_id_path)

    selfie_face_encodings = face_recognition.face_encodings(selfie_image)
    gov_id_face_encodings = face_recognition.face_encodings(gov_id_image)

    if len(selfie_face_encodings) == 0 or len(gov_id_face_encodings) == 0:
        return False

    selfie_face_encoding = selfie_face_encodings[0]
    gov_id_face_encoding = gov_id_face_encodings[0]

    matches = face_recognition.compare_faces([gov_id_face_encoding], selfie_face_encoding)
    return matches[0]

def capture_selfie():
 
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return None

    # Set optimal camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)    
    cap.set(cv2.CAP_PROP_CONTRAST, 128)       
    cap.set(cv2.CAP_PROP_SATURATION, 128)  
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) 
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)        
    
    # Allow camera to adjust settings
    time.sleep(2)
    
    # multiple frames capturing is allowed for the camera to adjust
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
    
    # capturing the final frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    # image enhancement techniques for better clarity of picture
    def enhance_image(image):

        image_float = image.astype(np.float32) / 255.0
        

        def white_balance(img):
            result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        

        balanced = white_balance(image_float)
        

        gamma = 1.1  
        enhanced = np.power(balanced, gamma)
        

        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2 
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
       
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        return enhanced
    

    enhanced_frame = enhance_image(frame)
    

    denoised = cv2.fastNlMeansDenoisingColored(enhanced_frame, 
                                              None, 
                                              h=5,        
                                              hColor=5,   
                                              templateWindowSize=7, 
                                              searchWindowSize=21)
    
    # Sharpen the image while preserving colors
    def sharpen_preserve_color(image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Sharpen only the L channel
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        l_sharp = cv2.filter2D(l, -1, kernel)
        

        sharp_lab = cv2.merge([l_sharp, a, b])
        return cv2.cvtColor(sharp_lab, cv2.COLOR_LAB2BGR)
    
    final_image = sharpen_preserve_color(denoised)
    
    # saving the final image
    selfie_path = "selfie.jpg"
    cv2.imwrite(selfie_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    

    cap.release()
    
    return selfie_path

# final api endpoint for the whole process we need to upload the gov id as a key value pair where key will be 'gov_id' and value would be the picture of the gov id
@app.route('/process-id', methods=['POST'])
def process_id():
    try:
        gov_id_image = request.files.get('gov_id')

        if gov_id_image is None:
            return jsonify({"error": "Government ID image is required."}), 400

     
        gov_id_path = "gov_id.jpg"
        gov_id_image.save(gov_id_path)

        # Processing the uploaded government ID
        cropped_image = crop_id_image(gov_id_path)
        gov_id_clarity = clarity_score(cropped_image)
        gov_id_assessment = assess_clarity(gov_id_clarity, is_gov_id=True)

        if not gov_id_assessment["acceptable"]:
            return jsonify({
                "error": gov_id_assessment["message"],
                "clarity_score": float(gov_id_clarity),
                "clarity_status": gov_id_assessment["status"]
            }), 400

        id_type = identify_gov_id(cropped_image)

        # Capturing candidate selfie
        selfie_path = capture_selfie()
        if selfie_path is None:
            return jsonify({"error": "Failed to capture selfie"}), 400

        # Processing the captured selfie
        selfie_image = cv2.imread(selfie_path)
        selfie_clarity = clarity_score(selfie_image)
        selfie_assessment = assess_clarity(selfie_clarity, is_gov_id=False)

        if not selfie_assessment["acceptable"]:
            return jsonify({
                "error": selfie_assessment["message"],
                "clarity_score": float(selfie_clarity),
                "clarity_status": selfie_assessment["status"]
            }), 400

        # face matching process
        is_match = verify_face(selfie_path, gov_id_path)

        result = {
            "gov_id_clarity": {
                "score": float(gov_id_clarity),
                "status": gov_id_assessment["status"],
                "message": gov_id_assessment["message"]
            },
            "id_type": id_type,
            "selfie_clarity": {
                "score": float(selfie_clarity),
                "status": selfie_assessment["status"],
                "message": selfie_assessment["message"]
            },
            "face_match": bool(is_match)
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# main function 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000)
