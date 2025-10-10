import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
from roboflow import Roboflow
from dotenv import load_dotenv
from flask import send_from_directory
from werkzeug.utils import secure_filename
import logging

load_dotenv()
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize Roboflow client

rf=Roboflow(api_key="9VDk4sfMMuanS6OE8aRU")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file):
    """Check if file size is within limits"""
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    return size <= MAX_FILE_SIZE
def classify_with_roboflow(image_path):
    """
    Use Roboflow SDK to classify the image
    """
    try:
        project = rf.workspace().project("ai-image-detector-dolex")
        model = project.version(8).model
        result = model.predict(image_path).json()
        
        logger.info(f"Roboflow API response: {result}")
        
        if 'predictions' in result and len(result['predictions']) > 0:
            prediction = result['predictions'][0]['class']
            confidence = result['predictions'][0]['confidence']
            
            # Normalize prediction labels to match expected format
            if prediction.lower() in ['synthetic', 'ai-generated', 'ai_generated', 'ai generated']:
                prediction = "AI-Generated Image"
            elif prediction.lower() in ['real', 'natural', 'authentic']:
                prediction = "Real Image"
            else:
                # If we get an unexpected label, try to determine from confidence
                if confidence > 0.5:
                    prediction = "AI-Generated Image" if "synthetic" in prediction.lower() or "ai" in prediction.lower() else "Real Image"
                else:
                    prediction = "Real Image"  # Default to real if uncertain
                    
            logger.info(f"Normalized prediction: {prediction} (confidence: {confidence})")
            return prediction, confidence, None
        else:
            logger.warning("No predictions found in Roboflow response")
            return "Real Image", 0.5, None  # Default to real with medium confidence
            
    except Exception as e:
        logger.error(f"Error with Roboflow inference: {str(e)}")
        return None, None, f"Classification error: {str(e)}"
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if file was uploaded
        if 'image' not in request.files:
            return render_template("index.html", error="No file selected. Please choose an image file.")
        
        file = request.files['image']
        
        # Check if file is empty
        if file.filename == '':
            return render_template("index.html", error="No file selected. Please choose an image file.")
        
        # Validate file extension
        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type. Please upload a JPG, PNG, or WEBP image.")
        
        # Validate file size
        if not validate_file_size(file):
            return render_template("index.html", error=f"File too large. Please upload an image smaller than {MAX_FILE_SIZE // (1024*1024)}MB.")
        
        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            # Save file
            file.save(filepath)
            logger.info(f"File saved: {unique_filename}")
            
            # Validate image with PIL
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Verify it's a valid image
            except Exception as e:
                os.remove(filepath)
                logger.error(f"Invalid image file: {str(e)}")
                return render_template("index.html", error="Invalid image file. Please upload a valid image.")
            
            # Classify image
            prediction, confidence, error = classify_with_roboflow(filepath)
            
            if error:
                os.remove(filepath)
                logger.error(f"Classification error: {error}")
                return render_template("index.html", error=f"Classification failed: {error}")
            
            logger.info(f"Classification successful: {prediction} (confidence: {confidence})")
            
            return render_template(
                "result.html",
                prediction=prediction,
                confidence=confidence,
                filename=unique_filename
            )
            
        except Exception as e:
            # Clean up file if it exists
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            
            logger.error(f"Unexpected error: {str(e)}")
            return render_template("index.html", error="An unexpected error occurred. Please try again.")
    
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)