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
        # Check if file exists and is readable
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return None, None, "Image file not found"
        
        # Check file size
        file_size = os.path.getsize(image_path)
        logger.info(f"Image file size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("Image file is empty")
            return None, None, "Image file is empty"
        
        # Initialize Roboflow client
        logger.info("Initializing Roboflow client...")
        project = rf.workspace().project("ai-image-detector-dolex")
        model = project.version(8).model
        logger.info("Roboflow model loaded successfully")
        
        # Make prediction
        logger.info(f"Making prediction for: {image_path}")
        result = model.predict(image_path)
        logger.info(f"Raw prediction result type: {type(result)}")
        
        # Convert to JSON if needed
        if hasattr(result, 'json'):
            result_json = result.json()
        else:
            result_json = result
            
        logger.info(f"Roboflow API response: {result_json}")
        
        # Handle different response formats
        predictions = []
        if isinstance(result_json, dict):
            if 'predictions' in result_json:
                predictions = result_json['predictions']
            elif 'results' in result_json:
                predictions = result_json['results']
        elif isinstance(result_json, list):
            predictions = result_json
        
        if predictions and len(predictions) > 0:
            # Get the first prediction
            first_prediction = predictions[0]
            logger.info(f"First prediction: {first_prediction}")
            
            # Extract class and confidence
            if isinstance(first_prediction, dict):
                prediction = first_prediction.get('class', first_prediction.get('label', 'unknown'))
                confidence = first_prediction.get('confidence', first_prediction.get('score', 0.0))
            else:
                # Handle case where prediction might be a string
                prediction = str(first_prediction)
                confidence = 0.5
                
            logger.info(f"Extracted - Class: {prediction}, Confidence: {confidence}")
            
            # Normalize prediction labels
            prediction_lower = prediction.lower()
            if any(term in prediction_lower for term in ['synthetic', 'ai-generated', 'ai_generated', 'ai generated', 'fake', 'artificial']):
                normalized_prediction = "AI-Generated Image"
            elif any(term in prediction_lower for term in ['real', 'natural', 'authentic', 'genuine']):
                normalized_prediction = "Real Image"
            else:
                # Default based on confidence
                normalized_prediction = "AI-Generated Image" if confidence > 0.5 else "Real Image"
                
            logger.info(f"Final prediction: {normalized_prediction} (confidence: {confidence})")
            return normalized_prediction, float(confidence), None
        else:
            logger.warning("No predictions found in Roboflow response")
            return "Real Image", 0.5, None
            
    except Exception as e:
        logger.error(f"Error with Roboflow inference: {str(e)}", exc_info=True)
        
        # Check if it's a connection/API error vs a file error
        error_msg = str(e).lower()
        if any(term in error_msg for term in ['connection', 'network', 'timeout', 'api', 'roboflow']):
            return None, None, f"Roboflow API connection error: {str(e)}"
        else:
            return None, None, f"Classification error: {str(e)}"
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/test-roboflow')
def test_roboflow():
    """Test endpoint to check Roboflow connection"""
    try:
        logger.info("Testing Roboflow connection...")
        project = rf.workspace().project("ai-image-detector-dolex")
        model = project.version(8).model
        logger.info("Roboflow model loaded successfully")
        return {"status": "success", "message": "Roboflow connection successful"}
    except Exception as e:
        logger.error(f"Roboflow connection test failed: {str(e)}")
        return {"status": "error", "message": f"Roboflow connection failed: {str(e)}"}
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
            logger.info(f"Starting classification for: {unique_filename}")
            prediction, confidence, error = classify_with_roboflow(filepath)
            
            if error:
                os.remove(filepath)
                logger.error(f"Classification error: {error}")
                return render_template("index.html", error=f"Classification failed: {error}")
            
            if prediction is None or confidence is None:
                os.remove(filepath)
                logger.error("Classification returned None values")
                return render_template("index.html", error="Classification failed: No valid result returned")
            
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