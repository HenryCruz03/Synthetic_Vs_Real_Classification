import os
import uuid
import time
from typing import List, Dict, Any
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from PIL import Image
from roboflow import Roboflow
from agent import run_agent_on_image
from dotenv import load_dotenv
from flask import send_from_directory
from werkzeug.utils import secure_filename
import logging

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Roboflow client
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

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
        rf=Roboflow(api_key=ROBOFLOW_API_KEY)
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


# NEW: Import your image generation function
from agent import generate_similar_images_with_gemini  # ← Make sure this exists in agent.py


def generate_and_save_images(predicted_class: str, original_image_path: str, num_images: int = 3) -> List[str]:
    """
    Generate similar images and save them to uploads folder.
    Returns list of generated image filenames.
    """
    generated_images = []
    
    logger.info(f"🎨 Generating {num_images} similar images for class: {predicted_class}")
    
    for i in range(num_images):
        try:
            # Generate single image
            image_bytes = generate_similar_images_with_gemini(
                predicted_class=predicted_class,
                original_image_path=original_image_path
            )
            
            # Create unique filename
            timestamp = int(time.time())
            filename = f"generated_{predicted_class.replace(' ', '_')}_{timestamp}_{i}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            # Track the filename
            generated_images.append(filename)
            
            logger.info(f"  ✓ Generated and saved: {filename}")
            
            # Cooldown to avoid quota limits
            if i < num_images - 1:
                time.sleep(2)
                
        except Exception as gen_error:
            logger.error(f"  ❌ Error generating image {i+1}: {gen_error}")
    
    return generated_images


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/test-roboflow')
def test_roboflow():
    """Test endpoint to check Roboflow connection"""
    try:
        logger.info("Testing Roboflow connection...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace().project("ai-image-detector-dolex")
        model = project.version(8).model
        logger.info("Roboflow model loaded successfully")
        return {"status": "success", "message": "Roboflow connection successful"}
    except Exception as e:
        logger.error(f"Roboflow connection test failed: {str(e)}")
        return {"status": "error", "message": f"Roboflow connection failed: {str(e)}"}

@app.route('/agent-demo', methods=["POST"])
def agent_demo():
    """Demo endpoint to run agent and return step-by-step JSON."""
    if 'image' not in request.files:
        return {"status": "error", "message": "No file provided. Use form-data with key 'image'"}, 400

    file = request.files['image']
    if file.filename == '':
        return {"status": "error", "message": "Empty filename"}, 400

    if not allowed_file(file.filename):
        return {"status": "error", "message": "Invalid file type"}, 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, f"demo_{uuid.uuid4().hex}_{filename}")
    file.save(temp_path)

    try:
        label, conf, details = run_agent_on_image(temp_path)
        return {
            "status": "success",
            "final": {"label": label, "confidence": conf},
            "steps": details.get("steps", {}),
        }
    except Exception as e:
        return {"status": "error", "message": f"Agent run failed: {str(e)}"}, 500
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


# NEW: Curation endpoint to upload images to Roboflow for training
@app.route('/curate', methods=["POST"])
def curate():
    """Upload a selected image to Roboflow for training."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        label = data.get('label')
        
        if not filename or not label:
            return jsonify({"success": False, "message": "Missing filename or label"}), 400
        
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "message": "File not found"}), 404
        
        # TODO: Implement actual Roboflow upload for training
        # Example:
        # rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        # project = rf.workspace().project("ai-image-detector-dolex")
        # project.upload(filepath, annotation_name=label)
        
        logger.info(f"✓ Curated image for training: {filename} with label: {label}")
        
        return jsonify({"success": True, "message": "Image uploaded for training"})
        
    except Exception as e:
        logger.error(f"Curation error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/training-curation', methods=["POST"])
def training_curation():
    """Endpoint for training data curation decisions."""
    try:
        data = request.get_json()
        if not data:
            return {"status": "error", "message": "No JSON data provided"}, 400

        action = data.get("action")  # "accept_all", "reject_all", "selective"
        selected_images = data.get("selected_images", [])  # List of image paths to accept
        image_id = data.get("image_id")  # Original image that triggered the enhancement

        if action == "accept_all":
            # Add all generated images to training data
            logger.info(f"Accepting all generated images for training data curation")
            return {
                "status": "success", 
                "message": "All generated images added to training data",
                "action": "accept_all"
            }

        elif action == "reject_all":
            # Reject all generated images
            logger.info(f"Rejecting all generated images")
            return {
                "status": "success", 
                "message": "All generated images rejected",
                "action": "reject_all"
            }

        elif action == "selective":
            # Accept only selected images
            if not selected_images:
                return {"status": "error", "message": "No images selected for selective acceptance"}, 400

            # Process selected images
            accepted_count = 0
            for img_data in selected_images:
                try:
                    # Here you would typically:
                    # 1. Move images to training data folder
                    # 2. Update training dataset metadata
                    # 3. Log the addition for tracking
                    logger.info(f"Adding image to training data: {img_data.get('filename', 'unknown')}")
                    accepted_count += 1
                except Exception as e:
                    logger.error(f"Error processing selected image {img_data}: {str(e)}")
                    continue

            logger.info(f"Selectively accepting {accepted_count} images for training data")
            return {
                "status": "success", 
                "message": f"Selected {accepted_count} images added to training data",
                "action": "selective",
                "accepted_images": selected_images,
                "accepted_count": accepted_count
            }
        else:
            return {"status": "error", "message": "Invalid action. Use 'accept_all', 'reject_all', or 'selective'"}, 400

    except Exception as e:
        logger.error(f"Training curation error: {str(e)}")
        return {"status": "error", "message": f"Training curation failed: {str(e)}"}, 500


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

            # Classify image via agent (detect → refine → validate)
            logger.info(f"Starting agent workflow for: {unique_filename}")
            try:
                prediction, confidence, _details = run_agent_on_image(filepath)
                error = None
            except Exception as e:
                prediction, confidence, error = None, None, f"Agent error: {str(e)}"

            if error:
                os.remove(filepath)
                logger.error(f"Classification error: {error}")
                return render_template("index.html", error=f"Classification failed: {error}")

            if prediction is None or confidence is None:
                os.remove(filepath)
                logger.error("Classification returned None values")
                return render_template("index.html", error="Classification failed: No valid result returned")

            logger.info(f"Classification successful: {prediction} (confidence: {confidence})")

            # NEW: Generate similar images after successful classification
            generated_images = []
            try:
                generated_images = generate_and_save_images(
                    predicted_class=prediction,
                    original_image_path=filepath,
                    num_images=3
                )
                logger.info(f"✓ Generated {len(generated_images)} images")
            except Exception as gen_error:
                logger.error(f"Image generation error: {gen_error}")
                # Continue even if generation fails

            # Extract enhancement data if available
            enhancement_data = None
            if _details and "steps" in _details and "enhancement" in _details["steps"]:
                enhancement_data = _details["steps"]["enhancement"]

            # Debug output
            logger.info(f"DEBUG: Passing {len(generated_images)} images to template: {generated_images}")

            return render_template(
                "result.html",
                prediction=prediction,
                confidence=confidence,
                filename=unique_filename,
                enhancement_data=enhancement_data,
                generated_images=generated_images  # ← NEW: Pass generated images
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