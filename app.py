import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from roboflow import Roboflow
from dotenv import load_dotenv
from flask import send_from_directory
load_dotenv()
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
# Initialize Roboflow client

rf=Roboflow(api_key="9VDk4sfMMuanS6OE8aRU")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def classify_with_roboflow(image_path):
    """
    Use Roboflow SDK to classify the image
    """
    try:
        project = rf.workspace().project("ai-image-detector-dolex")
        model = project.version(8).model
        result = model.predict(image_path).json()
        
        if 'predictions' in result and len(result['predictions']) > 0:
            prediction = result['predictions'][0]['class']
            confidence = result['predictions'][0]['confidence']
        else:
            prediction = "unknown"
            confidence = 0.0
            
        return prediction, confidence, None
    except Exception as e:
        print(f"Error with Roboflow inference: {str(e)}")
        return None, None, f"Classification error: {str(e)}"
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route("/", methods=["GET", "POST"])

def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            
            prediction, confidence, error = classify_with_roboflow(filepath)
            
            if error:
    
                os.remove(filepath)
                return render_template("index.html", error=error)
            
            

            
            return render_template(
                "result.html",
                prediction=prediction,
                confidence=confidence,
                filename=filename)

        else:
            error = "Please upload a valid image file (jpg, png, webp)."
            return render_template("index.html", error=error)
    
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)