import os
import json
import requests
import base64
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO
import time

from dotenv import load_dotenv
from roboflow import Roboflow
from PIL import Image as PILImage

import pytesseract
from google.generativeai import GenerativeModel
from google.generativeai import list_models

from google.cloud import aiplatform
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# LangChain core imports
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()





class DetectionResult(dict):
    """Typed container for detection results."""

    


def detect_with_roboflow(image_path: str) -> DetectionResult:
    """Run Roboflow model to detect whether image is AI-generated vs real.

    Returns a dict with keys: label, confidence, raw
    """
    
    
    
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY", "9VDk4sfMMuanS6OE8aRU"))
    project = rf.workspace().project("ai-image-detector-dolex")
    model = project.version(8).model
    result = model.predict(image_path)
    result_json = result.json() if hasattr(result, "json") else result

    # Coerce to our normalized schema
    predictions: List[Dict[str, Any]] = []
    if isinstance(result_json, dict):
        if "predictions" in result_json:
            predictions = result_json["predictions"]
        elif "results" in result_json:
            predictions = result_json["results"]
    elif isinstance(result_json, list):
        predictions = result_json

    if predictions:
        first = predictions[0]
        label = first.get("class", first.get("label", "unknown"))
        confidence = float(first.get("confidence", first.get("score", 0.0)))
    else:
        label = "unknown"
        confidence = 0.0

    # Normalize to two buckets used by the app
    label_lower = str(label).lower()
    if any(t in label_lower for t in ["synthetic", "ai-generated", "ai_generated", "ai generated", "fake", "artificial"]):
        normalized = "AI-Generated Image"
    elif any(t in label_lower for t in ["real", "natural", "authentic", "genuine"]):
        normalized = "Real Image"
    else:
        normalized = "AI-Generated Image" if confidence > 0.5 else "Real Image"

    return DetectionResult(label=normalized, confidence=confidence, raw=result_json)


def _classify(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Classify a list of images using Roboflow and generate similar images.
    Returns a list of classification results with generated image filenames.
    """
    classifications = []
    
    for img_path in image_paths:
        try:
            # Classify the image
            result = detect_with_roboflow(img_path)
            
            # Track generated images for this classification
            generated_images = []
            
            # Generate similar images (one at a time to avoid quota)
            predicted_class = result["label"]
            num_images_to_generate = 3
            
            print(f"🎨 Generating {num_images_to_generate} similar images for class: {predicted_class}")
            
            for i in range(num_images_to_generate):
                try:
                    # Generate single image
                    image_bytes = generate_similar_images_with_gemini(
                        predicted_class=predicted_class,
                        original_image_path=img_path
                    )
                    
                    # Create unique filename
                    timestamp = int(time.time())
                    filename = f"generated_{predicted_class}_{timestamp}_{i}.jpg"
                    filepath = os.path.join('uploads', filename)
                    
                    # Save the image
                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)
                    
                    # Track the filename (just the name, not full path)
                    generated_images.append(filename)
                    
                    print(f"  ✓ Generated and saved: {filename}")
                    
                    # Cooldown to avoid quota limits
                    if i < num_images_to_generate - 1:  # Don't sleep after last image
                        time.sleep(2)
                        
                except Exception as gen_error:
                    print(f"  ❌ Error generating image {i+1}: {gen_error}")
            
            # Add classification result with generated images
            classifications.append({
                "image_path": img_path,
                "label": result["label"],
                "confidence": result["confidence"],
                "generated_images": generated_images  # ← NEW: Include generated filenames
            })
            
            print(f"✓ Classified {os.path.basename(img_path)}: {result['label']} ({result['confidence']:.3f})")
            print(f"  Generated {len(generated_images)} images")
            
        except Exception as e:
            print(f"❌ Error classifying {img_path}: {e}")
            classifications.append({
                "image_path": img_path,
                "label": "unknown",
                "confidence": 0.0,
                "generated_images": [],  # ← Empty list on error
                "error": str(e)
            })
    
    return classifications

def extract_text_with_ocr(image_path: str) -> str:
    """Extracts text from the image using Tesseract OCR."""
    try:
        with PILImage.open(image_path) as img:
            return pytesseract.image_to_string(img)
    except Exception:
        return ""


def generate_similar_images_with_gemini(image_path: str, num_images: int = 3) -> List[str]:
    """
    Analyze an image with Gemini models for a descriptive text,
    then generate similar images using Vertex AI Imagen via REST.
    Returns a list of filepaths to the generated images.
    """
    ANALYSIS_MODEL = "models/gemini-2.5-flash-image-preview"
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "your-project-id")
    LOCATION = "us-central1"
    os.makedirs("uploads", exist_ok=True)
    generated_images: List[str] = []
    # === Step 1: Analyze the input image (same as before) ===
    try:
        vision_model = GenerativeModel(model_name=ANALYSIS_MODEL)
        with open(image_path, "rb") as f:
            image_part = {"mime_type": "image/jpeg", "data": f.read()}
        desc_response = vision_model.generate_content([
            "Analyze this image and describe its key characteristics including style, colors, composition, and main subjects.",
            image_part,
        ])
        image_description = (
            desc_response.text
            if hasattr(desc_response, "text") and desc_response.text
            else "A general image"
        )
        print(f"Image analysis: {image_description[:120]}...")
    except Exception as e:
        print(f"Error during analysis: {e}")
        image_description = "A general image"
    # === Step 2: Generate similar images using REST Imagen helper ===
    for i in range(num_images):
        prompt = f"Create a similar image based on this description: {image_description}"
        print(f"Requesting image {i+1}/{num_images} with prompt: {prompt}")
    
        try:
            image_bytes = _generate_reliable_imagen(prompt, PROJECT_ID, LOCATION)
            if not image_bytes:
                print(f"❌ Imagen generation returned no bytes for image {i+1}")
                continue
        # Save the image (PIL verification already done in _generate_reliable_imagen)
            gen_filename = f"imagen4_gen_{i}_{os.path.basename(image_path)}.jpg"
            gen_path = os.path.join("uploads", gen_filename)
            with open(gen_path, "wb") as out_img:
                out_img.write(image_bytes)
            generated_images.append(gen_path)
            print(f"✅ Successfully generated and saved: {gen_path}")
        
        # Add delay between requests to avoid hitting quota
            if i < num_images - 1:  # Don't wait after the last image
                print(f"⏳ Waiting 3 seconds before next request...")
                time.sleep(3)
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")
        # Still wait even on error to avoid hammering the API
            if i < num_images - 1:
                time.sleep(2)
    return generated_images

def _generate_reliable_imagen(prompt: str, project_id: str, location: str) -> Optional[bytes]:
    """
    Generate 3 images using Vertex AI Imagen via REST.
    Returns list of image bytes that are PIL-decodable.
    """
    try:
        # Get credentials
        credentials = None
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            credentials = service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            from google.auth import default
            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

        credentials.refresh(Request())
        access_token = credentials.token

        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/{location}/publishers/google/"
            f"models/imagen-3.0-generate-001:predict"
        )

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "instances": [
                {
                    "prompt": prompt,
                    "aspectRatio": "1:1",
                    "safetySettings": [
                        {"category": "HATE", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "SEXUAL", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "DANGEROUS", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    ],
                }
            ],
            "parameters": {
                "sampleCount": 1,  # Generate 1 images in one call
            },
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)

        if resp.status_code != 200:
            print(f"Imagen API error: {resp.status_code} - {resp.text}")
            return None

        data = resp.json()
        preds = data.get("predictions") or []
        if not preds:
            print(f"Imagen response has no predictions: {data}")
            return None

       
        # Extract the single image
        if not preds:
            return None
    
        pred = preds[0]
        img_b64 = pred.get("bytesBase64Encoded")
        if not img_b64:
            print(f"No bytesBase64Encoded in prediction")
            return None
        try:
            img_bytes = base64.b64decode(img_b64)
            test_img = PILImage.open(BytesIO(img_bytes))
            test_img.verify()
            print(f"✓ Successfully generated and decoded image")
            return img_bytes  # Return single bytes object, not list
        except Exception as e:
            print(f"❌ Error decoding image: {e}")
            return None

    except Exception as e:
        print(f"Error in _generate_reliable_imagen: {e}")
        return None


def fetch_similar_real_images(image_path: str, num_images: int = 3) -> List[str]:
    """Fetch similar real images using Google Custom Search API."""

    try:

        # Use Google Custom Search API to find similar real images

        api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")  # Ensure this is set correctly

        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")  # You'll need to create a Custom Search Engine

        if not api_key or not search_engine_id:

            print("Google Custom Search API not configured - skipping real image search")

            return []

        # Extract text from the image for search query

        ocr_text = extract_text_with_ocr(image_path)

        search_query = f"real photo {ocr_text}".strip() if ocr_text else "real photograph"

        # Search for similar images

        search_url = "https://www.googleapis.com/customsearch/v1"

        params = {

            'key': api_key,

            'cx': search_engine_id,

            'q': search_query,

            'searchType': 'image',

            'num': min(num_images, 3),  # Google Custom Search limits to 10 per request

            'imgType': 'photo',

            'imgSize': 'medium'

        }

        response = requests.get(search_url, params=params)

        if response.status_code == 200:

            data = response.json()

            real_images = []

            for i, item in enumerate(data.get('items', [])):

                try:

                    # Download the image

                    img_url = item['link']

                    img_response = requests.get(img_url, timeout=10)

                    if img_response.status_code == 200:

                        # Save the image

                        real_filename = f"real_{i}_{os.path.basename(image_path)}"

                        real_path = os.path.join("uploads", real_filename)

                        with open(real_path, "wb") as f:

                            f.write(img_response.content)

                        real_images.append(real_path)

                except Exception as e:

                    print(f"Error downloading real image {i}: {str(e)}")

                    continue

            return real_images

        else:

            print(f"Google Custom Search API error: {response.status_code}")

            return []

    except Exception as e:

        print(f"Error fetching similar real images: {str(e)}")

        return []
def upload_to_roboflow_for_training(image_path: str, label: str) -> bool:
    """
    Upload an image to Roboflow for training data.
    
    Args:
        image_path: Path to the image file
        label: Label for the image ("AI-Generated Image" or "Real Image")
    
    Returns:
        True if successful, False otherwise
    """
    try:
        api_key = os.getenv("ROBOFLOW_API_KEY")
       
        project = "ai-image-detector-dolex"
        
        # Roboflow upload API endpoint
        upload_url = f"https://api.roboflow.com/dataset/{project}/upload"
        
        # Normalize label to match Roboflow classes
        roboflow_label = "ai-generated" if "AI-Generated" in label else "real"
        
        params = {
            "api_key": api_key,
            "name": os.path.basename(image_path),
            "split": "train",  # or "valid" or "test"
        }
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            # Upload as multipart form data
            files = {
                "file": image_file
            }
            
            response = requests.post(
                upload_url,
                params=params,
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            image_id = result.get("id")
            
            # Now annotate the image with the label
            annotate_url = f"https://api.roboflow.com/dataset/{project}/annotate/{image_id}"
            annotation_data = {
                "api_key": api_key,
                "name": roboflow_label,
                # For classification, we just need the label
                "annotation": {
                    "classification": roboflow_label
                }
            }
            
            anno_response = requests.post(annotate_url, json=annotation_data, timeout=30)
            
            if anno_response.status_code == 200:
                print(f"✅ Successfully uploaded and annotated {os.path.basename(image_path)} as '{roboflow_label}'")
                return True
            else:
                print(f"❌ Failed to annotate image: {anno_response.status_code} - {anno_response.text}")
                return False
        else:
            print(f"❌ Failed to upload image: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading to Roboflow: {e}")
        return False


def make_refinement_tool() -> Tool:
    """Returns a simple rule-based refinement tool (no LLM required)."""
    
    def _run(input_str: str) -> str:
        data = json.loads(input_str)
        label = data.get("label", "unknown")
        confidence = float(data.get("confidence", 0.0))
        ocr_text = data.get("ocr", "").strip()
        
        # Simple rule-based refinement
        adjusted_confidence = confidence
        
        # If OCR text is very long, might indicate real document
        if len(ocr_text) > 50:
            if "AI-Generated Image" in label:
                adjusted_confidence *= 0.8  # Reduce confidence for AI if lots of text
            else:
                adjusted_confidence = min(1.0, adjusted_confidence * 1.1)  # Boost real
        
        # If confidence is very low, be more conservative
        if confidence < 0.3:
            adjusted_confidence = 0.2
            rationale = "Low confidence detection - defaulting to conservative estimate"
        elif confidence > 0.8:
            rationale = "High confidence detection - trusting model result"
        else:
            rationale = "Medium confidence - applying OCR-based adjustment"
        
        return json.dumps({
            "final_label": label,
            "adjusted_confidence": round(adjusted_confidence, 3),
            "rationale": rationale
        })

    return Tool(
        name="refine_detection",
        description="Refine detector decision using rule-based logic considering confidence and OCR evidence. Input is JSON string with label, confidence, ocr.",
        func=_run,
    )


class WorkflowAgent:
    """Agent orchestrating detect → refine → validate for images with low-confidence enhancement."""

    def __init__(self) -> None:

        self.refine_tool = make_refinement_tool()

    def run(self, image_path: str) -> Dict[str, Any]:

        detection = detect_with_roboflow(image_path)

       
        ocr_text = extract_text_with_ocr(image_path)

        
        refinement_input = json.dumps({
            "label": detection["label"],
            "confidence": detection["confidence"],
            "ocr": ocr_text,
        })
        refinement_raw = self.refine_tool.run(refinement_input)
        try:
            refinement = json.loads(refinement_raw)
        except Exception:
            refinement = {
                "final_label": detection["label"],
                "adjusted_confidence": float(detection["confidence"]) * 0.9,
                "rationale": "LLM refinement failed to parse; defaulting to detector with small penalty.",
            }

        # Step 4: simple validation logic (sanity check)
        final_label = refinement.get("final_label", detection["label"]) or detection["label"]
        adjusted_conf = float(refinement.get("adjusted_confidence", detection["confidence"]))
        adjusted_conf = max(0.0, min(1.0, adjusted_conf))

        # Step 5: Check if confidence is below threshold and generate/fetch similar images
        enhancement_data = None
        print(f"Final confidence: {adjusted_conf:.3f} (threshold: 0.6)")
        if adjusted_conf < 0.6:
            try:
                generated_images = generate_similar_images_with_gemini(image_path, num_images=3)
                real_images = fetch_similar_real_images(image_path, num_images=3)
        
        # Classify the generated images
                generated_classifications = _classify(generated_images)
                real_classifications = _classify(real_images)
    
                upload_results = {
                "generated": [],
                "real": []
                }
                print("\n=== Uploading to Roboflow for Training ===")
                for classification in generated_classifications:
                    success = upload_to_roboflow_for_training(
                        classification["image_path"],
                        classification["label"]
    )
                    upload_results["generated"].append({
                        "image": os.path.basename(classification["image_path"]),
                        "success": success
    })
                for classification in real_classifications:
                    success = upload_to_roboflow_for_training(
                        classification["image_path"],
                        classification["label"]
    )
                    upload_results["real"].append({
                        "image": os.path.basename(classification["image_path"]),
                        "success": success
    })  
                print(f"Upload Summary: {sum(r['success'] for r in upload_results['generated'])} generated, "f"{sum(r['success'] for r in upload_results['real'])} real images uploaded")
        
                enhancement_data = {
                        "triggered": True,
                        "reason": f"Confidence below threshold (0.6f): {adjusted_conf:.3f}",
                        "generated_images": generated_classifications,
                        "real_images": real_classifications,
                        "generated_image_urls": [f"/uploads/{os.path.basename(img)}" for img in generated_images],
                        "real_image_urls": [f"/uploads/{os.path.basename(img)}" for img in real_images],
                        "total_generated": len(generated_images),
                        "total_real": len(real_images),
                        "upload_results": upload_results,
                        "recommendation": self._generate_training_recommendation(generated_classifications, real_classifications)

        }
            except Exception as e:
                print(f"Enhancement failed: {str(e)}")
                enhancement_data = {
                    "triggered": True,
                    "error": str(e),
                    "reason": f"Confidence below threshold but enhancement failed"
        }
        else:
            enhancement_data = {
                "triggered": False,
                "reason": f"Confidence above threshold (0.6): {adjusted_conf:.3f}"
    }
            
            

        return {
            "steps": {
                "detection": detection,
                "ocr_text": ocr_text,
                "refinement": refinement,
                "enhancement": enhancement_data,
            },
            "final": {
                "label": final_label,
                "confidence": adjusted_conf,
            },
        }

    def _generate_training_recommendation(self, generated_classifications: List[Dict], real_classifications: List[Dict]) -> str:
        """Generate a recommendation for training data curation."""
        if not generated_classifications and not real_classifications:
            return "No similar images were generated/fetched. Consider manual data collection."
        
        # Analyze the classifications
        generated_labels = [c["label"] for c in generated_classifications if "label" in c]
        real_labels = [c["label"] for c in real_classifications if "label" in c]
        
        ai_generated_count = generated_labels.count("AI-Generated Image")
        real_count = real_labels.count("Real Image")
        
        if ai_generated_count > 0 and real_count > 0:
            return f"Mixed results: {ai_generated_count} AI-generated, {real_count} real images. Review each image individually for accurate labeling."
        elif ai_generated_count > 0:
            return f"All {ai_generated_count} generated images classified as AI-generated. Add to training data with AI-generated labels."
        elif real_count > 0:
            return f"All {real_count} real images classified as real. Add to training data with real image labels."
        else:
            return "Generated images have unclear classifications. Manual review recommended before adding to training data."


 
def run_agent_on_image(image_path: str) -> Tuple[str, float, Dict[str, Any]]:
    agent = WorkflowAgent()
    result = agent.run(image_path)
    final = result["final"]
    return final["label"], float(final["confidence"]), result


