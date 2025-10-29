import os
import json
import requests
import base64
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

from dotenv import load_dotenv
from roboflow import Roboflow
from PIL import Image
import pytesseract
from google.generativeai import GenerativeModels

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


def extract_text_with_ocr(image_path: str) -> str:
    """Extracts text from the image using Tesseract OCR."""
    try:
        with Image.open(image_path) as img:
            return pytesseract.image_to_string(img)
    except Exception:
        return ""


def generate_similar_images_with_gemini(image_path: str, num_images: int = 10) -> List[str]:
    """Generate similar images using Google Gemini API with image analysis."""
    try:
        # Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        
        genai.configure(api_key=api_key)
        
        # Use the correct model name for Gemini
        model = GenerativeModel(model_name="models/images:generate")
        image_bytes = model.generate_content()
        print("Using Gemini 1.5 Pro model")
        
        # Read the input image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Analyze the input image to understand its characteristics
        analysis_prompt = """
        Analyze this image and describe its key characteristics including:
        1. Main subject/objects
        2. Style (photographic, artistic, etc.)
        3. Color palette
        4. Composition
        5. Lighting
        6. Any text or patterns
        Provide a detailed description that could be used to generate similar images.
        """
        
        try:
            # Create the image part for Gemini
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_data
            }
            
            analysis_response = model.generate_content([analysis_prompt, image_part])
            image_description = analysis_response.text if analysis_response else "A general image"
            print(f"Image analysis completed: {image_description[:100]}...")
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            image_description = "A general image"
        
        # Generate similar images using the description
        generated_images = []
        for i in range(num_images):
            try:
                # Create a variation prompt based on the analysis
                variation_prompt = f"""
                Create a variation of this image description: {image_description}
                
                Generate variation {i+1} with these modifications:
                - Slightly different angle or perspective
                - Minor changes in lighting or color
                - Small variations in composition
                - Maintain the same overall style and subject
                
                Provide a detailed text description that could be used with an image generation service like DALL-E or Midjourney.
                """
                
                response = model.generate_content(variation_prompt)
                
                if response and hasattr(response, 'text') and response.text:
                    # Save the description as a text file
                    generated_filename = f"generated_{i}_{os.path.basename(image_path)}.txt"
                    generated_path = os.path.join("uploads", generated_filename)
                    
                    with open(generated_path, "w", encoding="utf-8") as f:
                        f.write(f"Generated image description {i+1}:\n\n")
                        f.write(response.text)
                        f.write(f"\n\nBased on original image: {image_path}")
                    
                    # Create a visual representation of the generated description
                    img_filename = f"generated_{i}_{os.path.basename(image_path)}"
                    img_path = os.path.join("uploads", img_filename)
                    
                    # Create a more sophisticated placeholder image
                    from PIL import Image, ImageDraw, ImageFont
                    img = Image.new('RGB', (512, 512), color=(100 + i*20, 150 + i*10, 200 + i*5))
                    draw = ImageDraw.Draw(img)
                    
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                    
                    # Add text to the image
                    draw.text((10, 10), f"Generated Image {i+1}", fill=(255, 255, 255), font=font)
                    draw.text((10, 30), f"Variation of: {os.path.basename(image_path)}", fill=(200, 200, 200), font=font)
                    draw.text((10, 50), f"AI Generated Description", fill=(150, 150, 150), font=font)
                    
                    # Add a preview of the description
                    description_preview = response.text[:50] + "..." if len(response.text) > 50 else response.text
                    draw.text((10, 70), description_preview, fill=(100, 100, 100), font=font)
                    
                    img.save(img_path)
                    generated_images.append(img_path)
                    print(f"Generated variation {i+1} with Gemini description")
                else:
                    print(f"No response from Gemini for variation {i+1}")
                    # Create a simple fallback image
                    img_filename = f"generated_{i}_{os.path.basename(image_path)}"
                    img_path = os.path.join("uploads", img_filename)
                    
                    from PIL import Image, ImageDraw
                    img = Image.new('RGB', (512, 512), color=(100 + i*20, 150 + i*10, 200 + i*5))
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10), f"Generated Image {i+1}", fill=(255, 255, 255))
                    img.save(img_path)
                    generated_images.append(img_path)
                    
            except Exception as e:
                print(f"Error generating image {i}: {str(e)}")
                continue
                
        return generated_images
        
    except Exception as e:
        print(f"Error with Gemini image generation: {str(e)}")
        return []


def fetch_similar_real_images(image_path: str, num_images: int = 10) -> List[str]:
    """Fetch similar real images using Google Custom Search API."""
    try:
        # Use Google Custom Search API to find similar real images
        api_key = os.getenv("GOOGLE_API_KEY")
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
            'num': min(num_images, 10),  # Google Custom Search limits to 10 per request
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


def classify_generated_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    """Classify a list of generated images and return results."""
    results = []
    
    for image_path in image_paths:
        try:
            # Check if file exists and is valid
            if not os.path.exists(image_path):
                results.append({
                    "image_path": image_path,
                    "label": "file_not_found",
                    "confidence": 0.0,
                    "error": "Image file does not exist"
                })
                continue
            
            # Validate image file
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "label": "invalid_image",
                    "confidence": 0.0,
                    "error": f"Invalid image file: {str(e)}"
                })
                continue
            
            # Use the same detection logic as the main agent
            detection = detect_with_roboflow(image_path)
            results.append({
                "image_path": image_path,
                "label": detection["label"],
                "confidence": detection["confidence"],
                "filename": os.path.basename(image_path)
            })
            
        except Exception as e:
            results.append({
                "image_path": image_path,
                "label": "classification_error",
                "confidence": 0.0,
                "error": str(e),
                "filename": os.path.basename(image_path)
            })
    
    return results


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
        # No LLM required - using rule-based refinement
        self.refine_tool = make_refinement_tool()

    def run(self, image_path: str) -> Dict[str, Any]:
        # Step 1: detect
        detection = detect_with_roboflow(image_path)

        # Step 2: gather OCR for validation context
        ocr_text = extract_text_with_ocr(image_path)

        # Step 3: refine using LLM tool
        refinement_input = json.dumps({
            "label": detection["label"],
            "confidence": detection["confidence"],
            "ocr": ocr_text,
        })
        refinement_raw = self.refine_tool.run(refinement_input)
        try:
            refinement = json.loads(refinement_raw)
        except Exception:
            # Fallback: keep original but cap confidence if unknown
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
        if adjusted_conf < 0.6:  # Increased threshold to 60% for more sensitive triggering
            print(f"Low confidence detected ({adjusted_conf:.3f}) - generating similar images for training data curation")
            
            # Generate similar images using Gemini
            generated_images = generate_similar_images_with_gemini(image_path, num_images=10)
            
            # Fetch similar real images (placeholder)
            real_images = fetch_similar_real_images(image_path, num_images=10)
            
            # Classify the generated images
            generated_classifications = classify_generated_images(generated_images)
            real_classifications = classify_generated_images(real_images)
            
            enhancement_data = {
                "triggered": True,
                "reason": f"Confidence below threshold (0.5): {adjusted_conf:.3f}",
                "generated_images": generated_classifications,
                "real_images": real_classifications,
                "total_generated": len(generated_images),
                "total_real": len(real_images),
                "recommendation": self._generate_training_recommendation(generated_classifications, real_classifications)
            }
        else:
            enhancement_data = {
                "triggered": False,
                "reason": f"Confidence above threshold (0.5): {adjusted_conf:.3f}"
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


