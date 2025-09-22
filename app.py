# Install required dependencies (run this in your terminal or Colab first if needed)
# pip install flask requests matplotlib Pillow numpy zyphra agentic_doc python-dotenv landingai

from flask import Flask, request, render_template, send_file, jsonify, url_for
import json
import numpy as np
import re
import requests
import os
import base64
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from io import BytesIO
from zyphra import ZyphraClient
from agentic_doc.parse import parse
from dotenv import load_dotenv
from landingai import LandingAI
from landingai.predict import OcrPredictor
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# API Keys (replace with your actual keys)
# LANDINGAI_API_KEY = "your_landing_ai_api_key_here"  # LandingAI API key

ZYPHRA_API_KEY = os.getenv("ZYPHRA_API_KEY")

# Load environment variables
load_dotenv()
LANDINGAI_API_KEY = os.getenv("LANDINGAI_API_KEY")
if not LANDINGAI_API_KEY:
    raise ValueError("LANDINGAI_API_KEY environment variable is not set")

app = Flask(__name__)

# Ensure static folder exists for serving audio files
os.makedirs('static', exist_ok=True)

# OCRProcessor class for PDFs (from reference code)
class OCRProcessor:
    def __init__(self):
        """Initialize the OCR processor."""
        self.api_key = os.getenv("VISION_AGENT_API_KEY")
        if not self.api_key:
            raise ValueError("VISION_AGENT_API_KEY environment variable is not set")

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text in markdown format
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        result = parse(file_path)
        if result and len(result) > 0:
            return result[0].markdown
        return ""

# LandingAI OCR Processor for images
class LandingAIOCRProcessor:
    def __init__(self, api_key: str):
        """Initialize the LandingAI OCR processor."""
        self.client = LandingAI(api_key=api_key)
        self.ocr_predictor = OcrPredictor(self.client)

    def extract_text(self, image_bytes: bytes) -> str:
        """
        Extract text from an image using LandingAI OCR.
        
        Args:
            image_bytes: Bytes of the image file
            
        Returns:
            Extracted text as a string
        """
        try:
            # Create a PIL Image from bytes
            image = PILImage.open(BytesIO(image_bytes))
            # Run OCR prediction
            results = self.ocr_predictor.predict(image)
            # Extract text from results (concatenate all predicted texts)
            extracted_text = " ".join([result.prediction for result in results if hasattr(result, 'prediction')])
            return extracted_text
        except Exception as e:
            print(f"Error extracting text with LandingAI: {str(e)}")
            return ""

# NewsToSpeechPipeline class
class NewsToSpeechPipeline:
    def __init__(self, api_key: str):
        self.client = ZyphraClient(api_key=api_key)

    def process_script(self,
                      script: str,
                      output_path: str = "news_audio.webm",
                      language_iso_code: str = "en-us",
                      speaking_rate: float = 15.0,
                      model: str = "zonos-v0.1-transformer",
                      mime_type: str = "audio/mp3",
                      emotion_settings: Optional[Dict[str, float]] = None,
                      voice_reference_path: Optional[str] = None,
                      **kwargs) -> str:
        params = {
            "text": script,
            "language_iso_code": language_iso_code,
            "speaking_rate": speaking_rate,
            "model": model,
            "mime_type": mime_type,
            "output_path": output_path
        }
        if emotion_settings:
            from zyphra.models.audio import EmotionWeights
            emotions = EmotionWeights(**emotion_settings)
            params["emotion"] = emotions
        if voice_reference_path and os.path.exists(voice_reference_path):
            with open(voice_reference_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            params["speaker_audio"] = audio_base64
        params.update(kwargs)
        try:
            result_path = self.client.audio.speech.create(**params)
            print(f"Audio generated successfully and saved to {result_path}")
            return result_path
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            raise

def process_file_to_text(file_bytes: bytes, file_extension: str, output_file: str = None) -> str:
    """
    Extract text from either an image or PDF using appropriate OCR.
    
    Args:
        file_bytes: Bytes of the file
        file_extension: File extension (e.g., '.jpg', '.pdf')
        output_file: Optional path to save extracted text
        
    Returns:
        Extracted text as string
    """
    ocr_markdown = ""
    try:
        if file_extension.lower() in ['.pdf']:
            # Save temp file for PDF processing
            temp_file_path = os.path.join('static', f'temp_file{file_extension}')
            with open(temp_file_path, 'wb') as f:
                f.write(file_bytes)
            
            # Use OCRProcessor for PDF
            ocr_processor = OCRProcessor()
            ocr_markdown = ocr_processor.extract_text(temp_file_path)
            
            # Clean up
            os.remove(temp_file_path)
        else:
            # Assume image for other extensions
            landing_ocr = LandingAIOCRProcessor(api_key=LANDINGAI_API_KEY)
            ocr_markdown = landing_ocr.extract_text(file_bytes)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(ocr_markdown)
            print(f"Extracted text saved to {output_file}")
        
        return ocr_markdown
    except Exception as e:
        error_msg = f"OCR processing failed: {str(e)}"
        print(error_msg)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(error_msg)
        return ""

def process_text_to_structured_news(ocr_text: str, output_file: str = None) -> Dict:
    """
    Use Groq (Llama3) to structure the extracted text into news format.
    This replaces Mistral LLM usage.
    
    Args:
        ocr_text: Extracted text from OCR
        output_file: Optional path to save JSON
        
    Returns:
        Structured news data as dict
    """
    try:
        if not ocr_text.strip():
            raise ValueError("No text extracted from the file")
        
        print("Extracting structured data using Groq LLM...")
        news_structure = {
            "headline": "The main title of the news article",
            "source": "News agency or publication source",
            "location": "Dateline location",
            "body_text": ["Array of paragraphs from the article"],
            "date": "Publication date if available, null if not found"
        }
        prompt = f"""
        This is the OCR text extracted from a news article image or PDF:
        <BEGIN_OCR_TEXT>
        {ocr_text}
        </END_OCR_TEXT>
        Convert this into a structured JSON with the following format:
        {json.dumps(news_structure, indent=2)}
        The output should be strictly JSON with no additional commentary.
        """
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are an AI specialized in extracting structured news data from raw text."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": 2048
            },
            timeout=30
        )
        if response.status_code == 200:
            structured_data = json.loads(response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}"))
        else:
            print(f"Groq API Error: {response.status_code}, {response.text}")
            raise ValueError(f"Failed to structure data: {response.text}")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        
        return structured_data
    except Exception as e:
        error_data = {"error": f"Structuring failed: {str(e)}"}
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2)
        return error_data

def generate_war_reporter_script_groq(summary, headline, location):
    prompt = f"""You are an experienced war correspondent reporting from dangerous conflict zones.
LOCATION: {location if location else "the conflict zone"}
HEADLINE: {headline}
SUMMARY: {summary}
Create a dramatic news report script with the following:
1. Introduce yourself with a unique war reporter persona name and briefly describe where you're reporting from.
2. Present the news in a dramatic, tense style typical of frontline reporting.
3. Include background sounds or environment descriptions in [brackets].
4. Add short pauses indicated by (pause) where appropriate for dramatic effect.
5. End with a signature sign-off phrase and your reporter name.
FORMAT YOUR RESPONSE AS A COMPLETE SCRIPT READY FOR TEXT-TO-SPEECH:
"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are an AI specialized in creating dramatic war correspondent scripts."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            print(f"Groq API Error: {response.status_code}, {response.text}")
    except requests.RequestException as e:
        print(f"Groq API request failed: {str(e)}")
    print("Falling back to rule-based generation due to API failure.")
    return rule_based_script_generation(summary, headline, location)

def rule_based_script_generation(summary, headline, location):
    reporter_names = ["Alex Harker", "Morgan Wells", "Jamie Frost", "Casey Rivers", 
                     "Taylor Stone", "Jordan Reed", "Sam Fletcher", "Riley Hayes"]
    sign_offs = ["Reporting live from the frontlines, {reporter}.",
                 "Back to you in the studio, this is {reporter}.",
                 "This is {reporter}, reporting from {location}.",
                 "For World News Network, this is {reporter} signing off.",
                 "The situation remains fluid. From {location}, I'm {reporter} reporting."]
    sounds = ["[distant explosions]", "[helicopter overhead]", "[sirens wailing]",
             "[crowd noise]", "[wind howling]", "[radio static]", "[gunfire in distance]"]
    np.random.seed(42)
    reporter = np.random.choice(reporter_names)
    sign_off = np.random.choice(sign_offs).format(location=location or "the conflict zone", reporter=reporter)
    opening_sound = np.random.choice(sounds)
    middle_sound = np.random.choice(sounds)
    summary_parts = summary.split('\n\n')
    first_part = summary_parts[0] if summary_parts else summary
    rest_parts = ' '.join(summary_parts[1:]) if len(summary_parts) > 1 else ""
    script = (f"{opening_sound} This is {reporter}, reporting live from {location or 'the conflict zone'}. (pause)\n\n"
              f"{headline}. (pause)\n\n"
              f"{first_part}\n\n"
              f"{middle_sound} (pause)\n\n")
    if rest_parts:
        script += f"{rest_parts}\n\n"
    script += f"{sign_off}"
    return script

def simple_sentence_tokenize(text):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def rank_sentences(sentences):
    if not sentences:
        return []
    keywords = ['arrested', 'charges', 'navy', 'officials', 'incident', 
                'killed', 'attack', 'explosion', 'conflict', 'military']
    scores = []
    for i, sentence in enumerate(sentences):
        position_score = 1.5 if i < len(sentences) * 0.2 else 1.2 if i > len(sentences) * 0.8 else 1.0
        length_score = 1.2 if 10 <= len(sentence.split()) <= 25 else 0.8
        content_score = 1.3 if any(keyword in sentence.lower() for keyword in keywords) else 1.0
        total_score = position_score * length_score * content_score
        scores.append((i, total_score))
    return scores

def extract_top_sentences(sentences, scores, num_sentences=5):
    if not sentences or not scores:
        return []
    num_sentences = min(5, len(sentences))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in scores[:num_sentences]])
    return [sentences[i] for i in top_indices]

def add_war_reporter_style(sentences):
    if not sentences:
        return ""
    transition_phrases = ["Reports from the front line indicate", "Our correspondents on the ground confirm",
                          "Breaking news from the conflict zone:", "Eyewitness accounts suggest",
                          "According to military analysts,", "The situation remains tense as"]
    np.random.seed(42)
    styled_text = f"{np.random.choice(transition_phrases)} {sentences[0]}\n\n"
    for i, sentence in enumerate(sentences[1:-1], 1):
        if i % 2 == 0 and len(sentences) > 3:
            styled_text += f"{np.random.choice(transition_phrases)} {sentence} "
        else:
            styled_text += f"{sentence} "
    if len(sentences) > 1:
        styled_text += f"\n\nThe situation continues to develop as {sentences[-1].lower()}"
    return styled_text

def process_json_news(json_data):
    try:
        headline = json_data.get('headline', 'Breaking News from Conflict Zone')
        body_text = ''
        raw_body_text = json_data.get('body_text', '')
        if isinstance(raw_body_text, List):
            body_text = ' '.join(raw_body_text)
        elif isinstance(raw_body_text, str):
            body_text = raw_body_text
        else:
            body_text = str(raw_body_text)
        location = json_data.get('location', 'the conflict zone')
        full_text = f"{headline}. {body_text}".strip()
        cleaned_text = re.sub(r'\s+', ' ', full_text)
        sentences = simple_sentence_tokenize(cleaned_text)
        scores = rank_sentences(sentences)
        top_sentences = extract_top_sentences(sentences, scores)
        war_reporter_summary = add_war_reporter_style(top_sentences)
        news_script = generate_war_reporter_script_groq(war_reporter_summary, headline, location)
        return {
            'headline': headline,
            'location': location,
            'summary': war_reporter_summary,
            'news_script': news_script
        }
    except Exception as e:
        print(f"Error processing news: {str(e)}")
        return {
            'headline': 'Error Processing News',
            'location': 'Unknown',
            'summary': 'An error occurred while processing the news data.',
            'news_script': 'Unable to generate script due to an error.'
        }

def clean_script(script: str) -> str:
    return re.sub(r'\[.*?\]|\(pause\)', '', script).strip()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty file name'}), 400
        
        # Get file extension
        _, file_extension = os.path.splitext(file.filename)
        
        # Read file bytes
        file_bytes = file.read()
        
        # Extract text using OCR
        text_path = os.path.join('static', 'extracted_text.txt')
        ocr_text = process_file_to_text(file_bytes, file_extension, text_path)
        
        if not ocr_text.strip():
            return jsonify({'error': 'No text could be extracted from the file'}), 400
        
        # Structure the text into JSON
        json_path = os.path.join('static', 'news_article.json')
        json_data = process_text_to_structured_news(ocr_text, json_path)
        
        if 'error' in json_data:
            return jsonify({'error': json_data['error']}), 500
        
        # Process the extracted JSON data
        processed_data = process_json_news(json_data)
        
        # Generate audio from the script
        try:
            tts_pipeline = NewsToSpeechPipeline(api_key=ZYPHRA_API_KEY)
            audio_path = os.path.join('static', 'summary_report.mp3')
            cleaned_script = clean_script(processed_data['news_script'])
            tts_pipeline.process_script(
                script=cleaned_script,
                output_path=audio_path,
                speaking_rate=15.0,
                emotion_settings={"serious": 0.8, "urgent": 0.7}
            )
            audio_url = url_for('static', filename='summary_report.mp3')
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            audio_url = None
        
        return jsonify({
            'json_data': json_data,
            'summary': processed_data['summary'],
            'news_script': processed_data['news_script'],
            'audio_url': audio_url
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)