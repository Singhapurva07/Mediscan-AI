from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
from werkzeug.utils import secure_filename
import json
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

load_dotenv()
app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'

# Load TrOCR model (runs locally)
print("🔄 Loading TrOCR model... (this may take a minute on first run)")
try:
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"✅ TrOCR loaded successfully on {device}")
except Exception as e:
    print(f"❌ Failed to load TrOCR: {e}")
    processor = None
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_with_trocr(image_path):
    """Extract text using local TrOCR model - handles handwriting well"""
    try:
        if processor is None or model is None:
            return None, "TrOCR model not loaded. Please restart the server."
        
        print(f"📤 Processing with TrOCR (local)...")
        
        # Open and prepare image
        image = Image.open(image_path).convert("RGB")
        
        # For better results, we'll process the image in chunks/lines
        # This helps TrOCR which works best on single lines of text
        width, height = image.size
        
        # Split image into horizontal strips (simulating text lines)
        strip_height = 80  # Height of each strip
        overlap = 20  # Overlap between strips
        
        extracted_texts = []
        
        y = 0
        while y < height:
            y_end = min(y + strip_height, height)
            
            # Crop strip
            strip = image.crop((0, y, width, y_end))
            
            # Process with TrOCR
            pixel_values = processor(images=strip, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if text.strip():  # Only add non-empty text
                extracted_texts.append(text.strip())
            
            y += strip_height - overlap
        
        # Combine all extracted text
        full_text = ' '.join(extracted_texts)
        
        if not full_text.strip():
            return None, "No text detected in image"
        
        print(f"✅ TrOCR extracted ({len(full_text)} chars):")
        print(f"   {full_text[:300]}...")
        
        return full_text, None
        
    except Exception as e:
        print(f"❌ TrOCR Error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"TrOCR error: {str(e)}"

def enhance_ocr_with_groq(trocr_text):
    """Use Groq to clean up and enhance OCR text"""
    try:
        if not GROQ_API_KEY or not trocr_text:
            return trocr_text, None
        
        print(f"🧠 Enhancing OCR with Groq...")
        
        prompt = f"""The following text was extracted from a medical prescription using OCR. It may contain errors, spacing issues, or unclear words. Please:

1. Fix obvious OCR mistakes
2. Properly format medicine names, dosages, and instructions
3. Add line breaks where appropriate
4. Keep all the original information, just make it more readable

OCR Text:
{trocr_text}

Return the cleaned and formatted text. Keep it medical and accurate."""

        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a medical text formatting assistant. Clean up OCR errors while preserving all medical information."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            enhanced_text = result['choices'][0]['message']['content'].strip()
            print(f"✅ Text enhanced")
            return enhanced_text, None
        else:
            print(f"⚠️ Enhancement failed, using raw OCR")
            return trocr_text, None
            
    except Exception as e:
        print(f"⚠️ Enhancement error: {e}, using raw OCR")
        return trocr_text, None

def analyze_prescription_with_groq(extracted_text):
    """Use Groq to intelligently analyze the prescription"""
    try:
        if not GROQ_API_KEY:
            return None, "Groq API key missing"
        
        print(f"🧠 Analyzing prescription with Groq AI...")
        
        prompt = f"""You are a medical prescription analyst. Analyze this extracted prescription text and provide detailed structured information.

Extracted Text:
{extracted_text}

Analyze and return a JSON object with:
{{
  "medicines_found": [
    {{"name": "medicine name", "dosage": "dosage if found", "frequency": "frequency if found", "category": "type like vitamin/antibiotic/painkiller/supplement"}}
  ],
  "condition_category": "one of: nutritional_support, fever, headache, cold, cough, diabetes, hypertension, pain, allergy, infection, digestive, skin, respiratory, cardiovascular, mental_health, general_wellness, other",
  "primary_symptoms": ["list of symptoms if mentioned"],
  "prescription_type": "treatment/supplement/preventive/maintenance",
  "confidence": "high/medium/low",
  "reasoning": "brief explanation of your analysis",
  "recommendations_needed": "what kind of medicine info would help this patient"
}}

CRITICAL RULES:
- If you see: vitamins, probiotics, nutritional drinks, protein powder, ensure, multivitamin, calcium, vitamin D, omega-3, iron supplements → condition_category: "nutritional_support"
- If you see: paracetamol, ibuprofen, aspirin → condition_category: "fever" or "pain"
- If you see: antibiotics (amoxicillin, azithromycin, ciprofloxacin) → condition_category: "infection"
- If you see: cetirizine, loratadine, antihistamine → condition_category: "allergy"
- If you see: omeprazole, antacid → condition_category: "digestive"
- If unclear or wellness focus → condition_category: "general_wellness"

Analyze what's ACTUALLY written in the text. Don't assume or guess.

Return ONLY valid JSON, no markdown."""

        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a medical text analysis expert. Always respond with valid JSON only. Analyze prescriptions accurately based on what medicines are actually present."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Groq Analysis Error: {response.status_code}")
            return None, f"Groq API error: {response.status_code}"
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Clean JSON from markdown
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            if content.startswith('json'):
                content = content[4:].strip()
        
        analysis = json.loads(content)
        print(f"✅ Analysis: {analysis['condition_category']} ({analysis['confidence']})")
        print(f"   Medicines found: {len(analysis.get('medicines_found', []))}")
        
        return analysis, None
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Content was: {content[:500]}")
        return None, "Failed to parse AI response"
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        return None, str(e)

def get_medicine_recommendations(category, analysis=None):
    """Get medicine recommendations based on category"""
    
    medicine_db = {
        'nutritional_support': [
            {'name': 'Multivitamin Complex', 'generic': 'Multiple vitamins & minerals', 'dosage': '1 tablet once daily', 'warning': 'Take with food for better absorption', 'type': 'Supplement', 'use': 'General health maintenance'},
            {'name': 'Vitamin D3', 'generic': 'Cholecalciferol', 'dosage': '1000-2000 IU daily', 'warning': 'Important for bone health and immunity', 'type': 'Vitamin', 'use': 'Bone health, immune support'},
            {'name': 'Probiotic Complex', 'generic': 'Lactobacillus & Bifidobacterium', 'dosage': '1 capsule daily', 'warning': 'Best taken on empty stomach', 'type': 'Probiotic', 'use': 'Gut health, digestion'},
            {'name': 'Protein Supplement (Ensure)', 'generic': 'Complete nutritional supplement', 'dosage': '1-2 servings daily', 'warning': 'Mix with milk or water', 'type': 'Nutritional', 'use': 'Protein supplementation'},
            {'name': 'Iron + Folic Acid', 'generic': 'Ferrous sulfate + Folic acid', 'dosage': '1 tablet daily', 'warning': 'May cause constipation, take with vitamin C', 'type': 'Mineral', 'use': 'Anemia prevention'},
            {'name': 'Calcium + Vitamin D', 'generic': 'Calcium carbonate + Cholecalciferol', 'dosage': '500-600mg twice daily', 'warning': 'Important for bone health', 'type': 'Mineral', 'use': 'Bone strength'},
            {'name': 'Omega-3 Fish Oil', 'generic': 'EPA & DHA', 'dosage': '1000mg daily', 'warning': 'Heart and brain health support', 'type': 'Supplement', 'use': 'Cardiovascular health'},
        ],
        'general_wellness': [
            {'name': 'Multivitamin', 'generic': 'Multiple vitamins', 'dosage': '1 tablet daily', 'warning': 'Take with meals', 'type': 'Supplement', 'use': 'General wellness'},
            {'name': 'Vitamin C', 'generic': 'Ascorbic acid', 'dosage': '500-1000mg daily', 'warning': 'Immune support', 'type': 'Vitamin', 'use': 'Immunity boost'},
            {'name': 'Zinc', 'generic': 'Zinc sulfate', 'dosage': '15-30mg daily', 'warning': 'Immune function', 'type': 'Mineral', 'use': 'Immune health'},
        ],
        'fever': [
            {'name': 'Paracetamol (Acetaminophen)', 'generic': 'Acetaminophen', 'dosage': '500-1000mg every 4-6 hours', 'warning': 'Max 4000mg/day. Liver damage risk if exceeded', 'type': 'Antipyretic', 'use': 'Fever, mild pain'},
            {'name': 'Ibuprofen', 'generic': 'Ibuprofen', 'dosage': '200-400mg every 4-6 hours', 'warning': 'Take with food. Avoid if stomach ulcers', 'type': 'NSAID', 'use': 'Fever, inflammation'},
            {'name': 'Aspirin', 'generic': 'Acetylsalicylic acid', 'dosage': '325-650mg every 4 hours', 'warning': 'Not for children under 12', 'type': 'NSAID', 'use': 'Fever, pain'},
            {'name': 'Mefenamic Acid', 'generic': 'Mefenamic acid', 'dosage': '500mg three times daily', 'warning': 'Take after meals. Max 7 days', 'type': 'NSAID', 'use': 'Fever, moderate pain'},
        ],
        'headache': [
            {'name': 'Paracetamol', 'generic': 'Acetaminophen', 'dosage': '500-1000mg every 6 hours', 'warning': 'First-line treatment', 'type': 'Analgesic', 'use': 'Headache, fever'},
            {'name': 'Ibuprofen', 'generic': 'Ibuprofen', 'dosage': '400-600mg every 8 hours', 'warning': 'Good for tension headaches', 'type': 'NSAID', 'use': 'Headache, inflammation'},
            {'name': 'Sumatriptan', 'generic': 'Sumatriptan', 'dosage': '50-100mg as needed', 'warning': 'Prescription only. For migraines', 'type': 'Triptan', 'use': 'Migraine'},
            {'name': 'Aspirin + Caffeine', 'generic': 'Aspirin/Caffeine combo', 'dosage': '250-500mg as needed', 'warning': 'Caffeine enhances pain relief', 'type': 'Combination', 'use': 'Headache'},
        ],
        'cold': [
            {'name': 'Cetirizine', 'generic': 'Cetirizine HCl', 'dosage': '10mg once daily', 'warning': 'May cause drowsiness', 'type': 'Antihistamine', 'use': 'Runny nose, sneezing'},
            {'name': 'Pseudoephedrine', 'generic': 'Pseudoephedrine', 'dosage': '30-60mg every 4-6 hours', 'warning': 'Decongestant. Avoid before sleep', 'type': 'Decongestant', 'use': 'Nasal congestion'},
            {'name': 'Loratadine', 'generic': 'Loratadine', 'dosage': '10mg once daily', 'warning': 'Non-drowsy', 'type': 'Antihistamine', 'use': 'Cold symptoms'},
            {'name': 'Phenylephrine', 'generic': 'Phenylephrine', 'dosage': '10mg every 4 hours', 'warning': 'Nasal decongestant', 'type': 'Decongestant', 'use': 'Stuffy nose'},
        ],
        'cough': [
            {'name': 'Dextromethorphan', 'generic': 'Dextromethorphan', 'dosage': '10-20mg every 4 hours', 'warning': 'For dry cough', 'type': 'Antitussive', 'use': 'Dry cough'},
            {'name': 'Guaifenesin', 'generic': 'Guaifenesin', 'dosage': '200-400mg every 4 hours', 'warning': 'Expectorant. Helps loosen mucus', 'type': 'Expectorant', 'use': 'Productive cough'},
            {'name': 'Bromhexine', 'generic': 'Bromhexine', 'dosage': '8mg three times daily', 'warning': 'Mucolytic', 'type': 'Mucolytic', 'use': 'Chest congestion'},
        ],
        'infection': [
            {'name': 'Amoxicillin', 'generic': 'Amoxicillin', 'dosage': '500mg three times daily', 'warning': 'Prescription required. Complete full course', 'type': 'Antibiotic', 'use': 'Bacterial infections'},
            {'name': 'Azithromycin', 'generic': 'Azithromycin', 'dosage': '500mg once daily for 3-5 days', 'warning': 'Prescription required', 'type': 'Antibiotic', 'use': 'Respiratory infections'},
            {'name': 'Ciprofloxacin', 'generic': 'Ciprofloxacin', 'dosage': '500mg twice daily', 'warning': 'Prescription required. Broad-spectrum', 'type': 'Antibiotic', 'use': 'Various infections'},
            {'name': 'Cephalexin', 'generic': 'Cephalexin', 'dosage': '500mg four times daily', 'warning': 'Prescription required', 'type': 'Antibiotic', 'use': 'Skin and soft tissue infections'},
        ],
        'diabetes': [
            {'name': 'Metformin', 'generic': 'Metformin HCl', 'dosage': '500-1000mg twice daily', 'warning': 'Take with meals', 'type': 'Biguanide', 'use': 'Type 2 diabetes'},
            {'name': 'Glipizide', 'generic': 'Glipizide', 'dosage': '5-10mg before meals', 'warning': 'Monitor blood sugar', 'type': 'Sulfonylurea', 'use': 'Type 2 diabetes'},
            {'name': 'Insulin Glargine', 'generic': 'Long-acting insulin', 'dosage': 'As prescribed', 'warning': 'Prescription required. Injectable', 'type': 'Insulin', 'use': 'Diabetes'},
        ],
        'hypertension': [
            {'name': 'Amlodipine', 'generic': 'Amlodipine besylate', 'dosage': '5-10mg once daily', 'warning': 'Monitor blood pressure', 'type': 'CCB', 'use': 'High blood pressure'},
            {'name': 'Lisinopril', 'generic': 'Lisinopril', 'dosage': '10-40mg once daily', 'warning': 'May cause dry cough', 'type': 'ACE Inhibitor', 'use': 'Hypertension'},
            {'name': 'Losartan', 'generic': 'Losartan potassium', 'dosage': '50-100mg once daily', 'warning': 'Good alternative to ACE inhibitors', 'type': 'ARB', 'use': 'High BP'},
        ],
        'pain': [
            {'name': 'Paracetamol', 'generic': 'Acetaminophen', 'dosage': '500-1000mg every 6 hours', 'warning': 'First choice for mild pain', 'type': 'Analgesic', 'use': 'Mild to moderate pain'},
            {'name': 'Ibuprofen', 'generic': 'Ibuprofen', 'dosage': '400-600mg every 8 hours', 'warning': 'For inflammatory pain', 'type': 'NSAID', 'use': 'Pain, inflammation'},
            {'name': 'Tramadol', 'generic': 'Tramadol HCl', 'dosage': '50-100mg every 6 hours', 'warning': 'Prescription required', 'type': 'Opioid', 'use': 'Moderate pain'},
            {'name': 'Diclofenac', 'generic': 'Diclofenac sodium', 'dosage': '50mg two-three times daily', 'warning': 'Strong NSAID. Take with food', 'type': 'NSAID', 'use': 'Strong pain relief'},
        ],
        'allergy': [
            {'name': 'Cetirizine', 'generic': 'Cetirizine HCl', 'dosage': '10mg once daily', 'warning': 'May cause mild drowsiness', 'type': 'Antihistamine', 'use': 'Allergic reactions'},
            {'name': 'Fexofenadine', 'generic': 'Fexofenadine', 'dosage': '120-180mg once daily', 'warning': 'Non-drowsy', 'type': 'Antihistamine', 'use': 'Allergies'},
            {'name': 'Loratadine', 'generic': 'Loratadine', 'dosage': '10mg once daily', 'warning': '24-hour relief', 'type': 'Antihistamine', 'use': 'Seasonal allergies'},
            {'name': 'Levocetirizine', 'generic': 'Levocetirizine', 'dosage': '5mg once daily', 'warning': 'More potent than cetirizine', 'type': 'Antihistamine', 'use': 'Allergic rhinitis'},
        ],
        'digestive': [
            {'name': 'Omeprazole', 'generic': 'Omeprazole', 'dosage': '20-40mg once daily', 'warning': 'For acid reflux. Take before meals', 'type': 'PPI', 'use': 'Acid reflux, GERD'},
            {'name': 'Ranitidine', 'generic': 'Ranitidine', 'dosage': '150mg twice daily', 'warning': 'H2 blocker', 'type': 'Antacid', 'use': 'Heartburn'},
            {'name': 'Pantoprazole', 'generic': 'Pantoprazole', 'dosage': '40mg once daily', 'warning': 'PPI for stomach acid', 'type': 'PPI', 'use': 'Stomach ulcers'},
        ],
        'other': [
            {'name': 'Consult Healthcare Provider', 'generic': 'Professional medical consultation', 'dosage': 'As needed', 'warning': 'For accurate diagnosis and treatment plan', 'type': 'Medical Advice', 'use': 'Proper medical evaluation'},
        ]
    }
    
    return medicine_db.get(category, medicine_db['other'])[:7]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        print("\n" + "="*60)
        print("📥 NEW REQUEST")
        print("="*60)
        
        if 'file' in request.files:
            file = request.files['file']
            print(f"📁 File: {file.filename}")
            
            if not file or not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # STEP 1: OCR with TrOCR (LOCAL)
            print("\n🔍 STEP 1: Local TrOCR Extraction")
            raw_text, ocr_error = extract_text_with_trocr(filepath)
            
            # Cleanup
            try:
                os.remove(filepath)
            except:
                pass
            
            if not raw_text:
                return jsonify({'error': f'OCR failed: {ocr_error}'}), 400
            
            # STEP 2: Enhance OCR text with Groq (optional)
            print("\n✨ STEP 2: Enhancing OCR text")
            extracted_text, _ = enhance_ocr_with_groq(raw_text)
            if not extracted_text:
                extracted_text = raw_text
            
            # STEP 3: Analyze with Groq
            print("\n🧠 STEP 3: AI Analysis")
            analysis, analysis_error = analyze_prescription_with_groq(extracted_text)
            
            if not analysis:
                print(f"⚠️ Analysis failed: {analysis_error}")
                return jsonify({'error': f'AI analysis failed: {analysis_error}'}), 400
            
            # STEP 4: Get recommendations
            print("\n💊 STEP 4: Medicine Recommendations")
            category = analysis.get('condition_category', 'other')
            medicines = get_medicine_recommendations(category, analysis)
            
            print(f"✅ Returning {len(medicines)} recommendations for: {category}")
            print("="*60 + "\n")
            
            return jsonify({
                'extracted_text': extracted_text,
                'detected_disease': category.replace('_', ' ').title(),
                'confidence': analysis.get('confidence', 'medium'),
                'reasoning': analysis.get('reasoning', ''),
                'medicines_in_prescription': analysis.get('medicines_found', []),
                'prescription_type': analysis.get('prescription_type', 'unknown'),
                'medicines': medicines,
                'medicine_count': len(medicines),
                'ai_analysis': analysis
            })
        
        elif 'illness' in request.form:
            category = request.form.get('illness')
            print(f"💊 Manual selection: {category}")
            medicines = get_medicine_recommendations(category)
            
            return jsonify({
                'detected_disease': category.replace('_', ' ').title(),
                'confidence': 'manual',
                'medicines': medicines,
                'medicine_count': len(medicines)
            })
        
        return jsonify({'error': 'No file or illness provided'}), 400
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 MEDISCAN AI - Local TrOCR + Groq Analysis")
    print("="*60)
    print("🌐 Server: http://localhost:5000")
    print(f"🔍 OCR: TrOCR (Local) {'✅' if model else '❌'}")
    print(f"🧠 AI: Groq API {'✅ Configured' if GROQ_API_KEY else '❌ NOT SET'}")
    print("="*60 + "\n")
    
    if not GROQ_API_KEY:
        print("⚠️  WARNING: GROQ_API_KEY not found!")
        print("   Add it to your .env file: GROQ_API_KEY=your_key_here")
        print("   Get your key at: https://console.groq.com/keys\n")
    
    if not model:
        print("⚠️  WARNING: TrOCR model not loaded!")
        print("   Install: pip install torch transformers pillow\n")
    
    app.run(debug=True, port=5000, use_reloader=False)