"""
Flask Application for LLM Compression Model Comparison
Deploys both BERT-base (teacher) and compressed model (student/distilled) for side-by-side comparison
"""

import os
import time
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configuration
MODEL_CONFIG = {
    'teacher': {
        'path': os.getenv('TEACHER_MODEL_PATH', './models/teacher'),
        'name': 'BERT-base',
        'description': 'Original BERT-base model (109M parameters)',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    'student': {
        'path': os.getenv('STUDENT_MODEL_PATH', './models/student'),
        'name': 'DistilBERT (Compressed)',
        'description': 'Compressed DistilBERT model (67M parameters)',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
}

# Global variables for models (loaded once at startup)
teacher_model = None
teacher_tokenizer = None
student_model = None
student_tokenizer = None

def load_models():
    """Load both teacher and student models"""
    global teacher_model, teacher_tokenizer, student_model, student_tokenizer
    
    print("🔄 Loading models...")
    
    try:
        # Load teacher model (BERT-base)
        teacher_path = MODEL_CONFIG['teacher']['path']
        if os.path.exists(teacher_path):
            print(f"📥 Loading teacher model from {teacher_path}")
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
            teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path)
            teacher_model.to(MODEL_CONFIG['teacher']['device'])
            teacher_model.eval()
            print(f"✅ Teacher model loaded on {MODEL_CONFIG['teacher']['device']}")
        else:
            # Fallback: load from HuggingFace if local path doesn't exist
            # Use a sentiment-analysis fine-tuned model for better results
            print(f"⚠️  Teacher model not found at {teacher_path}, loading from HuggingFace...")
            print("   Using sentiment-analysis fine-tuned model for SST-2")
            try:
                # Try to load a fine-tuned SST-2 model
                teacher_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
                teacher_model = AutoModelForSequenceClassification.from_pretrained(
                    "textattack/bert-base-uncased-SST-2"
                )
            except:
                # Fallback to base model if fine-tuned version not available
                print("   Fine-tuned model not available, using base BERT (will need fine-tuning)")
                teacher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                teacher_model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=2
                )
            teacher_model.to(MODEL_CONFIG['teacher']['device'])
            teacher_model.eval()
            print(f"✅ Teacher model loaded from HuggingFace on {MODEL_CONFIG['teacher']['device']}")
    except Exception as e:
        print(f"❌ Error loading teacher model: {e}")
        teacher_model = None
        teacher_tokenizer = None
    
    try:
        # Load student model (DistilBERT)
        student_path = MODEL_CONFIG['student']['path']
        if os.path.exists(student_path):
            print(f"📥 Loading student model from {student_path}")
            student_tokenizer = AutoTokenizer.from_pretrained(student_path)
            student_model = AutoModelForSequenceClassification.from_pretrained(student_path)
            student_model.to(MODEL_CONFIG['student']['device'])
            student_model.eval()
            print(f"✅ Student model loaded on {MODEL_CONFIG['student']['device']}")
        else:
            # Fallback: load from HuggingFace if local path doesn't exist
            # Use a sentiment-analysis fine-tuned model for better results
            print(f"⚠️  Student model not found at {student_path}, loading from HuggingFace...")
            print("   Using sentiment-analysis fine-tuned model: distilbert-base-uncased-finetuned-sst-2-english")
            student_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            student_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english",
                num_labels=2
            )
            student_model.to(MODEL_CONFIG['student']['device'])
            student_model.eval()
            print(f"✅ Student model loaded from HuggingFace on {MODEL_CONFIG['student']['device']}")
    except Exception as e:
        print(f"❌ Error loading student model: {e}")
        student_model = None
        student_tokenizer = None
    
    print("🎉 Model loading complete!")

def predict_sentiment(model, tokenizer, text, device, model_name):
    """
    Predict sentiment for a given text using the specified model
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        text: Input text string
        device: Device to run inference on
        model_name: Name of the model (for error messages)
    
    Returns:
        dict: Prediction results with sentiment, confidence, logits, and latency
    """
    if model is None or tokenizer is None:
        return {
            'error': f'{model_name} model not loaded',
            'sentiment': None,
            'confidence': None,
            'latency_ms': None
        }
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Measure inference time
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Synchronize for accurate GPU timing
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Get predictions
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class] * 100)
        
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'probabilities': {
                'negative': round(float(probabilities[0] * 100), 2),
                'positive': round(float(probabilities[1] * 100), 2)
            },
            'latency_ms': round(latency_ms, 2),
            'logits': logits.cpu().numpy()[0].tolist()
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'sentiment': None,
            'confidence': None,
            'latency_ms': None
        }

def get_model_info(model, model_name):
    """Get information about the model"""
    if model is None:
        return {
            'name': model_name,
            'parameters': 'N/A',
            'status': 'Not loaded'
        }
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        params_m = total_params / 1e6
        return {
            'name': model_name,
            'parameters': f"{params_m:.2f}M",
            'status': 'Loaded'
        }
    except:
        return {
            'name': model_name,
            'parameters': 'N/A',
            'status': 'Error'
        }

@app.route('/')
def index():
    """Render the main chatbot interface"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for sentiment prediction"""
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({
            'error': 'Text input is required',
            'teacher': None,
            'student': None
        }), 400
    
    # Get predictions from both models
    teacher_result = predict_sentiment(
        teacher_model,
        teacher_tokenizer,
        text,
        MODEL_CONFIG['teacher']['device'],
        MODEL_CONFIG['teacher']['name']
    )
    
    student_result = predict_sentiment(
        student_model,
        student_tokenizer,
        text,
        MODEL_CONFIG['student']['device'],
        MODEL_CONFIG['student']['name']
    )
    
    # Calculate speedup if both latencies are available
    speedup = None
    if (teacher_result.get('latency_ms') and student_result.get('latency_ms') and
        student_result.get('latency_ms') > 0):
        speedup = round(teacher_result['latency_ms'] / student_result['latency_ms'], 2)
    
    # Check if predictions match
    agreement = teacher_result.get('sentiment') == student_result.get('sentiment')
    
    return jsonify({
        'text': text,
        'teacher': {
            **teacher_result,
            'model_info': get_model_info(teacher_model, MODEL_CONFIG['teacher']['name'])
        },
        'student': {
            **student_result,
            'model_info': get_model_info(student_model, MODEL_CONFIG['student']['name'])
        },
        'comparison': {
            'speedup': speedup,
            'agreement': agreement,
            'accuracy_difference': None  # Could be calculated if we had ground truth
        }
    })

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """API endpoint to get model information"""
    return jsonify({
        'teacher': {
            **MODEL_CONFIG['teacher'],
            **get_model_info(teacher_model, MODEL_CONFIG['teacher']['name'])
        },
        'student': {
            **MODEL_CONFIG['student'],
            **get_model_info(student_model, MODEL_CONFIG['student']['name'])
        },
        'device': {
            'cuda_available': torch.cuda.is_available(),
            'teacher_device': MODEL_CONFIG['teacher']['device'],
            'student_device': MODEL_CONFIG['student']['device']
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'teacher_loaded': teacher_model is not None,
        'student_loaded': student_model is not None
    })

if __name__ == '__main__':
    # Load models at startup
    load_models()
    
    # Get port from environment variable or use default (5001 to avoid macOS AirPlay conflict)
    port = int(os.getenv('FLASK_PORT', 5001))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    
    # Run the Flask app
    print("\n🚀 Starting Flask server...")
    print(f"📱 Access the chatbot interface at: http://localhost:{port}")
    print("🔍 API endpoints:")
    print("   - POST /api/predict - Get predictions from both models")
    print("   - GET /api/model_info - Get model information")
    print("   - GET /health - Health check")
    print(f"\n💡 To use a different port, set FLASK_PORT environment variable")
    print(f"   Example: FLASK_PORT=8080 python app.py\n")
    
    app.run(debug=True, host=host, port=port)

