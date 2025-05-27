# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F 
import pandas as pd
import numpy as np
from io import StringIO
import datetime 
import logging 

app = Flask(__name__)
CORS(app)


app.logger.setLevel(logging.INFO) # Set level to INFO


INPUT_SIZE = 12
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 2
DROPOUT_RATE = 0.3
MODEL_STATE_DICT_PATH = 'models/ecg_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app.logger.info(f"--- Using device: {device} ---")

pytorch_model = None 
try:
    from model_definition import StackedLSTM_MultiPooling 
    pytorch_model = StackedLSTM_MultiPooling(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE
    )
    pytorch_model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location=device))
    pytorch_model.to(device)
    pytorch_model.eval()
    app.logger.info(f"--- PyTorch model loaded successfully from {MODEL_STATE_DICT_PATH} ---")
except FileNotFoundError:
    app.logger.error(f"--- FATAL ERROR: Model file not found at {MODEL_STATE_DICT_PATH}. Application will not work. ---")
    
except ImportError:
    app.logger.error(f"--- FATAL ERROR: Could not import model definition (e.g., StackedLSTM_MultiPooling from model_definition.py). ---")
except Exception as e:
    app.logger.error(f"--- FATAL ERROR loading PyTorch model: {e} ---", exc_info=True)
    
def preprocess_ecg_data_from_string(csv_data_string):
    try:
        data_io = StringIO(csv_data_string)
        df = pd.read_csv(data_io, header=0, usecols=range(12))
    except pd.errors.EmptyDataError:
        raise ValueError("No data: The CSV string is empty.")
    except Exception as e:
        raise ValueError(f"Error reading CSV from string: {str(e)}")

    if df.shape != (5000, 12):
        raise ValueError(f"Data must be 5000×12 after reading, got {df.shape}")

    arr = df.values.astype(np.float32)
    arr_for_norm = arr.T 
    arr_for_norm = np.expand_dims(arr_for_norm, axis=0) 
    mean = arr_for_norm.mean(axis=(0, 2), keepdims=True)
    std  = arr_for_norm.std (axis=(0, 2), keepdims=True)
    std_safe = np.where(std == 0, 1e-8, std) 
    if np.any(std == 0):
        app.logger.warning("Warning: Zero standard deviation found in one or more channels during preprocessing. Using 1e-8 for those.")
    normalized_arr = (arr_for_norm - mean) / std_safe
    final_arr = normalized_arr.transpose(0, 2, 1)
    return final_arr

# --- API Routes ---
@app.route('/')
def home():
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    app.logger.info(f"--- INFO: Home route / accessed at {timestamp} ---")
    return "PyTorch ECG Atrial Fibrillation Detection API is running!", 200

# --- ADDED DEDICATED HEALTH CHECK ENDPOINT ---
@app.route('/healthz')
def healthz():
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    app.logger.info(f"--- INFO: Health check route /healthz accessed at {timestamp} ---")
    if pytorch_model is not None:
        return jsonify(status="healthy", model_loaded=True, timestamp=timestamp), 200
    else:
        app.logger.error(f"--- ERROR: Health check /healthz called but model is not loaded! ---")
        return jsonify(status="unhealthy", model_loaded=False, message="Model not loaded", timestamp=timestamp), 503 

@app.route('/predict', methods=['POST'])
def predict():
    if pytorch_model is None:
        app.logger.error("--- ERROR: Predict called but model is not loaded. ---")
        return jsonify({'error': 'Model not loaded. Backend issue.'}), 503 

    if 'ecg_file' not in request.files:
        return jsonify({'error': 'No file part in the request. Send "ecg_file".'}), 400

    file = request.files['ecg_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file and file.filename.lower().endswith('.csv'):
        try:
            csv_data_string = file.read().decode('utf-8')
            processed_data_np = preprocess_ecg_data_from_string(csv_data_string)
            input_tensor = torch.from_numpy(processed_data_np).float().to(device)

            with torch.no_grad():
                output_logits = pytorch_model(input_tensor)
            
            probabilities = F.softmax(output_logits, dim=1)
            afib_probability = float(probabilities[0, 1].item()) 

            threshold = 0.5
            if afib_probability >= threshold:
                prediction_label = "Atrial Fibrillation Potentially Detected"
            else:
                prediction_label = "Normal Sinus Rhythm Likely"

            return jsonify({
                'prediction_label': prediction_label,
                'probability_afib': afib_probability,
                'threshold_used': threshold,
                'all_class_probabilities': probabilities.cpu().numpy().tolist()[0]
            })

        except ValueError as ve:
            app.logger.warning(f"--- WARN: ValueError during prediction: {str(ve)} ---") # Client error
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            app.logger.error(f"--- ERROR: Unexpected error during prediction: {e} ---", exc_info=True) # Server error
            return jsonify({'error': 'An unexpected server error occurred.'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a .csv file.'}), 400

if __name__ == '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
                                               
    if not app.debug: 
        logging.basicConfig(level=logging.INFO)

    app.run(debug=True, host='0.0.0.0', port=5001)