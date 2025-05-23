# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F # For softmax
import pandas as pd
import numpy as np
from io import StringIO

# --- Import your model definition ---
from model_definition import StackedLSTM_MultiPooling # This should match your class name

app = Flask(__name__)
CORS(app)

# --- Model Loading ---
# Parameters for your StackedLSTM_MultiPooling model
INPUT_SIZE = 12
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 2 # Your model outputs logits for 2 classes
DROPOUT_RATE = 0.3 # Default from your class, explicitly state it for clarity

MODEL_STATE_DICT_PATH = 'models/ecg_model.pth' # YOUR MODEL PATH
# Ensure the 'models' directory exists in afib_pytorch_backend, or adjust path
# If 'ecg_model.pth' is directly in afib_pytorch_backend, then path is 'ecg_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Using device: {device} ---")

# 1. Instantiate your model structure
pytorch_model = StackedLSTM_MultiPooling(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT_RATE
)

# 2. Load the learned weights (state_dict)
try:
    # If your .pth file IS the state_dict:
    pytorch_model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location=device))
    
    # If your .pth file IS THE ENTIRE MODEL (less common for .pth but possible if saved with torch.save(model, PATH)):
    # pytorch_model = torch.load(MODEL_STATE_DICT_PATH, map_location=device)
    # Make sure `pytorch_model` is an instance of StackedLSTM_MultiPooling after this line if you use this option.

    pytorch_model.to(device)
    pytorch_model.eval()
    print(f"--- PyTorch model loaded successfully from {MODEL_STATE_DICT_PATH} ---")
except FileNotFoundError:
    print(f"--- ERROR: Model file not found at {MODEL_STATE_DICT_PATH}. Please check the path. ---")
    pytorch_model = None
except Exception as e:
    print(f"--- Error loading PyTorch model: {e} ---")
    pytorch_model = None

# --- Your Preprocessing Function (copied from your example) ---
def preprocess_ecg_data_from_string(csv_data_string):
    """
    Load & preprocess ECG CSV string (5000×12) exactly as in training:
      • Normalize per-channel: mean/std over (batch, timesteps)
      • Return np.ndarray shape (1, 5000, 12), dtype float32
    """
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
        print("Warning: Zero standard deviation found in one or more channels. Using 1e-8 for those.")
        
    normalized_arr = (arr_for_norm - mean) / std_safe
    final_arr = normalized_arr.transpose(0, 2, 1) # (1, 5000, 12)

    return final_arr


# --- API Routes ---
@app.route('/')
def home():
    return "PyTorch ECG Atrial Fibrillation Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    global pytorch_model
    if pytorch_model is None:
        return jsonify({'error': 'Model not loaded. Backend issue.'}), 500

    if 'ecg_file' not in request.files:
        return jsonify({'error': 'No file part in the request. Send "ecg_file".'}), 400

    file = request.files['ecg_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file and file.filename.lower().endswith('.csv'):
        try:
            csv_data_string = file.read().decode('utf-8')
            processed_data_np = preprocess_ecg_data_from_string(csv_data_string) # (1, 5000, 12)
            
            input_tensor = torch.from_numpy(processed_data_np).float().to(device) # Ensure float and move to device
            # Your snippet had: if tensor.ndim == 2: tensor = tensor.unsqueeze(0)
            # Our preprocessor already makes it (1, 5000, 12), so ndim will be 3. This check isn't strictly needed here
            # but doesn't hurt if kept. For now, assuming preprocessor is robust.

            with torch.no_grad():
                output_logits = pytorch_model(input_tensor) # Shape: (1, num_classes) e.g. (1,2)

            # Apply softmax to get probabilities
            probabilities = F.softmax(output_logits, dim=1) # Shape: (1, num_classes)

            # Get the probability of the "Atrial Fibrillation" class
            # IMPORTANT: Confirm that index 1 is indeed your AFib class.
            # If index 0 is AFib and index 1 is Normal, use probs[0, 0].
            afib_probability = float(probabilities[0, 1].item())

            threshold = 0.5
            # Your snippet's diagnosis:
            # diagnosis = "You may have atrial fibrillation" if af_prob > 0.5 else "Your ECG report looks fine"
            if afib_probability >= threshold:
                prediction_label = "Atrial Fibrillation Potentially Detected" # Or your preferred wording
            else:
                prediction_label = "Normal Sinus Rhythm Likely" # Or your preferred wording

            return jsonify({
                'prediction_label': prediction_label,
                'probability_afib': afib_probability, # Probability of the positive (AFib) class
                'threshold_used': threshold,
                'all_class_probabilities': probabilities.cpu().numpy().tolist()[0] # Optional: send all probs
            })

        except ValueError as ve:
            app.logger.error(f"ValueError: {str(ve)}")
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            app.logger.error(f"Prediction Error: {e}", exc_info=True)
            return jsonify({'error': f'Server error during prediction: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a .csv file.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)