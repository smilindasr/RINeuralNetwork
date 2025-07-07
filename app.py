from flask import Flask, request, jsonify
import torch
from predict2 import predict_move, PolicyValueNetwork, get_latest_model
from train2 import train_model, load_training_data_from_jsonl
import numpy as np

app = Flask(__name__)
model = PolicyValueNetwork()
# Load the model at startup
model_path = get_latest_model()
if model_path:
    model.load_state_dict(torch.load(model_path))
    print("Model loaded", model_path)
else:
    print("No pretrained model found. Loading a default model.")

@app.route('/predict', methods=['POST'])
def predict():

        data = request.get_json()

        # Validate input
        if not data or 'board' not in data or 'current_player' not in data:
            return jsonify({
                'error': 'Invalid input. Required fields: board (3x3 array), current_player (1 or -1)'
            }), 400

        board = data['board']
        current_player = data['current_player']

        # Optional parameters
        temperature = data.get('temperature', 1.0)
        add_noise = data.get('add_noise', True)
        noise_alpha = data.get('noise_alpha', 0.3)
        noise_epsilon = data.get('noise_epsilon', 0.3)

        # Make prediction
        policy, value = predict_move(
            model,
            board,
            current_player,
            temperature=temperature,
            add_noise=add_noise,
            noise_alpha=noise_alpha,
            noise_epsilon=noise_epsilon
        )

        return jsonify({
            'policy': policy.tolist(),
            'value': float(value)
        })

    #except Exception as e:
    #    return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()

        # Optional training parameters
        epochs = 50
        learning_rate = .001

        # Train the model
        try:
            train_model(data, epochs=epochs, lr=learning_rate)


            return jsonify({
                'message': 'Model training completed successfully',
                'epochs_completed': epochs
            })

        except Exception as e:
            return jsonify({'error': f'Training failed: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load-model', methods=['POST'])
def load_model():
    try:
        # Get the latest model path
        model_path = get_latest_model()

        if not model_path:
            return jsonify({
                'error': 'No model files found'
            }), 404

        model.load_state_dict(torch.load(model_path))

        return jsonify({
            'message': 'Model loaded successfully',
            'model_path': model_path
        })

    except Exception as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
