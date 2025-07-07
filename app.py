from flask import Flask, request, jsonify
import torch
from predict2 import predict_move, PolicyValueNetwork, get_latest_model
import numpy as np

app = Flask(__name__)

# Load the model at startup
model_path = get_latest_model()
if not model_path:
    raise RuntimeError("No trained model found!")

model = PolicyValueNetwork()
model.load_state_dict(torch.load(model_path))

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        add_noise = data.get('add_noise', False)
        noise_alpha = data.get('noise_alpha', 0.3)
        noise_epsilon = data.get('noise_epsilon', 0.25)

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

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
