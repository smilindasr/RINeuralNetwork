import torch
import numpy as np
from model2 import PolicyValueNetwork
import glob
import os
import json
import sys

def get_latest_model():
    """Find the most recent model file based on timestamp in filename."""
    model_files = glob.glob('model2_*.pth')
    if not model_files:
        return None
    latest_model = max(model_files, key=lambda x: os.path.getctime(x))
    return latest_model

def add_dirichlet_noise(policy, board_state, alpha=0.3, epsilon=0.1):
    """
    Add Dirichlet noise to the policy for move exploration.

    Args:
        policy: Original move probabilities
        board_state: Current board state to mask invalid moves
        alpha: Dirichlet distribution parameter (lower means more concentrated)
        epsilon: Weight of noise (0 = no noise, 1 = only noise)
    """
    # Create mask for valid moves
    valid_moves = (np.array(board_state).flatten() == 0)
    num_valid_moves = valid_moves.sum()

    if num_valid_moves == 0:
        return policy

    # Generate Dirichlet noise only for valid moves
    noise = np.zeros(9)
    noise[valid_moves] = np.random.dirichlet([alpha] * num_valid_moves)

    # Combine original policy with noise
    noisy_policy = (1 - epsilon) * policy + epsilon * noise

    # Mask invalid moves and renormalize
    noisy_policy = noisy_policy * valid_moves
    if noisy_policy.sum() > 0:
        noisy_policy /= noisy_policy.sum()

    return noisy_policy

def format_board(board):
    """Format the board state in a readable way."""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    return '\n'.join([' '.join(symbols[cell] for cell in row) for row in board])

def format_move_probs(board_state, policy, threshold=0.01):
    """Format move probabilities in a readable way, showing board positions."""
    board = np.array(board_state)
    policy = np.array(policy)

    # Create a list of moves with their probabilities
    moves = []
    for i in range(9):
        if policy[i] >= threshold:
            row, col = i // 3, i % 3
            moves.append((i, row, col, policy[i]))

    # Sort by probability
    moves.sort(key=lambda x: x[3], reverse=True)

    # Format the output
    result = []
    for move_idx, row, col, prob in moves:
        cell_content = board[row][col]
        cell_str = '.' if cell_content == 0 else 'X' if cell_content == 1 else 'O'
        result.append(f"Move ({row},{col}) [{cell_str}]: {prob:.3f}")

    return result

def predict_move(model, board_state, current_player, temperature=1.0, add_noise=False,
                noise_alpha=0.3, noise_epsilon=0.25):
    """
    Predict the next move using the policy network.

    Args:
        model: The trained PolicyValueNetwork model
        board_state: Current board state (3x3 list/array)
        current_player: Current player (1 for X, -1 for O)
        temperature: Temperature for move selection (higher = more exploratory)
        add_noise: Whether to add Dirichlet noise for exploration
        noise_alpha: Dirichlet distribution parameter
        noise_epsilon: Weight of noise
        return_value: Whether to return the predicted value along with the move

    Returns:
        If return_value is False:
            (move_row, move_col), policy
        If return_value is True:
            (move_row, move_col), policy, value
    """
    model.eval()
    with torch.no_grad():
        # Prepare input
        board = torch.tensor(
            [cell for row in board_state for cell in row],
            dtype=torch.float32
        ).unsqueeze(0)
        player = torch.tensor([current_player], dtype=torch.float32).unsqueeze(0)

        # Get predictions
        policy, value = model(board, player)

        # Convert to numpy
        policy = policy.squeeze().numpy()
        value = value.squeeze().item()

        # Add Dirichlet noise if requested
        if add_noise:
            policy = add_dirichlet_noise(
                policy,
                board_state,
                alpha=noise_alpha,
                epsilon=noise_epsilon
            )

        # Apply temperature
        if temperature != 1.0:
            policy = np.power(policy, 1.0 / temperature)
            policy /= policy.sum()

        # Mask invalid moves
        valid_moves = (np.array(board_state).flatten() == 0)
        policy = policy * valid_moves
        if policy.sum() > 0:
            policy /= policy.sum()

        return policy, value

def print_prediction_details(board_state, current_player, policy, value=None,
                           temperature=1.0, noise_params=None):
    """Print detailed prediction information."""
    print(f"\n{'='*60}")
    print(f"Prediction Details:")
    print(f"Current Player: {'X' if current_player == 1 else 'O'}")
    print(f"Temperature: {temperature}")
    if noise_params:
        print(f"Dirichlet Noise: α={noise_params['alpha']}, ε={noise_params['epsilon']}")

    print(f"\nBoard State:")
    print(format_board(board_state))

    if value is not None:
        print(f"\nPosition Evaluation: {value:.3f}")
        print(f"{'X' if value > 0 else 'O'} is {'winning' if abs(value) > 0.7 else 'ahead'}"
              if abs(value) > 0.3 else "Position is roughly equal")

    print("\nMove Probabilities:")
    moves = format_move_probs(board_state, policy)
    for move in moves:
        print(f"  {move}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Load the model
    model_path = get_latest_model()
    if not model_path:
        #print("No trained model found!")
        exit(1)
    input_data = json.loads(sys.argv[1])
    #print(input_data)
    #print(f"Loading model from {model_path}")
    model = PolicyValueNetwork()
    model.load_state_dict(torch.load(model_path))

    # Example board state
    board_state = [
        [0, 1, -1],
        [0, 1, 0],
        [-1, 0, 0]
    ]
    current_player = 1  # X's turn

    # Get prediction with different settings
    #print("\nPrediction with default settings (T=1.0, with noise):")
    policy, value = predict_move(
        model,
        input_data['board'],
        input_data['current_player']
    )

    print_prediction_details(
        board_state,
        current_player,
        policy,
        value,
        noise_params={'alpha': 0.3, 'epsilon': 0.25}
    )

    print(f"Policy${policy.tolist()}$,Value${value}$")
