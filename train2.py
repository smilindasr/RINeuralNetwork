import sys
import os
import glob
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
from model2 import prepare_data, PolicyValueNetwork

def get_latest_model():
    """Find the most recent model file based on timestamp in filename."""
    model_files = glob.glob('model2_*.pth')
    if not model_files:
        return None
    latest_model = max(model_files, key=lambda x: os.path.getctime(x))
    return latest_model

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

def calculate_policy_determinism(policy):
    """Calculate how deterministic a policy is (0 = uniform, 1 = deterministic)."""
    policy = np.array(policy)
    max_prob = np.max(policy)
    return float(max_prob)

def train_step(model, optimizer, policy_loss_fn, board, player, target_policy, target_value):
    """Perform a single training step and return the losses."""
    optimizer.zero_grad()

    # Forward pass
    policy_pred, value_pred = model(board, player)

    # Calculate losses
    policy_loss = policy_loss_fn(policy_pred.log(), target_policy)
    value_loss = nn.MSELoss()(value_pred, target_value)

    # Combined loss (you can adjust the weights)
    total_loss = policy_loss + value_loss

    # Calculate determinism
    determinism = calculate_policy_determinism(target_policy.numpy())

    # Print debugging info more frequently for highly deterministic cases
    if determinism > 0.3 and torch.rand(1).item() < 0.1:  # Print 10% of deterministic cases
        print(f"\n{'='*60}")
        print(f"Training Step Info (Determinism: {determinism:.2f})")
        print(f"Current Player: {'X' if player.item() == 1 else 'O'}")
        print(f"\nBoard State:")
        print(format_board(board.view(3, 3).numpy()))
        print(f"\nLosses:")
        print(f"  Policy Loss: {policy_loss.item():.4f}")
        print(f"  Value Loss: {value_loss.item():.4f}")
        print(f"  Total Loss: {total_loss.item():.4f}")
        print(f"Predicted Value: {value_pred.item():.3f}")
        print(f"Target Value: {target_value.item():.3f}")

        print("\nPredicted Moves:")
        pred_moves = format_move_probs(board.view(3, 3).numpy(), policy_pred.squeeze().detach().numpy())
        for move in pred_moves:
            print(f"  {move}")

        print("\nTarget Moves:")
        target_moves = format_move_probs(board.view(3, 3).numpy(), target_policy.squeeze().detach().numpy())
        for move in target_moves:
            print(f"  {move}")
        print(f"{'='*60}\n")

    # Backward pass
    total_loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item(), total_loss.item()

def load_training_data_from_jsonl(jsonl_path):
    """Load training data from a JSONL file."""
    try:
        training_data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    # Validate required fields
                    required_fields = ['board', 'current_player', 'policy', 'value']
                    missing_fields = [field for field in required_fields if field not in example]
                    if missing_fields:
                        print(f"Warning: Skipping example due to missing fields: {missing_fields}")
                        continue

                    # Validate data types and shapes
                    if not isinstance(example['board'], list) or len(example['board']) != 3 or \
                       not all(len(row) == 3 for row in example['board']):
                        print("Warning: Skipping example due to invalid board shape")
                        continue

                    if not isinstance(example['policy'], list) or len(example['policy']) != 9:
                        print("Warning: Skipping example due to invalid policy shape")
                        continue

                    if not isinstance(example['current_player'], (int, float)) or \
                       example['current_player'] not in [1, -1]:
                        print("Warning: Skipping example due to invalid current_player value")
                        continue

                    if not isinstance(example['value'], (int, float)) or \
                       not -1 <= example['value'] <= 1:
                        print("Warning: Skipping example due to invalid value")
                        continue

                    training_data.append(example)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                    continue

        if not training_data:
            raise ValueError("No valid training examples found in the file")

        print(f"Successfully loaded {len(training_data)} training examples from {jsonl_path}")
        return training_data

    except Exception as e:
        print(f"Error loading training data from JSONL: {str(e)}")
        return None

def train_model(training_data, epochs=100, lr=0.001):
    # Initialize model
    model = PolicyValueNetwork()

    # Load latest model if exists
    latest_model_path = get_latest_model()
    if latest_model_path:
        print(f"Loading existing model from {latest_model_path}")
        model.load_state_dict(torch.load(latest_model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')

    # Prepare data
    boards, players, policies, values = prepare_data(training_data)

    try:
        running_loss = 0.0
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        min_loss_improvement = 0.001

        for epoch in range(epochs):
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_loss = 0.0
            num_samples = max(1, int(0.05 * len(boards)))  # Sample 5% of data
            sample_indices = np.random.choice(len(boards), num_samples, replace=False)

            # Train on sampled data
            for i in sample_indices:
                policy_loss, value_loss, loss = train_step(
                    model,
                    optimizer,
                    policy_loss_fn,
                    boards[i].unsqueeze(0),  # Add batch dimension
                    players[i].unsqueeze(0),
                    policies[i].unsqueeze(0),
                    values[i].unsqueeze(0)
                )
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_loss += loss

            # Calculate average losses
            avg_policy_loss = total_policy_loss / num_samples
            avg_value_loss = total_value_loss / num_samples
            avg_loss = total_loss / num_samples
            running_loss = 0.9 * running_loss + 0.1 * avg_loss

            # Early stopping check with minimum improvement threshold
            if running_loss < best_loss - min_loss_improvement:
                best_loss = running_loss
                patience_counter = 0
                # Save best model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f'model2_{timestamp}_best.pth'
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model with loss {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

            if (epoch + 1) % 10 == 0:  # Print progress every 10 epochs
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"Average Policy Loss: {avg_policy_loss:.4f}")
                print(f"Average Value Loss: {avg_value_loss:.4f}")
                print(f"Average Total Loss: {avg_loss:.4f}")
                print(f"Running Loss: {running_loss:.4f}")
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 40)

    finally:
        # Always save final model state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_save_path = f'model2_{timestamp}_final.pth'
        torch.save(model.state_dict(), final_save_path)
        print(f"Saved final model to {final_save_path}")

    return model

if __name__ == "__main__":

    jsonl_path = "training_data.jsonl"
    training_data = load_training_data_from_jsonl(jsonl_path)
    if training_data is None:
        print("Failed to load training data from JSONL. Exiting.")
        sys.exit(1)

    train_model(training_data)
