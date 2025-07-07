import sys
import json
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F

def check_policy_validity(policy):
    """Check if policy is valid (sums to 1 and non-negative)."""
    policy_sum = sum(policy)
    is_non_negative = all(p >= 0 for p in policy)
    return {
        'sum': policy_sum,
        'is_valid_sum': abs(policy_sum - 1.0) < 1e-6,
        'is_non_negative': is_non_negative,
        'min_value': min(policy),
        'max_value': max(policy)
    }

def check_policy_alignment(board, policy):
    """Check if policy aligns with valid moves."""
    board_flat = np.array(board).flatten()
    valid_moves = (board_flat == 0)
    policy_array = np.array(policy)

    # Check if any probability is assigned to invalid moves
    invalid_move_probs = policy_array[~valid_moves]
    valid_move_probs = policy_array[valid_moves]

    return {
        'invalid_moves_total_prob': float(invalid_move_probs.sum()),
        'valid_moves_total_prob': float(valid_move_probs.sum()),
        'num_valid_moves': int(valid_moves.sum()),
        'num_invalid_moves': int((~valid_moves).sum()),
        'invalid_moves_with_prob': int((invalid_move_probs > 0).sum())
    }

def calculate_entropy(policy):
    """Calculate the entropy of the policy distribution."""
    policy = np.array(policy)
    # Add small epsilon to avoid log(0)
    log_policy = np.log(policy + 1e-10)
    entropy = -np.sum(policy * log_policy)
    return float(entropy)

def analyze_position_coverage(training_data):
    """Analyze the coverage of different game states."""
    position_counts = defaultdict(int)
    num_pieces_dist = defaultdict(int)

    for data in training_data:
        board = np.array(data['board'])
        # Convert board to string for counting unique positions
        board_str = str(board.tolist())
        position_counts[board_str] += 1

        # Count number of pieces
        num_x = np.sum(board == 1)
        num_o = np.sum(board == -1)
        num_pieces_dist[(num_x, num_o)] += 1

    return {
        'unique_positions': len(position_counts),
        'total_positions': len(training_data),
        'position_frequency': dict(position_counts),
        'num_pieces_distribution': dict(num_pieces_dist)
    }

def find_similar_positions(training_data, similarity_threshold=0.8):
    """Find similar positions and check policy consistency."""
    similar_positions = []
    n = len(training_data)

    for i in range(n):
        board1 = np.array(data['board'] for data in training_data[i])
        policy1 = np.array(data['policy'] for data in training_data[i])

        for j in range(i + 1, n):
            board2 = np.array(data['board'] for data in training_data[j])
            policy2 = np.array(data['policy'] for data in training_data[j])

            # Check board similarity
            board_similarity = np.mean(board1 == board2)
            if board_similarity > similarity_threshold:
                # Calculate policy similarity using cosine similarity
                policy_similarity = F.cosine_similarity(
                    torch.tensor(policy1).unsqueeze(0),
                    torch.tensor(policy2).unsqueeze(0)
                ).item()

                similar_positions.append({
                    'position1_idx': i,
                    'position2_idx': j,
                    'board_similarity': float(board_similarity),
                    'policy_similarity': float(policy_similarity)
                })

    return similar_positions

def analyze_training_data(data):
    """Analyze training data quality and generate comprehensive report."""
    results = {
        'total_samples': len(data),
        'policy_validity': [],
        'policy_alignment': [],
        'policy_entropy': [],
        'position_coverage': None,
        'similar_positions': None,
        'issues_found': []
    }

    # Analyze each training sample
    for idx, sample in enumerate(data):
        # Check policy validity
        policy_valid = check_policy_validity(sample['policy'])
        results['policy_validity'].append(policy_valid)

        if not policy_valid['is_valid_sum']:
            results['issues_found'].append(
                f"Sample {idx}: Policy sum is {policy_valid['sum']:.6f} (should be 1.0)"
            )
        if not policy_valid['is_non_negative']:
            results['issues_found'].append(
                f"Sample {idx}: Policy contains negative values"
            )

        # Check policy alignment with valid moves
        alignment = check_policy_alignment(sample['board'], sample['policy'])
        results['policy_alignment'].append(alignment)

        if alignment['invalid_moves_with_prob'] > 0:
            results['issues_found'].append(
                f"Sample {idx}: Policy assigns probability to {alignment['invalid_moves_with_prob']} invalid moves"
            )

        # Calculate policy entropy
        entropy = calculate_entropy(sample['policy'])
        results['policy_entropy'].append(entropy)

        if entropy < 0.1:
            results['issues_found'].append(
                f"Sample {idx}: Very low policy entropy ({entropy:.3f}), might be too deterministic"
            )
        elif entropy > 2.0:
            results['issues_found'].append(
                f"Sample {idx}: Very high policy entropy ({entropy:.3f}), might be too uniform"
            )

    # Analyze position coverage
    results['position_coverage'] = analyze_position_coverage(data)

    if results['position_coverage']['unique_positions'] < len(data):
        results['issues_found'].append(
            f"Dataset contains duplicate positions: {results['position_coverage']['unique_positions']} unique vs {len(data)} total"
        )

    # Find similar positions and check policy consistency
    results['similar_positions'] = find_similar_positions(data)

    # Add summary statistics
    results['summary'] = {
        'avg_entropy': np.mean(results['policy_entropy']),
        'avg_invalid_moves_prob': np.mean([
            a['invalid_moves_total_prob'] for a in results['policy_alignment']
        ]),
        'position_coverage_ratio': results['position_coverage']['unique_positions'] / len(data)
    }

    return results

def print_analysis_report(results):
    """Print a human-readable analysis report."""
    print("\n=== Training Data Analysis Report ===")
    print(f"\nTotal Samples: {results['total_samples']}")

    print("\nSummary Statistics:")
    print(f"- Average Policy Entropy: {results['summary']['avg_entropy']:.3f}")
    print(f"- Average Invalid Moves Probability: {results['summary']['avg_invalid_moves_prob']:.3f}")
    print(f"- Position Coverage Ratio: {results['summary']['position_coverage_ratio']:.3f}")

    print("\nPosition Coverage:")
    coverage = results['position_coverage']
    print(f"- Unique Positions: {coverage['unique_positions']}")
    print(f"- Total Positions: {coverage['total_positions']}")

    print("\nPiece Distribution:")
    for (num_x, num_o), count in coverage['num_pieces_distribution'].items():
        print(f"- X:{num_x}, O:{num_o} -> {count} positions")

    if results['similar_positions']:
        print("\nSimilar Positions Found:")
        for sp in results['similar_positions']:
            print(f"- Positions {sp['position1_idx']} and {sp['position2_idx']}:")
            print(f"  Board Similarity: {sp['board_similarity']:.3f}")
            print(f"  Policy Similarity: {sp['policy_similarity']:.3f}")

    if results['issues_found']:
        print("\nIssues Found:")
        for issue in results['issues_found']:
            print(f"- {issue}")
    else:
        print("\nNo major issues found in the training data.")

if __name__ == "__main__":
    # Read training data from command line argument
    if len(sys.argv) > 1:
        training_data = json.loads(sys.argv[1])
    else:
        # Example training data
        training_data = [
            {
                "board": [[0, 1, -1],
                          [0, 1, 0],
                          [-1, 0, 0]],
                "policy": [0.1, 0.2, 0, 0, 0, 0.3, 0, 0.2, 0.2]
            },
            {
                "board": [[1, 1, -1],
                          [0, -1, 0],
                          [-1, 0, 0]],
                "policy": [0, 0, 0, 0.4, 0, 0.3, 0, 0.2, 0.1]
            }
        ]

    # Analyze the data
    analysis_results = analyze_training_data(training_data)

    # Print the report
    print_analysis_report(analysis_results)

    # Also output JSON results if needed
    if '--json' in sys.argv:
        print("\nJSON Results:")
        print(json.dumps(analysis_results, indent=2))
