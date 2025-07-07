import sys
import json
import numpy as np
import torch
from model import TicTacToeNet
from collections import defaultdict
import math

from model2 import PolicyNetwork


def count_parameters(model):
    """Count trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count parameters by layer type
    params_by_type = defaultdict(int)
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Determine layer type from parameter name
            if 'conv' in name:
                layer_type = 'conv'
            elif 'linear' in name:
                layer_type = 'linear'
            elif 'bn' in name or 'batch_norm' in name:
                layer_type = 'batch_norm'
            else:
                layer_type = 'other'
            params_by_type[layer_type] += param.numel()

    return total_params, dict(params_by_type)

def estimate_model_capacity(model):
    """Estimate model's capacity and complexity metrics."""
    total_params, params_by_type = count_parameters(model)

    # Calculate VC dimension approximation
    # For neural networks, VC dimension is roughly O(W log W) where W is the number of parameters
    vc_dimension = total_params * math.log2(total_params)

    # Calculate number of effective parameters (assuming batch norm reduces effective params)
    effective_params = total_params - params_by_type.get('batch_norm', 0) * 0.5

    return {
        'total_parameters': total_params,
        'parameters_by_type': params_by_type,
        'vc_dimension': vc_dimension,
        'effective_parameters': effective_params
    }

def analyze_data_complexity(training_data):
    """Analyze complexity and diversity of training data."""
    num_samples = len(training_data)

    # Count unique board positions
    unique_boards = set(str(data['board']) for data in training_data)
    num_unique_boards = len(unique_boards)

    # Calculate average number of pieces per position
    avg_pieces = np.mean([
        sum(row.count(1) + row.count(-1) for row in data['board'])
        for data in training_data
    ])

    # Calculate policy complexity (average entropy)
    avg_entropy = np.mean([
        -sum(p * np.log2(p + 1e-10) for p in data['policy'])
        for data in training_data
    ])

    return {
        'num_samples': num_samples,
        'num_unique_boards': num_unique_boards,
        'average_pieces_per_position': float(avg_pieces),
        'average_policy_entropy': float(avg_entropy),
        'data_complexity_score': float(num_unique_boards * avg_entropy)
    }

def estimate_minimum_samples(model_capacity):
    """Estimate minimum number of training samples needed based on model capacity."""
    # Rule of thumb: need at least 10 samples per parameter for good generalization
    min_samples_params = model_capacity['total_parameters'] * 10

    # Alternative estimate based on VC dimension
    min_samples_vc = model_capacity['vc_dimension'] * 2

    return max(min_samples_params, min_samples_vc)

def calculate_complexity_metrics(model_capacity, data_complexity):
    """Calculate metrics comparing model capacity to data complexity."""
    min_samples = estimate_minimum_samples(model_capacity)
    actual_samples = data_complexity['num_samples']

    # Calculate various metrics
    params_per_sample = model_capacity['total_parameters'] / actual_samples
    samples_ratio = actual_samples / min_samples
    complexity_ratio = model_capacity['effective_parameters'] / data_complexity['data_complexity_score']

    return {
        'parameters_per_sample': float(params_per_sample),
        'samples_ratio': float(samples_ratio),
        'complexity_ratio': float(complexity_ratio),
        'minimum_recommended_samples': int(min_samples)
    }

def analyze_model_fit(model_capacity, data_complexity, metrics):
    """Analyze if model size is appropriate for the data."""
    issues = []
    warnings = []
    recommendations = []

    # Check number of samples
    if metrics['samples_ratio'] < 0.2:
        issues.append(
            f"Severe underfit risk: Only {metrics['samples_ratio']:.1%} of recommended samples. "
            f"Need at least {metrics['minimum_recommended_samples']} samples."
        )
    elif metrics['samples_ratio'] < 0.5:
        warnings.append(
            f"Potential underfit: Have {metrics['samples_ratio']:.1%} of recommended samples. "
            f"Consider getting more training data."
        )

    # Check parameters per sample
    if metrics['parameters_per_sample'] > 1000:
        issues.append(
            f"Severe overfit risk: {metrics['parameters_per_sample']:.0f} parameters per sample. "
            "Model is likely too complex for the data."
        )
    elif metrics['parameters_per_sample'] > 100:
        warnings.append(
            f"Potential overfit: {metrics['parameters_per_sample']:.0f} parameters per sample. "
            "Consider simplifying the model or getting more data."
        )

    # Check complexity ratio
    if metrics['complexity_ratio'] > 10:
        warnings.append(
            f"Model might be too complex: {metrics['complexity_ratio']:.1f}x more complex than data. "
            "Consider simplifying the model architecture."
        )

    # Generate recommendations
    if metrics['samples_ratio'] < 1.0:
        target_samples = int(metrics['minimum_recommended_samples'] * 1.2)  # 20% buffer
        recommendations.append(
            f"Recommend increasing dataset size to at least {target_samples} samples "
            f"(currently have {data_complexity['num_samples']})."
        )

    if metrics['parameters_per_sample'] > 100:
        recommendations.append(
            "Consider simplifying the model by:"
            "\n- Reducing number of convolutional filters"
            "\n- Reducing number of linear layer units"
            "\n- Removing layers"
        )

    return {
        'issues': issues,
        'warnings': warnings,
        'recommendations': recommendations
    }

def print_analysis_report(model_capacity, data_complexity, metrics, analysis):
    """Print a human-readable analysis report."""
    print("\n=== Model Complexity Analysis Report ===")

    print("\nModel Architecture:")
    print(f"- Total Parameters: {model_capacity['total_parameters']:,}")
    print("- Parameters by Type:")
    for layer_type, count in model_capacity['parameters_by_type'].items():
        print(f"  • {layer_type}: {count:,}")
    print(f"- Effective Parameters: {model_capacity['effective_parameters']:,}")

    print("\nTraining Data:")
    print(f"- Total Samples: {data_complexity['num_samples']:,}")
    print(f"- Unique Board Positions: {data_complexity['num_unique_boards']:,}")
    print(f"- Average Pieces per Position: {data_complexity['average_pieces_per_position']:.2f}")
    print(f"- Average Policy Entropy: {data_complexity['average_policy_entropy']:.2f}")

    print("\nComplexity Metrics:")
    print(f"- Parameters per Sample: {metrics['parameters_per_sample']:.1f}")
    print(f"- Samples Ratio: {metrics['samples_ratio']:.1%}")
    print(f"- Minimum Recommended Samples: {metrics['minimum_recommended_samples']:,}")

    if analysis['issues']:
        print("\n⚠️ Critical Issues:")
        for issue in analysis['issues']:
            print(f"- {issue}")

    if analysis['warnings']:
        print("\n⚠️ Warnings:")
        for warning in analysis['warnings']:
            print(f"- {warning}")

    if analysis['recommendations']:
        print("\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"- {rec}")

    if not (analysis['issues'] or analysis['warnings']):
        print("\n✅ Model complexity appears appropriate for the dataset size.")

if __name__ == "__main__":
    # Create model instance
    # model = TicTacToeNet()
    model = PolicyNetwork()
    # Load training data
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

    # Analyze model and data
    model_capacity = estimate_model_capacity(model)
    data_complexity = analyze_data_complexity(training_data)
    metrics = calculate_complexity_metrics(model_capacity, data_complexity)
    analysis = analyze_model_fit(model_capacity, data_complexity, metrics)

    # Print report
    print_analysis_report(model_capacity, data_complexity, metrics, analysis)

    # Output JSON if requested
    if '--json' in sys.argv:
        results = {
            'model_capacity': model_capacity,
            'data_complexity': data_complexity,
            'metrics': metrics,
            'analysis': analysis
        }
        print("\nJSON Results:")
        print(json.dumps(results, indent=2))
