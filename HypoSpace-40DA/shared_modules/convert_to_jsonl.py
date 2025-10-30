#!/usr/bin/env python3
"""
Universal converter for HypoSpace datasets to JSONL format for fine-tuning.
This script converts generated datasets to JSONL format suitable for model fine-tuning.
Supports causal, boolean, and 3d tasks.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def convert_causal_dataset(dataset: Dict) -> Dict:
    """Convert causal dataset to instruction format."""
    # Create instruction based on actual prompt used in causal task
    instruction = """You are given observations from perturbation experiments on a causal system.

Semantics:
- When a node is perturbed, the perturbed node is 0.
- A node is 1 if it is a downstream descendant of the perturbed node in the causal graph.
- All other nodes are 0.

Nodes: A, B, C

Observations:"""
    
    # Create input from observations
    observations = dataset['observations']
    observation_lines = [obs['string'] for obs in observations]
    input_text = "\n".join(observation_lines)
    
    # Create output from ground truth graphs in the format used by the model
    ground_truth_graphs = dataset['ground_truth_graphs']
    output_lines = []
    for i, graph in enumerate(ground_truth_graphs, 1):
        edges = graph['edges']
        
        # Format edges in the format expected by the model
        if edges:
            edge_str = ", ".join([f"{edge[0]}->{edge[1]}" for edge in edges])
            output_line = f"Graph: {edge_str}"
        else:
            output_line = "Graph: No edges"
        output_lines.append(output_line)
    
    output_text = "\n".join(output_lines)
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }


def convert_boolean_dataset(dataset: Dict) -> Dict:
    """Convert boolean dataset to instruction format."""
    # Create instruction based on actual prompt used in boolean task
    instruction = """You are given partial observations of a Boolean function with variables: x, y

Allowed operators: AND (conjunction), OR (disjunction), NOT (negation)

Observations (input -> output):"""
    
    # Create input from observations
    observations = dataset['observations']
    observation_lines = [obs['string'] for obs in observations]
    input_text = "\n".join(observation_lines)
    
    # Create output from ground truth expressions in the format used by the model
    # For fine-tuning, we want to show just one expression per line, not all of them
    ground_truth_expressions = dataset['ground_truth_expressions']
    output_lines = []
    for expr in ground_truth_expressions:
        formula = expr['formula']
        output_line = f"Expression: {formula}"
        output_lines.append(output_line)
    
    output_text = "\n".join(output_lines)
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }


def convert_3d_dataset(dataset: Dict) -> Dict:
    """Convert 3d dataset to instruction format."""
    # Create instruction based on actual prompt used in 3d task
    instruction = """You are given observations of a 3D structure made of unit blocks on a 3x3 grid.
Each observation shows a view of the structure from a specific angle.
The maximum height of the structure is 3 layers.

Observations (Top View - shows 1 if ANY layer has a block at that position):"""
    
    # Create input from observations
    # For 3D task, the observation is a string of 0s and 1s
    observation_str = dataset['observation']
    # Convert to grid format for better readability
    grid_size = int(len(observation_str) ** 0.5)
    input_lines = ["Top view:"]
    for i in range(grid_size):
        row = observation_str[i * grid_size:(i + 1) * grid_size]
        input_lines.append(" ".join(row))
    input_text = "\n".join(input_lines)
    
    # Create output from ground truth structures
    ground_truth_structures = dataset['ground_truth_structures']
    output_lines = ["Structure:"]
    
    # For fine-tuning, we'll use the first structure as an example
    if ground_truth_structures:
        first_structure = ground_truth_structures[0]
        layers = first_structure['layers']
        
        # Convert layers to the format expected by the model
        for i, layer_str in enumerate(layers, 1):
            output_lines.append(f"Layer {i}:")
            grid_size = int(len(layer_str) ** 0.5)
            for j in range(grid_size):
                row = layer_str[j * grid_size:(j + 1) * grid_size]
                output_lines.append(" ".join(row))
    
    output_text = "\n".join(output_lines)
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }


def convert_dataset_to_jsonl(input_file: str, output_file: str, task_type: str) -> None:
    """
    Convert dataset to JSONL format.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file
        task_type: Type of task ('causal', 'boolean', or '3d')
    """
    # Read input dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract datasets
    if 'datasets' in data:
        datasets = data['datasets']
    elif 'datasets_by_n_observations' in data:
        # Flatten datasets from observation groups
        datasets = []
        for n_obs, obs_datasets in data['datasets_by_n_observations'].items():
            datasets.extend(obs_datasets)
    elif 'observation_sets' in data:
        # 3D dataset format
        datasets = data['observation_sets']
    else:
        raise ValueError("Unknown dataset format")
    
    # Select conversion function based on task type
    if task_type == 'causal':
        convert_function = convert_causal_dataset
    elif task_type == 'boolean':
        convert_function = convert_boolean_dataset
    elif task_type == '3d':
        convert_function = convert_3d_dataset
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Convert to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for dataset in datasets:
            try:
                # Convert dataset to instruction format
                json_line = convert_function(dataset)
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Warning: Skipping dataset due to conversion error: {e}")
                continue
    
    print(f"Converted {len(datasets)} datasets to JSONL format")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert HypoSpace dataset to JSONL format for fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, help="Output JSONL file path")
    parser.add_argument("--task", type=str, choices=['causal', 'boolean', '3d'], required=True, 
                        help="Task type")
    
    args = parser.parse_args()
    
    if args.output is None:
        # Generate output filename based on input
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix('.jsonl'))
    
    convert_dataset_to_jsonl(args.input, args.output, args.task)


if __name__ == "__main__":
    main()