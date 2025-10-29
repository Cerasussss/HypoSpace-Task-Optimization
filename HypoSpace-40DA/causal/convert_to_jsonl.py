#!/usr/bin/env python3
"""
Convert causal dataset to JSONL format for fine-tuning.
This script converts the generated causal dataset to JSONL format suitable for model fine-tuning.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def convert_dataset_to_jsonl(input_file: str, output_file: str) -> None:
    """
    Convert causal dataset to JSONL format.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file
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
    else:
        raise ValueError("Unknown dataset format")

    # Convert to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for dataset in datasets:
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

            # Write to JSONL
            json_line = {
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

    print(f"Converted {len(datasets)} datasets to JSONL format")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert causal dataset to JSONL format for fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, help="Output JSONL file path")

    args = parser.parse_args()

    if args.output is None:
        # Generate output filename based on input
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix('.jsonl'))

    convert_dataset_to_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()