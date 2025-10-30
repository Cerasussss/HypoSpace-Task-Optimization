#!/usr/bin/env python3
"""
Generate a Boolean dataset for model fine-tuning.
This script generates a dataset with varied observation combinations 
and their corresponding ground truth Boolean expressions for training models.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
import json
import numpy as np
from itertools import product, combinations
from datetime import datetime
import random
import sys

# Add project root to path and use local modules
sys.path.insert(0, '/opt/data/private/HypoSpace-40DA')
from boolean.boolean_dataset import BooleanExpression, BooleanObservation, BooleanDiscoveryGame

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

def generate_balanced_boolean_dataset(
    variables: List[str],
    operators: Set[str],
    max_depth: int,
    n_observations_range: Tuple[int, int],
    mechanistic_opts: Dict[str, Any],
    seed: Optional[int] = None,
    samples_per_obs_count: int = 100
) -> List[Dict]:
    """
    Generate a balanced Boolean dataset for fine-tuning with varied observation counts.
    
    Args:
        variables: List of variable names (e.g., ['x', 'y'])
        operators: Set of allowed operators
        max_depth: Maximum expression depth
        n_observations_range: Tuple of (min_observations, max_observations)
        mechanistic_opts: Options for mechanistic deduplication
        seed: Random seed for reproducibility
        samples_per_obs_count: Number of samples to generate for each observation count
        
    Returns:
        List of dataset dictionaries
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Generate all expressions once
    print(f"Generating all Boolean expressions with variables {variables}...")
    all_exprs = BooleanDiscoveryGame.generate_all_expressions(
        variables, operators, max_depth, mechanistic_opts
    )
    print(f"Generated {len(all_exprs)} expressions")
    
    # Generate all possible inputs for 2 variables
    all_inputs = list(product([0, 1], repeat=len(variables)))
    
    datasets = []
    dataset_id = 0

    # Generate samples for each observation count in the range
    for n_obs in range(n_observations_range[0], min(n_observations_range[1], len(all_inputs)) + 1):
        print(f"Generating {samples_per_obs_count} samples with {n_obs} observations...")
        
        # Sample observation combinations
        sampled_combinations = []
        attempts = 0
        max_attempts = samples_per_obs_count * 10  # Limit attempts to avoid infinite loop
        
        # Generate all combinations first
        all_combinations = list(combinations(all_inputs, n_obs))
        print(f"  Total possible combinations: {len(all_combinations)}")
        
        # Sample combinations
        if len(all_combinations) <= samples_per_obs_count:
            # If we have fewer combinations than requested samples, use all of them
            selected_combinations = all_combinations
        else:
            # Otherwise, sample randomly
            indices = rng.choice(len(all_combinations), size=samples_per_obs_count, replace=False)
            selected_combinations = [all_combinations[i] for i in indices]
        
        print(f"  Selected {len(selected_combinations)} combinations")
        
        # Process each combination
        for obs_combo in selected_combinations:
            # For each combination of inputs, try all possible output assignments
            # With n observations, there are 2^n possible output assignments
            # We'll sample a few of these
            n_output_assignments = min(2**n_obs, 10)  # Limit to 10 to avoid explosion
            for _ in range(n_output_assignments):
                # Randomly assign outputs
                output_assignment = rng.choice([0, 1], size=n_obs)
                
                # Create observations
                observations = []
                for inputs, output in zip(obs_combo, output_assignment):
                    inputs_dict = {var: val for var, val in zip(variables, inputs)}
                    observations.append(BooleanObservation(inputs_dict, int(output)))
                
                # Find all compatible expressions
                compatible_exprs = []
                for expr in all_exprs:
                    is_compatible = True
                    for obs in observations:
                        if expr.evaluate(obs.inputs) != obs.output:
                            is_compatible = False
                            break
                    if is_compatible:
                        compatible_exprs.append(expr)
                
                # Only create dataset if there are compatible expressions
                if compatible_exprs:
                    dataset_id += 1
                    # Create observation description
                    obs_desc = "_".join([f"{inputs[0]}{inputs[1]}" for inputs in obs_combo])
                    out_desc = "".join(map(str, output_assignment))
                    
                    dataset = {
                        "id": f"finetune_{dataset_id:05d}",
                        "n_observations": int(n_obs),
                        "observation_inputs": [list(inputs) for inputs in obs_combo],
                        "observation_outputs": [int(output) for output in output_assignment],
                        "observations": [{"inputs": obs.inputs, "output": int(obs.output), "string": obs.to_string()} 
                                       for obs in observations],
                        "ground_truth_expressions": [
                            {
                                "formula": expr.formula,
                                "canonical_form": expr.get_canonical_form(),
                                "mechanistic_key": str(expr.mechanistic_key(**mechanistic_opts)),
                                "truth_table": {str(k): int(v) for k, v in expr.truth_table.items()}
                            } for expr in compatible_exprs
                        ],
                        "n_compatible_expressions": int(len(compatible_exprs)),
                        "variables": variables,
                        "operators": list(operators),
                        "max_depth": int(max_depth)
                    }
                    datasets.append(dataset)

    print(f"Total datasets generated: {len(datasets)}")
    return datasets

def main():
    parser = argparse.ArgumentParser(description="Generate Boolean dataset for model fine-tuning")
    parser.add_argument("--variables", nargs='+', default=['x', 'y'], help="Variable names")
    parser.add_argument("--operators", choices=['basic', 'extended', 'full'], default='extended', help="Operator set")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum expression depth")
    parser.add_argument("--min-observations", type=int, default=1, help="Minimum number of observations")
    parser.add_argument("--max-observations", type=int, default=None, help="Maximum number of observations (default: 2^num_variables)")
    parser.add_argument("--samples-per-obs-count", type=int, default=50, help="Number of samples per observation count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    # Mechanistic knobs (defaults align with your earlier choice)
    parser.add_argument("--mech-no-comm", dest="comm", action="store_false",help="Disable commutativity collapse (ordered children)")
    parser.add_argument("--mech-no-idem", dest="idem", action="store_false",help="Disable idempotence collapse for AND/OR (keep x AND x)")
    parser.add_argument("--mech-no-flat", dest="flat", action="store_false",help="Disable associativity flattening")
    
    parser.set_defaults(comm=True, idem=True, flat=True)
    
    args = parser.parse_args()
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"boolean_finetuning_dataset_{timestamp}.json"
    
    if args.max_observations is None:
        args.max_observations = 2**len(args.variables)
    
    mechanistic_opts = dict(
        apply_commutativity=args.comm,
        apply_idempotence_and_or=args.idem,
        flatten_associativity=args.flat
    )
    
    # Define operator sets
    operator_sets = {
        'basic': {'AND', 'OR'},
        'extended': {'AND', 'OR', 'NOT'},
        'full': {'AND', 'OR', 'NOT', 'NOR'}
    }
    operators = operator_sets.get(args.operators, {'AND', 'OR', 'NOT'})
    
    # Generate dataset
    print("=" * 60)
    print("GENERATING BOOLEAN FINETUNING DATASET")
    print("=" * 60)
    print(f"Variables: {args.variables}")
    print(f"Operators: {operators}")
    print(f"Max depth: {args.max_depth}")
    print(f"Observations range: {args.min_observations} to {args.max_observations}")
    print(f"Samples per observation count: {args.samples_per_obs_count}")
    print()
    
    datasets = generate_balanced_boolean_dataset(
        variables=args.variables,
        operators=operators,
        max_depth=args.max_depth,
        n_observations_range=(args.min_observations, args.max_observations),
        mechanistic_opts=mechanistic_opts,
        seed=args.seed,
        samples_per_obs_count=args.samples_per_obs_count
    )
    
    # Create result structure
    result = {
        "metadata": {
            "variables": args.variables,
            "operators": list(operators),
            "max_depth": int(args.max_depth),
            "min_observations": int(args.min_observations),
            "max_observations": int(args.max_observations),
            "samples_per_obs_count": int(args.samples_per_obs_count),
            "mechanistic_opts": mechanistic_opts,
            "seed": int(args.seed) if args.seed is not None else None,
            "n_datasets": int(len(datasets)),
            "generation_date": datetime.now().isoformat()
        },
        "datasets": datasets
    }
    
    # Convert numpy types to native Python types
    result = convert_numpy_types(result)
    
    # Save to file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDataset saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()