#!/usr/bin/env python3
"""
Generate a causal dataset for model fine-tuning.
This script generates a dataset with varied observation combinations 
and their corresponding ground truth causal graphs for training models.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import json
import math
import numpy as np
import networkx as nx
from itertools import combinations
from datetime import datetime
import random
import sys

# Add project root to path and use local modules
sys.path.insert(0, '/opt/data/private/HypoSpace-40DA')
from causal.modules.models import CausalGraph

def _combo_has_unique_perturbed_nodes(combo) -> bool:
    """True iff no two observations perturb the same node."""
    return len({o.perturbed_node for o in combo}) == len(combo)

class PerturbationObservation:
    """Represents a perturbation and its effects."""
    
    def __init__(self, perturbed_node: str, effects: Dict[str, int]):
        """
        Args:
            perturbed_node: The node that was perturbed
            effects: Dictionary mapping node names to binary effects (0 or 1)
        """
        self.perturbed_node = perturbed_node
        self.effects = effects
    
    def to_tuple(self) -> Tuple:
        """Convert to hashable tuple for comparison."""
        sorted_effects = tuple(sorted(self.effects.items()))
        return (self.perturbed_node, sorted_effects)
    
    def to_string(self) -> str:
        """Human-readable string representation."""
        effect_str = " ".join([f"{node}:{val}" for node, val in sorted(self.effects.items())])
        return f"Perturb({self.perturbed_node}) -> {effect_str}"
    
    def __eq__(self, other) -> bool:
        return self.to_tuple() == other.to_tuple()
    
    def __hash__(self) -> int:
        return hash(self.to_tuple())

class CausalFinetuningDatasetGenerator:
    """Generate datasets for model fine-tuning with balanced sampling."""
    
    @staticmethod
    def get_perturbation_effects(
        graph: CausalGraph,
        perturbed_node: str,
        *,
        desc_map: Optional[Dict[str, Set[str]]] = None
    ) -> PerturbationObservation:
        """
        Effects of perturbing a node:
        - perturbed node -> 0
        - descendants(perturbed) -> 1
        - everyone else -> 0
        """
        G = graph.to_networkx()
        if desc_map is None:
            # Precompute descendants for all nodes if not provided
            desc_map = {n: nx.descendants(G, n) for n in G.nodes}

        effects = {n: (1 if n in desc_map[perturbed_node] else 0) for n in graph.nodes}
        effects[perturbed_node] = 0  # ensure intervention node is 0
        return PerturbationObservation(perturbed_node, effects)
        
    @staticmethod
    def generate_all_dags(nodes: List[str], max_edges: Optional[int] = None) -> List[CausalGraph]:
        """
        Generate ALL possible DAGs with the given nodes.
        
        Args:
            nodes: List of node names
            max_edges: Maximum number of edges (None for no limit)
            
        Returns:
            List of all possible DAGs
        """
        all_dags = []
        all_possible_edges = [(i, j) for i in nodes for j in nodes if i != j]
        
        if max_edges is None:
            max_edges = len(all_possible_edges)
        
        # Try all possible edge combinations
        for edge_count in range(min(max_edges + 1, len(all_possible_edges) + 1)):
            for edge_combo in combinations(all_possible_edges, edge_count):
                # Check if this forms a valid DAG
                test_graph = CausalGraph(nodes=nodes, edges=list(edge_combo))
                G = test_graph.to_networkx()
                
                if nx.is_directed_acyclic_graph(G):
                    all_dags.append(test_graph)
        
        return all_dags
    
    @staticmethod
    def generate_balanced_dataset(
        nodes: List[str],
        n_observations_range: Tuple[int, int],
        max_edges: Optional[int] = None,
        seed: Optional[int] = None,
        samples_per_obs_count: int = 100
    ) -> List[Dict]:
        """
        Generate a balanced dataset for fine-tuning with varied observation counts.
        
        Args:
            nodes: List of node names
            n_observations_range: Tuple of (min_observations, max_observations)
            max_edges: Maximum edges in generated DAGs
            seed: Random seed for reproducibility
            samples_per_obs_count: Number of samples to generate for each observation count
            
        Returns:
            List of dataset dictionaries
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate all possible DAGs
        print(f"Generating all DAGs with {len(nodes)} nodes...")
        all_dags = CausalFinetuningDatasetGenerator.generate_all_dags(nodes, max_edges)
        print(f"Generated {len(all_dags)} DAGs")
        
        # Precompute descendants/effects per DAG (speedup)
        dag_caches = []  # list of (dag, desc_map, effects_by_node)
        for dag in all_dags:
            G = dag.to_networkx()
            desc_map = {n: nx.descendants(G, n) for n in G.nodes}
            effects_by_node = {}
            for n in dag.nodes:
                # Build once per (dag, node)
                effects = {m: (1 if m in desc_map[n] else 0) for m in dag.nodes}
                effects[n] = 0
                effects_by_node[n] = PerturbationObservation(n, effects)
            dag_caches.append((dag, desc_map, effects_by_node))
        
        # Generate all possible perturbation observations (dedup)
        all_possible_observations = set()
        for _, _, effects_by_node in dag_caches:
            for n in nodes:
                all_possible_observations.add(effects_by_node[n])

        # Sort observations for deterministic ordering
        all_possible_observations = sorted(list(all_possible_observations), 
                                          key=lambda o: (o.perturbed_node, sorted(o.effects.items())))
        print(f"Total possible observations: {len(all_possible_observations)}")

        datasets = []
        dataset_id = 0

        # Generate samples for each observation count in the range
        for n_obs in range(n_observations_range[0], min(n_observations_range[1], len(nodes)) + 1):
            print(f"Generating {samples_per_obs_count} samples with {n_obs} observations...")
            
            # Sample observation combinations
            sampled_combinations = []
            attempts = 0
            max_attempts = samples_per_obs_count * 10  # Limit attempts to avoid infinite loop
            
            while len(sampled_combinations) < samples_per_obs_count and attempts < max_attempts:
                # Randomly sample a combination
                combo = tuple(random.sample(all_possible_observations, n_obs))
                
                # Only keep combinations with unique perturbed nodes
                if _combo_has_unique_perturbed_nodes(combo):
                    sampled_combinations.append(combo)
                
                attempts += 1
            
            print(f"  Generated {len(sampled_combinations)} valid combinations")
            
            # Process each combination
            for obs_combo in sampled_combinations:
                compatible_dags = []
                for dag, _, effects_by_node in dag_caches:
                    # All observations must match exactly what this DAG predicts
                    if all(effects_by_node[o.perturbed_node].effects == o.effects for o in obs_combo):
                        compatible_dags.append(dag)

                if compatible_dags:
                    dataset_id += 1
                    dataset = {
                        "id": f"finetune_{dataset_id:05d}",
                        "n_observations": n_obs,
                        "observations": [
                            {
                                "perturbed_node": o.perturbed_node,
                                "effects": o.effects,
                                "string": o.to_string(),
                            } for o in obs_combo
                        ],
                        "ground_truth_graphs": [dag.to_dict() for dag in compatible_dags],
                        "n_compatible_graphs": len(compatible_dags),
                        "nodes": nodes,
                        "max_edges": max_edges
                    }
                    datasets.append(dataset)

        print(f"Total datasets generated: {len(datasets)}")
        return datasets

def main():
    parser = argparse.ArgumentParser(description="Generate causal dataset for model fine-tuning")
    parser.add_argument("--nodes", type=int, default=3, help="Number of nodes in graphs")
    parser.add_argument("--min-observations", type=int, default=1, help="Minimum number of observations")
    parser.add_argument("--max-observations", type=int, default=None, help="Maximum number of observations (default: number of nodes)")
    parser.add_argument("--samples-per-obs-count", type=int, default=100, help="Number of samples per observation count")
    parser.add_argument("--max-edges", type=int, default=None, help="Maximum edges in DAGs (default: no limit)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"causal_finetuning_dataset_n{args.nodes}_{timestamp}.json"
    
    if args.max_observations is None:
        args.max_observations = args.nodes
    
    # Generate node names
    nodes = [chr(65 + i) for i in range(args.nodes)]  # A, B, C, ...
    
    # Generate dataset
    print("=" * 60)
    print("GENERATING CAUSAL FINETUNING DATASET")
    print("=" * 60)
    print(f"Nodes: {nodes}")
    print(f"Observations range: {args.min_observations} to {args.max_observations}")
    print(f"Samples per observation count: {args.samples_per_obs_count}")
    print(f"Max edges: {args.max_edges}")
    print()
    
    datasets = CausalFinetuningDatasetGenerator.generate_balanced_dataset(
        nodes=nodes,
        n_observations_range=(args.min_observations, args.max_observations),
        max_edges=args.max_edges,
        seed=args.seed,
        samples_per_obs_count=args.samples_per_obs_count
    )
    
    # Create result structure
    result = {
        "metadata": {
            "nodes": nodes,
            "min_observations": args.min_observations,
            "max_observations": args.max_observations,
            "samples_per_obs_count": args.samples_per_obs_count,
            "max_edges": args.max_edges,
            "seed": args.seed,
            "n_datasets": len(datasets),
            "generation_date": datetime.now().isoformat()
        },
        "datasets": datasets
    }
    
    # Save to file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDataset saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()