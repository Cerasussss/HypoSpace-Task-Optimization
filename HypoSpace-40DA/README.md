# HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination

## Abstract
As language models are increasingly used in scientific workflows, evaluating their ability to propose sets of explanations—not just a single correct answer—becomes critical. Many scientific problems are underdetermined: multiple, mechanistically distinct hypotheses are consistent with the same observations. We introduce HypoSpace, a diagnostic suite that treats LLMs as samplers of finite hypothesis sets and measures three complementary indicators: Validity (precision of proposals consistent with observations), Uniqueness (non-redundancy among proposals), and Recovery (coverage of the enumerated admissible set). We instantiate HypoSpace in three structured domains with deterministic validators and exactly enumerated hypothesis spaces: (i) causal graphs from perturbations, (ii) gravity-constrained 3D voxel reconstruction from top-down projections, and (iii) Boolean genetic interactions. Across instruction-tuned and reasoning-focused models, Validity often remains high while Uniqueness and Recovery degrade as the admissible space grows, revealing mode collapse that is invisible to correctness-only metrics. HypoSpace offers a controlled probe—rather than a leaderboard—for methods that explicitly explore and cover admissible explanation spaces.

## Scripts

### Create datasets
Example: generate causal dataset with 3 nodes
```bash
nohup python -u generate_causal_dataset.py \
  --nodes 3 \
  --seed 33550336 \
  --output "datasets/node03/n3_all_observations.json" \
  > log_datasets_node03 2>&1 &
```

### Run Benchmark 
Example: for causal dataset with 3 nodes
```bash
nohup python -u run_causal_benchmark.py \
      --dataset "datasets/node03/n3_all_observations.json" \
      --config "config/config_gpt4o.yaml" \
      --n-samples "30" \
      --query-multiplier "1.0" \
      --seed "33550336" > gpt4o_node03 2>&1 &
```

### Run Benchmark with Local Qwen Model
Example: for causal dataset with 3 nodes using local Qwen model
```bash
nohup python -u run_causal_benchmark.py \
      --dataset "datasets/node03/n3_all_observations.json" \
      --config "causal/config/config_qwen_local.yaml" \
      --n-samples "30" \
      --query-multiplier "1.0" \
      --seed "33550336" > qwen_local_node03 2>&1 &
```

For more details on using local Qwen models, see [causal/README_QWEN_LOCAL.md](causal/README_QWEN_LOCAL.md)
