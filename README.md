# H-MAPPO for Multi-Elevator DAG Task Offloading

This repository contains the implementation of the algorithm:

> **Joint DAG Task Offloading and Resource Allocation for Multi-Elevator Systems in Cloud-Edge-End Environments**

The project studies deadline-aware DAG task offloading and computation-resource allocation in a cloud-edge-end system. H-MAPPO extends MAPPO with heuristic action-conditioned features and urgency-aware event-driven scheduling.

## Main components

- Event-driven cloud-edge-end simulation environment
- H-MAPPO, MAPPO, IPPO, DDQN, and TPTDTO
- Greedy, HEFT, and Local-only baselines
- Standardized public DAG workloads
- Elevator fault-diagnosis, group-dispatch, and predictive-maintenance workloads
- Pretrained models and experimental result files

## Repository structure

~~~text
.
├── configs/                       # Public-DAG configurations
├── elevator_experiments/          # Elevator datasets and experiments
├── image/                         # Public-DAG result files
├── model/                         # System and DAG models
├── offloading_data/               # Public DAGGen-v3 workloads
├── outputs/                       # Pretrained models and training logs
├── rl/                            # Learning algorithms and heuristics
└── scripts/                       # Training and evaluation scripts
~~~

## Environment

Python 3.10 or later is recommended.

~~~bash
conda create -n hmappo python=3.10 -y
conda activate hmappo
pip install torch numpy matplotlib pyyaml pydotplus
~~~

Run all commands from the repository root.

## Training

### Public DAG workloads

Train H-MAPPO:

~~~bash
python scripts/train_event_ppo.py --config configs/event_ppo_paper_public_v3.yaml --min_subtasks 10 --max_subtasks 30 --device cuda
~~~

Train MAPPO or IPPO by using the baseline configuration and the corresponding variant:

~~~bash
python scripts/train_event_ppo.py --config configs/event_ppo_baseline_paper_public_v3.yaml --min_subtasks 10 --max_subtasks 30 --device cuda --variant mappo
python scripts/train_event_ppo.py --config configs/event_ppo_baseline_paper_public_v3.yaml --min_subtasks 10 --max_subtasks 30 --device cuda --variant ippo
~~~

Train DDQN:

~~~bash
python scripts/train_event_offpolicy.py --config configs/event_ppo_baseline_paper_public_v3.yaml --device cuda --algorithm ddqn
~~~

Train TPTDTO:

~~~bash
python scripts/train_event_ppo.py --config configs/tptdto_paper_public_v3.yaml --device cuda
~~~

### Elevator DAG workloads

The elevator configurations are stored in <code>elevator_experiments/configs/</code>. For example:

~~~bash
python scripts/train_event_ppo.py --config elevator_experiments/configs/elevator_v1.yaml --device cuda
python scripts/train_event_ppo.py --config elevator_experiments/configs/mappo_elevator_v1.yaml --device cuda
python scripts/train_event_offpolicy.py --config elevator_experiments/configs/ddqn_elevator_v1.yaml --device cuda --algorithm ddqn
~~~

Replace <code>cuda</code> with <code>cpu</code> if a compatible GPU is unavailable. Trained models and logs are saved under <code>outputs/</code>.

## Evaluation

Evaluate the algorithms on the public DAG workloads:

~~~bash
python scripts/eval_public_heft_comparison.py --episodes 100 --device cpu --use_paper_v3_runs
~~~

Generate the ablation results from the included evaluation CSV:

~~~bash
python scripts/eval_public_v3_ablation.py --plot-only
~~~

Evaluate the algorithms on the three elevator-service workloads:

~~~bash
python elevator_experiments/evaluate_all.py --episodes 100 --device cpu --output-name all_algorithms_metrics_final100.csv
~~~

Run the robustness evaluation:

~~~bash
python elevator_experiments/evaluate_robustness.py --episodes 30 --device cpu --task-size 20 --edge-available 1.0 0.8 0.6 0.4 --bandwidths 10 6 4 2 --output-prefix robustness_edge_link_final30
~~~

Evaluation results are saved as CSV files under <code>image/</code> or <code>elevator_experiments/results/</code>. Generated plots are saved under <code>image/</code> or <code>elevator_experiments/figures/</code>.
