## Robust and Computation-aware Gaussian Process

This is the robust and computation-aware Gaussian process (RCaGP) code ([arxiv link](https://arxiv.org/abs/2505.21133)) applied for high-throughput BO. The repository includes the code to run EULBO and DPP-BO methods using RCaGP as well as the baseline methods. The codes are forked from the official implementation of EULBO by Maus, et al., 2024. 

## Environment 

Run ```pip install -r requirements.txt``` to setup the environment used to run all experiments.

## 1. UCI regression
Use the argument ```--dataset``` to set the dataset you'd like to run by providing the string id for the specific task.

## 2. High-throughput BO
| dataset | UCI Dataset Name |
|----------|----------|
| yacht   | Yacht   |
| boston   | Boston   |
| energy   | Energy   |
| parkinsons   | Parkinsons   |

#### Example Commands

```
OUTLIERS_TYPE="asymmetric" 
MODEL_NAME="RCaGP"
FRACTION=0.2
SEEDS=20
DATASET="yacht"
EPSILON=0.2
OBSVAR=1.0

srun python uci_experiment.py --dataset $DATASET --outlier_type $OUTLIERS_TYPE --fraction $FRACTION --seeds $SEEDS --epsilon 0.1 --models $MODEL_NAME --obsvar $OBSVAR
```

#### Tasks
Use the argument ```--task_id``` to set the optimization task you'd like to run by providing the string id for the specific task.

| task_id | Task Name |
|----------|----------|
| hartmann6   | Hartmann 6D   |
| lunar   | Lunar Lander 12D   |
| rover   | Rover 60D   |
| dna   | Lasso-DNA 180D   |

Lassobench DNA task requires the following additional steps to set up:

```
git clone https://github.com/ksehic/LassoBench.git
cd LassoBench/
pip install -e .
```

#### Example Commands

```
TASK="hartmann6"
MODEL="RCSVGP"
AF="EI"
EULBO="True"
USE_KG="False"
USE_TURBO="False"
RCSVGP_BASELINE="True"
RCAGP_BASELINE="False"
CAGP_BASELINE="False"
INFORMATIVE_MEAN="True"
MAX_N_ORACLE_CALLS=10000
SEED=10

srun python run_bo.py --task_id $TASK --seed $SEED --eulbo $EULBO --use_turbo $USE_TURBO --use_kg $USE_KG --rcsvgp_baseline $RCSVGP_BASELINE --rcagp_baseline $RCAGP_BASELINE --cagp_baseline $CAGP_BASELINE --informative_mean $INFORMATIVE_MEAN --max_n_oracle_calls $MAX_N_ORACLE_CALLS - run - done
```

This repo is set up to track BO progress using the Weights and Biases (wandb) API.


