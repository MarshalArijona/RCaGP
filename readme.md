## Robust Computation-aware Gaussian Process for High-throughput Bayesian Optimization

This is the implementation of robust computation-aware Gaussian process (RCaGP) applied for expected utility BO (EULBO). This repository includes the code to run EULBO method with RCaGP as well as the baseline methods. The codes are the modification of the official implementation of EULBO by Maus, et al., 2024. 


## Environment 

Run ```pip install -r requirements.txt``` to setup the environment used to run all experiments.


## Tasks
Use the argument ```--task_id``` to set the optimization task you'd like to run by providing the string id for the specific task.

| task_id | Task Name |
|----------|----------|
| hartmann6   | Hartmann 6D   |
| michalewicz   | Michalewicz 10D   |
| lunar   | Lunar Lander 12D   |
| rover   | Rover 60D   |


## Example Commands

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


