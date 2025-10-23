import sys 
sys.path.append("../")
import numpy as np
import torch
import fire 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import signal 
import copy 
import gpytorch
import wandb
import time
import math

wandb.login(key="[YOUR_KEY]")

from src.variational.variational_ELBO import CaGPVariationalELBO, RCaGPVariationalELBO, RCaGP_DPPVariationalELBO, RCSVGPVariationalELBO
from src.models.SVGP import SVGP
from src.models.RCSVGP import RCSVGP
from src.models.RCaGP import RCaGP, RCaGP_DPP, CaGP
from src.generate_candidates import generate_batch
from src.train_model import (
    update_model_elbo,
    update_model_and_generate_candidates_eulbo,
)

from utils.linop_matern_kernel import LinopMaternKernel
from utils.informative_mean import InformativeMeanPrior, simulate_expert_correction
from utils.weight_function import IMQ, get_soft_threshold, StandardWeight
from utils.create_wandb_tracker import create_wandb_tracker
from utils.set_seed import set_seed
from utils.get_random_init_data import get_random_init_data
from utils.turbo import TurboState, update_state
from utils.set_inducing_points_with_moss23 import get_optimal_inducing_points

# for exact gp baseline:
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

# for specific tasks
from tasks.michalewicz import Michalewicz10
from tasks.hartmann import Hartmann6D
from tasks.rover import RoverObjective
try:
    from tasks.lunar import LunarLanderObjective
    successful_lunar_import = True
except:
    print("Warning: failed to import LunarLanderObjective, current environment does not support needed imports for lunar lander task")
    successful_lunar_import = False

try:
    from tasks.lasso_dna import LassoDNA
    successful_lasso_dna_import = True 
except:
    print("Warning: failed to import LassoDNA Objective, current environment does not support needed imports for Lasso DNA task")
    successful_lasso_dna_import = False 

task_id_to_objective = {}
task_id_to_objective['hartmann6'] = Hartmann6D
task_id_to_objective['michalewicz'] = Michalewicz10
if successful_lunar_import:
    task_id_to_objective['lunar'] = LunarLanderObjective 
task_id_to_objective['rover'] = RoverObjective
if successful_lasso_dna_import: 
    task_id_to_objective['dna'] = LassoDNA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Optimize(object):
    """
    Run Approximation Aware Bayesian Optimization (AABO)
    Args:
        task_id: String id for optimization task in task_id_to_objective dict 
        seed: Random seed to be set. If None, no particular random seed is set
        wandb_entity: Username for your wandb account for wandb logging
        wandb_project_name: Name of wandb project where results will be logged, if none specified, will use default f"run-aabo-{task_id}"
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        bsz: Acquisition batch size
        train_bsz: batch size used for model training/updates
        num_initialization_points: Number of initial random data points used to kick off optimization
        lr: Learning rate for model updates
        x_next_lr: Learning rate for x next updates with EULBO method 
        acq_func: Acquisition function used for warm-starting model, must be either ei, logei, or ts (logei--> Log Expected Imporvement, ei-->Expected Imporvement, ts-->Thompson Sampling)
        n_update_epochs: Number of epochs to update the model for on each optimization step
        n_inducing_pts: Number of inducing points for GP
        grad_clip: clip the gradeint at this value during model training 
        eulbo: If True, use EULBO for model training and canidate selection (AABO), otherwise use the standard ELBO (i.e. standard BO baselines).
        use_turbo: If True, use trust region BO, used for higher-dim tasks in the paper 
        use_kg: If True, use EULBO-KG. Otherwise, use EULBO-EI 
        exact_gp_baseline: If True, instead of AABO run baseline of vanilla BO with exact GP 
        ablation1_fix_indpts_and_hypers: If True, run AABO ablation from paper where inducing points and hyperparams remain fixed (not udated by EULBO)
        ablation2_fix_hypers: If True, run AABO ablation from paper where hyperparams remain fixed (not udated by EULBO)
        moss23_baseline: If True, instead of AABO run the moss et al. 2023 paper method baseline (use inducing point selection method of every iteration of optimization)
        inducing_pt_init_w_moss23: If True, use moss et al. 2023 paper method to initialize inducing points at the start of optimizaiton 
        normalize_ys: If True, normalize objective values for training (recommended, typical when using GP models)
        max_allowed_n_failures_improve_loss: We train model until the loss fails to improve for this many epochs
        max_allowed_n_epochs: Although we train to convergence, we also cap the number of epochs to this max allowed value
        n_warm_start_epochs: Number of epochs used to warm start the GP model with standard ELBO before beginning training with EULBO
        alternate_eulbo_updates: If true, we alternate updates of model and x_next when training with EULBO (imporves training convergence and stability)
        update_on_n_pts: Update model on this many data points on each iteration.
        num_kg_samples: number of samples used to compute log utility with KG 
        num_mc_samples_qei: number of MC samples used to ocmpute log utility with aEI 
        float_dtype_as_int: specify integer either 32 or 64, dictates whether to use torch.float32 or torch.float64 
        use_botorch_stable_log_softplus: if True, use botorch new implementation of log softplus (https://botorch.org/api/_modules/botorch/utils/safe_math.html#log_softplus)
        verbose: if True, print optimization progress updates 
        ppgpr:  if True, use PPGPR instead of SVGP 
        run_id: Optional string run id. Only use is for wandb logging to identify a specific run
    """
    def __init__(
        self,
        task_id: str="hartmann6",
        seed: int=None,
        wandb_entity: str="arijonamarshal-aalto-university",
        wandb_project_name: str="",
        max_n_oracle_calls: int=20000,
        bsz: int=1, 
        train_bsz: int=20,
        num_initialization_points: int=250, 
        lr: float=0.01,
        x_next_lr: float=0.001,
        acq_func: str="ei",
        n_update_epochs: int=5,
        n_inducing_pts: int=5,
        grad_clip: float=2.0,
        eulbo: bool=True,
        use_turbo=False,
        use_kg=False,
        exact_gp_baseline=False,
        ablation1_fix_indpts_and_hypers=False,
        ablation2_fix_hypers=False,
        moss23_baseline=False,
        rcsvgp_baseline=False,
        rcagp_baseline=False,
        cagp_baseline=False,
        inducing_pt_init_w_moss23=False,
        normalize_ys=True,
        max_allowed_n_failures_improve_loss: int=3,
        max_allowed_n_epochs: int=30, 
        n_warm_start_epochs: int=10,
        alternate_eulbo_updates=True,
        update_on_n_pts=25,
        num_kg_samples=64,
        num_mc_samples_qei=64,
        float_dtype_as_int=64,
        use_botorch_stable_log_softplus=False,
        verbose=True,
        ppgpr=False,
        run_id="",
        informative_mean=False,
        p_outliers=0.25,
        lb_outliers=1.25,
        ub_outliers=2.25,
        head_idx=225,
        robust=True,
        noise_type="asymmetric"
    ):
        if float_dtype_as_int == 32:
            self.dtype = torch.float32
            torch.set_default_dtype(torch.float32)
        elif float_dtype_as_int == 64:
            self.dtype = torch.float64
            torch.set_default_dtype(torch.float64)
        else:
            assert 0, f"float_dtype_as_int must be one of: [32, 64], instead got {float_dtype_as_int}"

        if ablation1_fix_indpts_and_hypers:
            assert eulbo
            assert not ablation2_fix_hypers
        
        if ablation2_fix_hypers:
            assert eulbo
            assert not ablation1_fix_indpts_and_hypers
        
        if moss23_baseline:
            assert not eulbo
            assert not exact_gp_baseline
            assert inducing_pt_init_w_moss23

        if eulbo:
            assert not exact_gp_baseline

        if exact_gp_baseline:
            assert not eulbo
        
        # log all args to wandb
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        wandb_config_dict = {k: v for method_dict in self.method_args.values() for k, v in method_dict.items()}

        self.run_id = run_id
        self.ppgpr = ppgpr
        self.use_botorch_stable_log_softplus = use_botorch_stable_log_softplus
        self.init_training_complete = False
        self.n_warm_start_epochs = n_warm_start_epochs
        self.normalize_ys = normalize_ys
        self.ablation1_fix_indpts_and_hypers = ablation1_fix_indpts_and_hypers
        self.ablation2_fix_hypers = ablation2_fix_hypers
        self.inducing_pt_init_w_moss23 = inducing_pt_init_w_moss23
        self.moss23_baseline = moss23_baseline
        self.exact_gp_baseline = exact_gp_baseline
        self.use_turbo = use_turbo
        self.update_on_n_pts = update_on_n_pts
        self.verbose = verbose
        self.x_next_lr = x_next_lr
        self.alternate_eulbo_updates = alternate_eulbo_updates
        self.max_allowed_n_failures_improve_loss = max_allowed_n_failures_improve_loss
        self.max_allowed_n_epochs = max_allowed_n_epochs
        self.eulbo = eulbo
        self.max_n_oracle_calls=max_n_oracle_calls
        self.n_inducing_pts=n_inducing_pts
        self.lr=lr
        self.n_update_epochs=n_update_epochs
        self.train_bsz=train_bsz
        self.grad_clip=grad_clip
        self.bsz=bsz
        self.acq_func=acq_func
        self.num_kg_samples = num_kg_samples
        self.use_kg = use_kg
        self.num_mc_samples_qei = num_mc_samples_qei
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(seed)
        self.rcsvgp_baseline = rcsvgp_baseline
        self.rcagp_baseline = rcagp_baseline
        self.cagp_baseline = cagp_baseline
        self.informative_mean = informative_mean
        self.p_outliers = p_outliers
        self.lb_outliers = lb_outliers
        self.ub_outliers = ub_outliers
        self.head_idx = head_idx
        self.robust = robust
        self.noise_type = noise_type

        # start wandb tracker
        if self.rcsvgp_baseline:
            if self.informative_mean:
                model_group = "RCSVGP_expert"
            else:
                model_group = "RCSVGP"
        elif self.rcagp_baseline:
            if self.informative_mean:
                model_group = "RCaGP_expert"
            else:
                model_group = "RCaGP"
        elif self.cagp_baseline:
            model_group = "CaGP"
        else:
            model_group = "SVGP"

        if self.use_kg:
            acq_name = "kg"
        else:
            acq_name = self.acq_func
            
        if not wandb_project_name:
            if self.use_turbo:
                if self.moss23_baseline:
                    wandb_project_name = f"run-aabo-moss-turbo-{task_id}-{acq_name}-{self.noise_type}-{self.p_outliers}"    
                else:
                    wandb_project_name = f"run-aabo-turbo-{task_id}-{acq_name}-{self.noise_type}-{self.p_outliers}"
            
            else:
                if self.moss23_baseline:
                    wandb_project_name = f"run-aabo-moss-{task_id}-{acq_name}-{self.noise_type}-{self.p_outliers}"
                else:
                    wandb_project_name = f"run-aabo-hvfner-{task_id}-{acq_name}-{self.noise_type}-{self.p_outliers}"

        self.tracker = create_wandb_tracker(
            wandb_project_name=wandb_project_name,
            wandb_entity=wandb_entity,
            config_dict=wandb_config_dict,
            wandb_group=model_group,
            wandb_job_type=acq_func,
            wandb_name=wandb_project_name + f"-{model_group}-{acq_func}-{seed}",
        )
        signal.signal(signal.SIGINT, self.handler)

        # define objective and get initialization data
        if task_id in task_id_to_objective:
            self.objective = task_id_to_objective[task_id](dtype=self.dtype)
            # get random init training data
            self.train_x, self.train_y = get_random_init_data(
                num_initialization_points=num_initialization_points,
                objective=self.objective,
            )
            
        if self.verbose:
            print("train x shape:", self.train_x.shape)
            print("train y shape:", self.train_y.shape)

        # get normalized train y
        self.train_y_mean = self.train_y.mean()
        self.train_y_std = self.train_y.std()
        if self.train_y_std == 0:
            self.train_y_std = 1
        if self.normalize_ys:
            self.normed_train_y = (self.train_y - self.train_y_mean) / self.train_y_std
        else:
            self.normed_train_y = self.train_y

        #contaminated normalized observations
        
        outliers_noise, self.outliers_idx, outliers_mask = self.objective.get_contamination(self.normed_train_y, probability=self.p_outliers)
        self.normed_train_y_std = self.normed_train_y.std()
        self.normed_train_y = self.objective.inject_contamination(self.normed_train_y, self.normed_train_y_std, outliers_noise, outliers_mask, low=self.lb_outliers, high=self.ub_outliers, noise_type=self.noise_type)
        self.contaminated_train_y = self.normed_train_y * self.train_y_std + self.train_y_mean

        #initialize turbo state
        self.tr_state = TurboState(
            dim=self.train_x.shape[-1],
            batch_size=self.bsz,
            best_value=self.train_y.max().item(),
        )
        
        # initialize GP model
        if not self.exact_gp_baseline:
            # get inducing points
            if len(self.train_x) >= self.n_inducing_pts:
                inducing_points = self.train_x[0:self.n_inducing_pts,:].clone()
            else:
                n_extra_ind_pts = self.n_inducing_pts - len(self.train_x)
                extra_ind_pts = torch.rand(n_extra_ind_pts, self.objective.dim)*(self.objective.ub - self.objective.lb) + self.objective.lb
                inducing_points = torch.cat((self.train_x, extra_ind_pts), -2)
            self.initial_inducing_points = copy.deepcopy(inducing_points) 
            
            if task_id == "lunar":
                self.sigma_sq = 5.0
            else:
                self.sigma_sq = 1.0

            learn_inducing_locations = True
            if self.moss23_baseline:
                learn_inducing_locations = False

            input_dim = self.train_x.size(0)  # high dimensional input
            loc = math.sqrt(2) + 0.5 * math.log(input_dim)
            scale = math.sqrt(3)
            lengthscale_prior = LogNormalPrior(loc=loc, scale=scale)
            #lengthscale_prior = None

            # define approximate GP model
            if self.rcsvgp_baseline:
                if self.informative_mean and len(self.outliers_idx) > 0:
                    
                    #defining mask to fetch the inliers and outliers
                    mask = torch.ones(self.train_x.size(0), dtype=torch.bool)
                    mask[self.outliers_idx] = False
                    x_inliers = self.train_x[mask]
                    x_outliers = self.train_x[self.outliers_idx]
                    y_inliers = self.contaminated_train_y[mask]
                    y_outliers = self.contaminated_train_y[self.outliers_idx]
                    self.Y_corrections = simulate_expert_correction(y_outliers.to(self.device), 
                                            sigma_sq=self.sigma_sq)
                    shifting_const = 0.0
                    
                    mean_module = InformativeMeanPrior(
                        x_inliers.to(self.device),
                        x_outliers.to(self.device),
                        y_inliers.to(self.device),
                        y_outliers.to(self.device),
                        self.Y_corrections.to(self.device),
                        self.train_y_mean,
                        self.train_y_std,
                        sigma_sq_correction=self.sigma_sq,
                        const=shifting_const,
                    ).to(self.device)
                else:
                    mean_module = gpytorch.means.ConstantMean().to(self.device)
                    const = 0.0
                    mean_module.constant = torch.tensor(const)
                    for param in mean_module.parameters():
                        param.requires_grad = False
                
                covar_module = gpytorch.kernels.ScaleKernel(LinopMaternKernel(lengthscale_prior=lengthscale_prior))
                likelihood=gpytorch.likelihoods.GaussianLikelihood()
                beta = 1.0
                C = 20
                weight_function = IMQ(mean_module, beta, C)

                self.model = RCSVGP(
                    inducing_points.to(self.device),
                    self.train_x.to(self.device), 
                    self.normed_train_y.squeeze().to(self.device),
                    mean_module,
                    covar_module,
                    likelihood.to(self.device),
                    weight_function,
                    learn_inducing_locations=learn_inducing_locations
                ).to(self.device)

                if self.inducing_pt_init_w_moss23:
                    optimal_inducing_points = get_optimal_inducing_points(
                        model=self.model,
                        prev_inducing_points=inducing_points.to(self.device),
                    )

                    self.model = RCSVGP(
                        optimal_inducing_points.to(self.device),
                        self.train_x.to(self.device),
                        self.normed_train_y.squeeze().to(self.device),
                        mean_module,
                        covar_module,
                        likelihood.to(self.device),
                        weight_function,
                        learn_inducing_locations=learn_inducing_locations,
                    ).to(self.device)
                    self.initial_inducing_points = copy.deepcopy(optimal_inducing_points)

            elif self.rcagp_baseline:
                if informative_mean and len(self.outliers_idx) > 0:
                    #defining mask to fetch the inliers and outliers
                    mask = torch.ones(self.train_x.size(0), dtype=torch.bool)
                    mask[self.outliers_idx] = False
                    x_inliers = self.train_x[mask]
                    x_outliers = self.train_x[self.outliers_idx]
                    
                    y_inliers = self.contaminated_train_y[mask]
                    y_outliers = self.contaminated_train_y[self.outliers_idx]

                    self.Y_corrections = simulate_expert_correction(y_outliers.to(self.device), 
                                                        sigma_sq=self.sigma_sq)
                    shifting_const = 0.0
                    mean_module = InformativeMeanPrior(
                        x_inliers.to(self.device),
                        x_outliers.to(self.device),
                        y_inliers.to(self.device),
                        y_outliers.to(self.device),
                        self.Y_corrections.to(self.device),
                        self.train_y_mean,
                        self.train_y_std,
                        sigma_sq_correction=self.sigma_sq,
                        const=shifting_const,
                    ).to(self.device)
                else:
                    mean_module = gpytorch.means.ConstantMean().to(self.device)
                    const = 0.0
                    mean_module.constant = torch.tensor(const)
                    for param in mean_module.parameters():
                        param.requires_grad = False
                    
                projection_dim = 5
                covar_module = gpytorch.kernels.ScaleKernel(LinopMaternKernel(lengthscale_prior=lengthscale_prior))
                likelihood=gpytorch.likelihoods.GaussianLikelihood()
                beta = 1.0
                C = 20
                weight_function = IMQ(mean_module, beta, C)
                
                if self.inducing_pt_init_w_moss23:
                    self.model = RCaGP_DPP(
                        train_inputs=self.train_x.to(self.device),
                        train_targets=self.normed_train_y.squeeze().to(self.device),
                        weight_function=weight_function,
                        mean_module=mean_module,
                        covar_module=covar_module,
                        likelihood=likelihood.to(self.device),
                        inducing_points=inducing_points.to(self.device),
                    ).to(self.device)

                    optimal_inducing_points = get_optimal_inducing_points(
                        model=self.model,
                        prev_inducing_points=inducing_points.to(self.device))

                    self.model = RCaGP_DPP(
                        train_inputs=self.train_x.to(self.device),
                        train_targets=self.normed_train_y.squeeze().to(self.device),
                        weight_function=weight_function,
                        mean_module=mean_module,
                        covar_module=covar_module,
                        likelihood=likelihood.to(self.device),
                        inducing_points=optimal_inducing_points.to(self.device),
                    ).to(self.device)
                    self.initial_inducing_points = copy.deepcopy(optimal_inducing_points)

                else:
                    self.model = RCaGP(
                        train_inputs=self.train_x.to(self.device),
                        train_targets=self.normed_train_y.squeeze().to(self.device),
                        weight_function=weight_function,
                        mean_module=mean_module,
                        covar_module=covar_module,
                        likelihood=likelihood.to(self.device),
                        projection_dim=projection_dim,
                        initialization="random",
                    ).to(self.device)

            elif self.cagp_baseline:
                mean_module = gpytorch.means.ConstantMean()
                const = 0.0
                mean_module.constant = torch.tensor(const)
                for param in mean_module.parameters():
                    param.requires_grad = False
                
                covar_module = gpytorch.kernels.ScaleKernel(LinopMaternKernel(lengthscale_prior=lengthscale_prior))
                likelihood=gpytorch.likelihoods.GaussianLikelihood()
                projection_dim = 5

                if self.inducing_pt_init_w_moss23:
                    weight_function = StandardWeight(likelihood.noise.to(self.device))
                    self.model = RCaGP_DPP(
                        train_inputs=self.train_x.to(self.device),
                        train_targets=self.normed_train_y.squeeze().to(self.device),
                        weight_function=weight_function,
                        mean_module=mean_module,
                        covar_module=covar_module,
                        likelihood=likelihood.to(self.device),
                        inducing_points=inducing_points.to(self.device)
                    ).to(self.device)

                    optimal_inducing_points = get_optimal_inducing_points(
                        model=self.model,
                        prev_inducing_points=inducing_points.to(self.device),
                    )

                    self.model = RCaGP_DPP(
                        train_inputs=self.train_x.to(self.device),
                        train_targets=self.normed_train_y.squeeze().to(self.device),
                        weight_function=weight_function,
                        mean_module=mean_module,
                        covar_module=covar_module,
                        likelihood=likelihood.to(self.device),
                        inducing_points=optimal_inducing_points.to(self.device),
                    ).to(self.device)
                    self.initial_inducing_points = copy.deepcopy(optimal_inducing_points)

                else:                
                    self.model = CaGP(
                        train_inputs=self.train_x.to(self.device),
                        train_targets=self.normed_train_y.squeeze().to(self.device),
                        mean_module=mean_module,
                        covar_module=covar_module,
                        likelihood=likelihood.to(self.device),
                        projection_dim=projection_dim,
                        initialization="random",
                    ).to(self.device)

            else:               
                mean_module = gpytorch.means.ConstantMean()
                const=0.0
                mean_module.constant = torch.tensor(const)
                for param in mean_module.parameters():
                    param.requires_grad = False
                
                covar_module = gpytorch.kernels.ScaleKernel(LinopMaternKernel(lengthscale_prior=lengthscale_prior))
                likelihood=gpytorch.likelihoods.GaussianLikelihood()
                weight_function = StandardWeight(likelihood.noise.to(self.device))

                '''
                self.model = RCSVGP(
                    inducing_points.to(self.device),
                    self.train_x.to(self.device), 
                    self.normed_train_y.squeeze().to(self.device),
                    mean_module,
                    covar_module,
                    likelihood.to(self.device),
                    weight_function,
                    learn_inducing_locations=learn_inducing_locations,
                ).to(self.device)
                '''

                self.model = SVGP(
                    mean_module=mean_module,
                    covar_module=covar_module,
                    inducing_points=inducing_points.to(self.device),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood().to(self.device),
                    learn_inducing_locations=learn_inducing_locations,
                ).to(self.device)
                
                if self.inducing_pt_init_w_moss23:
                    optimal_inducing_points = get_optimal_inducing_points(
                        model=self.model,
                        prev_inducing_points=inducing_points.to(self.device),
                    )
                    self.model = RCSVGP(
                        optimal_inducing_points.to(self.device),
                        self.train_x.to(self.device),
                        self.normed_train_y.squeeze().to(self.device),
                        mean_module,
                        covar_module,
                        likelihood.to(self.device),
                        weight_function,
                        learn_inducing_locations=learn_inducing_locations,
                    ).to(self.device)
                    self.initial_inducing_points = copy.deepcopy(optimal_inducing_points)
                
    def grab_data_for_update(self,):
        if not self.init_training_complete:
            x_update_on = self.train_x
            normed_y_update_on = self.normed_train_y.squeeze()
            
            if self.train_x.shape[0] >= (self.update_on_n_pts + self.head_idx) :
                self.init_training_complete = True
        else:
            x_update_on = self.train_x[-self.update_on_n_pts:]
            head_x = self.train_x[:self.head_idx]
            x_update_on = torch.cat((head_x, x_update_on), dim=0)

            normed_y_update_on = self.normed_train_y.squeeze()[-self.update_on_n_pts:]
            head_normed_y = self.normed_train_y.squeeze()[:self.head_idx]
            normed_y_update_on = torch.cat((head_normed_y, normed_y_update_on))
        return x_update_on, normed_y_update_on

    def run(self):
        ''' 
        Main optimization loop
        '''

        #iteration = self.objective.num_calls
        update_every = 20

        #while iteration < self.max_n_oracle_calls:
        while self.objective.num_calls < self.max_n_oracle_calls:    
            #iteration += 1
            iteration = self.objective.num_calls

            # Update wandb with optimization progress
            best_score_found = self.train_y.max().item()
            best_score_contaminated_found = self.contaminated_train_y.max().item()
            #n_calls_ = iteration
            n_calls_ = self.objective.num_calls

            print(f"After {n_calls_} oracle calls, Best reward = {best_score_found}")
            
            # Log data to wandb on each loop
            dict_log = {
                "best_found":best_score_found,
                "n_oracle_calls":n_calls_,
                "best_contaminated_found":best_score_contaminated_found,
            }
            self.tracker.log(dict_log)

            # Update model on data collected
            if self.exact_gp_baseline:
                # re-init exact gp model
                self.model = SingleTaskGP(
                    self.train_x,
                    self.normed_train_y,
                    covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood().to(self.device),
                )
                exact_gp_mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
                # fit model to data
                fit_gpytorch_mll(exact_gp_mll)
            
            else:
                if self.eulbo:
                    n_epochs_elbo = self.n_warm_start_epochs
                    train_to_convergence_elbo = False
                else:
                    n_epochs_elbo = self.n_update_epochs
                    train_to_convergence_elbo = True
                x_update_on, normed_y_update_on = self.grab_data_for_update()

                if self.rcsvgp_baseline:
                    if self.informative_mean and iteration % update_every == 0:
                        old_informative_mean = self.model.mean_module
                        sigma_sq_correction = old_informative_mean.sigma_sq_correction

                        #fetching the inliers, outliers, and corrections
                        len_y = self.contaminated_train_y.shape[0]
                        condition = len_y - self.update_on_n_pts
                        
                        head_outliers_idx = self.outliers_idx[self.outliers_idx < self.head_idx]
                        head_mask = torch.ones(self.head_idx, dtype=torch.bool)
                        head_mask[head_outliers_idx] = False
                        
                        tail_outliers_idx = self.outliers_idx[self.outliers_idx >= condition] - condition
                        tail_mask = torch.ones(self.update_on_n_pts, dtype=torch.bool)
                        tail_mask[tail_outliers_idx] = False

                        head_x = x_update_on[:self.head_idx]
                        tail_x = x_update_on[-self.update_on_n_pts:] 
                        head_x_inliers = head_x[head_mask]
                        tail_x_inliers = tail_x[tail_mask]
                        x_inliers = torch.cat((head_x_inliers, tail_x_inliers), dim=-2)

                        head_x_outliers = head_x[head_outliers_idx]
                        tail_x_outliers = tail_x[tail_outliers_idx]
                        x_outliers = torch.cat((head_x_outliers, tail_x_outliers), dim=-2)

                        head_y = self.contaminated_train_y[:self.head_idx]
                        tail_y = self.contaminated_train_y[-self.update_on_n_pts:]
                        head_y_inliers = head_y[head_mask]
                        tail_y_inliers = tail_y[tail_mask]
                        y_inliers = torch.cat((head_y_inliers, tail_y_inliers), dim=-2)

                        head_y_outliers = head_y[head_outliers_idx]
                        tail_y_outliers = tail_y[tail_outliers_idx]
                        y_outliers = torch.cat((head_y_outliers, tail_y_outliers), dim=-2)

                        head_y_true = self.train_y[:self.head_idx][head_outliers_idx]
                        tail_y_true = self.train_y[-self.update_on_n_pts:][tail_outliers_idx]
                        y_true = torch.cat((head_y_true, tail_y_true), dim=0)

                        corrections = simulate_expert_correction(y_true.to(self.device), sigma_sq=self.sigma_sq)
                        
                        const = old_informative_mean.const
                        
                        if len(corrections) > 0:
                            mean_module = InformativeMeanPrior(
                                x_inliers.to(self.device),
                                x_outliers.to(self.device),
                                y_inliers.to(self.device),
                                y_outliers.to(self.device),
                                corrections.to(self.device),
                                self.train_y_mean,
                                self.train_y_std,
                                sigma_sq_correction,
                                const=const,
                            ).to(self.device)
                        else:
                            mean_module = self.model.mean_module
                    
                    else:
                        mean_module = self.model.mean_module

                    inducing_points = self.model.inducing_points
                    covar_module = self.model.covar_module
                    likelihood = self.model.likelihood.to(self.device)
                    beta = self.model.weight_function.beta
                    C = self.model.weight_function.C
                    weight_function = IMQ(mean_module, beta, C)
                    learn_inducing_locations = self.model.learn_inducing_locations

                    self.model = RCSVGP(
                        inducing_points,
                        x_update_on.to(self.device), 
                        normed_y_update_on.to(self.device),
                        mean_module,
                        covar_module,
                        likelihood,
                        weight_function,
                        learn_inducing_locations=learn_inducing_locations,
                    ).to(self.device)

                    model_mll = RCSVGPVariationalELBO(
                        likelihood,
                        self.model
                    )

                elif self.rcagp_baseline:

                    if self.informative_mean and iteration % update_every == 0:                      
                        
                        old_informative_mean = self.model.mean_module
                        sigma_sq_correction = old_informative_mean.sigma_sq_correction

                        #fetching the inliers, outliers, and corrections
                        len_y = self.contaminated_train_y.shape[0]
                        condition = len_y - self.update_on_n_pts

                        head_outliers_idx = self.outliers_idx[self.outliers_idx < self.head_idx]
                        head_mask = torch.ones(self.head_idx, dtype=torch.bool)
                        head_mask[head_outliers_idx] = False

                        tail_outliers_idx = self.outliers_idx[self.outliers_idx >= condition] - condition
                        tail_mask = torch.ones(self.update_on_n_pts, dtype=torch.bool)
                        tail_mask[tail_outliers_idx] = False

                        head_x = x_update_on[:self.head_idx]
                        tail_x = x_update_on[-self.update_on_n_pts:] 
                        head_x_inliers = head_x[head_mask]
                        tail_x_inliers = tail_x[tail_mask]
                        x_inliers = torch.cat((head_x_inliers, tail_x_inliers), dim=0)
                        
                        head_x_outliers = head_x[head_outliers_idx]
                        tail_x_outliers = tail_x[tail_outliers_idx]
                        x_outliers = torch.cat((head_x_outliers, tail_x_outliers), dim=0)

                        head_y = self.contaminated_train_y[:self.head_idx]
                        tail_y = self.contaminated_train_y[-self.update_on_n_pts:]
                        head_y_inliers = head_y[head_mask]
                        tail_y_inliers = tail_y[tail_mask]
                        y_inliers = torch.cat((head_y_inliers, tail_y_inliers), dim=0)

                        head_y_outliers = head_y[head_outliers_idx]
                        tail_y_outliers = tail_y[tail_outliers_idx]
                        y_outliers = torch.cat((head_y_outliers, tail_y_outliers), dim=0)

                        head_y_true = self.train_y[:self.head_idx][head_outliers_idx]
                        tail_y_true = self.train_y[-self.update_on_n_pts:][tail_outliers_idx]
                        y_true = torch.cat((head_y_true, tail_y_true), dim=0)
                        corrections = simulate_expert_correction(y_true.to(self.device), sigma_sq=self.sigma_sq)

                        const = old_informative_mean.const

                        if corrections.shape[0] > 0:
                            mean_module = InformativeMeanPrior(
                                x_inliers.to(self.device),
                                x_outliers.to(self.device),
                                y_inliers.to(self.device),
                                y_outliers.to(self.device),
                                corrections.to(self.device),
                                self.train_y_mean,
                                self.train_y_std,
                                sigma_sq_correction=sigma_sq_correction,
                                const=const,
                            ).to(self.device)
                        else:
                            mean_module = self.model.mean_module
                        
                    else:
                        mean_module = self.model.mean_module
                    
                    covar_module = self.model.covar_module
                    likelihood = self.model.likelihood.to(self.device)

                    beta = self.model.weight_function.beta
                    C = self.model.weight_function.C
                    weight_function = IMQ(mean_module, beta, C)

                    if self.moss23_baseline:
                        inducing_points = self.model.inducing_points
                        self.model = RCaGP_DPP(
                            train_inputs=x_update_on.to(self.device),
                            train_targets=normed_y_update_on.to(self.device),
                            weight_function=weight_function,
                            mean_module=mean_module,
                            covar_module=covar_module,
                            likelihood=likelihood.to(self.device),
                            inducing_points=inducing_points.to(self.device),
                        ).to(self.device)                       
                        
                        model_mll = RCaGP_DPPVariationalELBO(
                            likelihood,
                            self.model
                        )
                    else:
                        projection_dim = self.model.projection_dim
                        self.model = RCaGP(
                            train_inputs=x_update_on.to(self.device),
                            train_targets=normed_y_update_on.to(self.device),
                            weight_function=weight_function,
                            mean_module=mean_module,
                            covar_module=covar_module,
                            likelihood=likelihood,
                            projection_dim=projection_dim,
                            initialization="random",
                        ).to(self.device)
                        
                        model_mll = RCaGPVariationalELBO(
                            likelihood,
                            self.model
                        )
                
                elif self.cagp_baseline:
                    mean_module = self.model.mean_module
                    covar_module = self.model.covar_module
                    likelihood = self.model.likelihood.to(self.device)

                    if self.moss23_baseline:
                        inducing_points = self.model.inducing_points
                        weight_function = self.model.weight_function
                        self.model = RCaGP_DPP(
                            train_inputs=x_update_on.to(self.device),
                            train_targets=normed_y_update_on.to(self.device),
                            weight_function=weight_function,
                            mean_module=mean_module,
                            covar_module=covar_module,
                            likelihood=likelihood.to(self.device),
                            inducing_points=inducing_points.to(self.device),
                        ).to(self.device)  

                        model_mll = RCaGP_DPPVariationalELBO(
                            likelihood,
                            self.model
                        )
                    else:
                        projection_dim = self.model.projection_dim
                        self.model = CaGP(
                            train_inputs=x_update_on.to(self.device),
                            train_targets=normed_y_update_on.to(self.device),
                            mean_module=mean_module,
                            covar_module=covar_module,
                            likelihood=likelihood,
                            projection_dim=projection_dim,
                            initialization="random",
                        ).to(self.device)
                        
                        model_mll = CaGPVariationalELBO(
                            likelihood,
                            self.model
                        )

                else:
                    model_mll = None 

                update_model_dict = update_model_elbo(
                        model=self.model,
                        train_x=x_update_on,
                        train_y=normed_y_update_on,
                        lr=self.lr,
                        mll=model_mll,
                        n_epochs=n_epochs_elbo,
                        train_bsz=self.train_bsz,
                        grad_clip=self.grad_clip,
                        train_to_convergence=train_to_convergence_elbo,
                        max_allowed_n_failures_improve_loss=self.max_allowed_n_failures_improve_loss,
                        max_allowed_n_epochs=self.max_allowed_n_epochs,
                        moss23_baseline=self.moss23_baseline,
                        ppgpr=self.ppgpr,
                        rcagp_baseline=self.rcagp_baseline,
                        rcsvgp_baseline=self.rcsvgp_baseline,
                        cagp_baseline=self.cagp_baseline,
                    )
                self.model = update_model_dict["model"]
            
            # Generate a batch of candidates 
            x_next = generate_batch(
                model=self.model,
                X=self.train_x,  
                Y=self.normed_train_y,
                batch_size=self.bsz,
                acqf=self.acq_func,
                device=self.device,
                absolute_bounds=(self.objective.lb, self.objective.ub),
                use_turbo=self.use_turbo,
                tr_length=self.tr_state.length,
                dtype=self.dtype,
            )
            
            # If using EULBO, use above model update and candidate generaiton as warm start
            if self.eulbo:
                lb = self.objective.lb
                ub = self.objective.ub
                update_model_dict = update_model_and_generate_candidates_eulbo(
                    model=self.model,
                    train_x=x_update_on,
                    train_y=normed_y_update_on,
                    lb=lb,
                    ub=ub,
                    lr=self.lr,
                    mll=model_mll,
                    n_epochs=self.n_update_epochs,
                    train_bsz=self.train_bsz,
                    grad_clip=self.grad_clip,
                    normed_best_f=self.normed_train_y.max(),
                    acquisition_bsz=self.bsz,
                    max_allowed_n_failures_improve_loss=self.max_allowed_n_failures_improve_loss,
                    max_allowed_n_epochs=self.max_allowed_n_epochs,
                    init_x_next=x_next,
                    x_next_lr=self.x_next_lr,
                    alternate_updates=self.alternate_eulbo_updates,
                    num_kg_samples=self.num_kg_samples,
                    use_kg=self.use_kg,
                    dtype=self.dtype,
                    num_mc_samples_qei=self.num_mc_samples_qei,
                    ablation1_fix_indpts_and_hypers=self.ablation1_fix_indpts_and_hypers,
                    ablation2_fix_hypers=self.ablation2_fix_hypers,
                    use_turbo=self.use_turbo,
                    tr_length=self.tr_state.length,
                    use_botorch_stable_log_softplus=self.use_botorch_stable_log_softplus,
                    ppgpr=self.ppgpr,
                    rcagp_baseline=self.rcagp_baseline,
                    rcsvgp_baseline=self.rcsvgp_baseline,
                    cagp_baseline=self.cagp_baseline,
                )
                self.model = update_model_dict["model"]
                x_next = update_model_dict["x_next"]

            # Evaluate candidates
            y_next = self.objective(x_next)
            len_data_before_update = len(self.train_y)

            # Update data
            self.train_x = torch.cat((self.train_x, x_next), dim=-2)
            self.train_y = torch.cat((self.train_y, y_next), dim=-2)

            # Contaminate data
            normed_y_next = (y_next - self.train_y_mean) / self.train_y_std
            noise_y_next, noise_idx_y_next, noise_mask_y_next = self.objective.get_contamination(normed_y_next, seed=n_calls_, probability=self.p_outliers)
            contaminated_normed_y_next = self.objective.inject_contamination(normed_y_next, self.normed_train_y_std, noise_y_next, noise_mask_y_next, low=self.lb_outliers, high=self.ub_outliers)
            contaminated_y_next = contaminated_normed_y_next * self.train_y_std + self.train_y_mean

            # Update unnormalized contaminated and normed y
            self.contaminated_train_y = torch.cat((self.contaminated_train_y, contaminated_y_next), dim=0)
            self.normed_train_y = torch.cat((self.normed_train_y, contaminated_normed_y_next), dim=0)

            # Update outliers idx (for expert)
            if len(noise_idx_y_next) > 0:
                noise_idx_y_next = noise_idx_y_next + len_data_before_update
                self.outliers_idx = torch.cat((self.outliers_idx, noise_idx_y_next))

            # If running TuRBO, update trust region state
            if self.use_turbo:
                self.tr_state = update_state(
                    state=self.tr_state,
                    Y_next=y_next,
                )
                if self.tr_state.restart_triggered:
                    self.tr_state = TurboState(
                        dim=self.train_x.shape[-1],
                        batch_size=self.bsz,
                        best_value=self.train_y.max().item(),
                    )
        self.tracker.finish()
        return self

    def handler(self, signum, frame):
        # If we Ctrl-c, make sure we terminate wandb tracker
        print("Ctrl-c hass been pressed, terminating wandb tracker...")
        self.tracker.finish()
        msg = "tracker terminated, now exiting..."
        print(msg, end="", flush=True)
        exit(1)

    def done(self):
        return None

def new(**kwargs):
    return Optimize(**kwargs)

if __name__ == "__main__":
    fire.Fire(Optimize)