import torch
import gpytorch
import numpy as np
import random
import time
import argparse
import pandas as pd

import sys 
sys.path.append("../")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader

from src.variational.variational_ELBO import CaGPVariationalELBO, RCaGPVariationalELBO, RCSVGPVariationalELBO
from src.models.RCSVGP import RCSVGP
from src.models.RCaGP import RCaGP, CaGP
from utils.weight_function import IMQ, StandardWeight
from utils.linop_matern_kernel import LinopMaternKernel
from utils.informative_mean import InformativeMeanPrior, simulate_expert_correction

# --- Custom PyTorch Dataset ---
class UCI_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- DATA LOADING ---
def load_uci_dataset(name):
    """
    Loads dataset from UCI repository based on the name.
    """
    if name == "boston":
        url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
        df = pd.read_csv(url)
        X = df.drop('medv', axis=1).values
        y = df['medv'].values
        projection_dim = 5
        n_inducing_pts = 100
    
    elif name == "energy":
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
        df = pd.read_excel(url)
        X = df.iloc[:, :-2].values
        y = df.iloc[:, -2].values  # Heating Load (or use -1 for Cooling Load)
        projection_dim = 5
        n_inducing_pts = 100

    elif name == "yacht":
        url = 'yacht/yacht.data'
        df = pd.read_csv(url, sep=r'\s+', header=None)
        df = df.dropna()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        projection_dim = 5
        n_inducing_pts = 100

    elif name == "parkinsons":
        # Load the local file
        df = pd.read_csv('parkinsons/parkinsons.data')  # update the path accordingly
        # Drop the 'name' column (it's just an identifier)
        df = df.drop(columns=['name'])
        df = df.drop(columns=['status'])
        
        X = df.drop(columns=['PPE']).values
        y = df['PPE'].values
        projection_dim = 5
        n_inducing_pts = 100

    else:
        raise ValueError("Unknown dataset")

    return X, y, projection_dim, n_inducing_pts

def get_expert_mean_prior(train_x, train_y, outliers_idx, train_y_mean, train_y_std, device, sigma_sq):

    mask = torch.ones(train_x.size(0), dtype=torch.bool)
    mask[outliers_idx] = False
    x_inliers = train_x[mask]
    x_outliers = train_x[outliers_idx]
    
    y_inliers = train_y[mask]
    y_outliers = train_y[outliers_idx]

    Y_corrections = simulate_expert_correction(y_outliers.to(device), 
                                        sigma_sq=sigma_sq)
    shifting_const = 0.0
    mean_module = InformativeMeanPrior(
        x_inliers.to(device),
        x_outliers.to(device),
        y_inliers.to(device),
        y_outliers.to(device),
        Y_corrections.to(device),
        train_y_mean,
        train_y_std,
        sigma_sq_correction=sigma_sq,
        const=shifting_const,
    ).to(device)
    
    return mean_module

# --- GP MODEL SELECTION (Dynamically Import Models) ---
def get_gp_model(model_name, 
                train_x, 
                train_y, 
                outliers_idx, 
                train_y_mean, 
                train_y_std, 
                likelihood, 
                device, 
                sigma_sq, 
                projection_dim, 
                n_inducing_pts,
                raw_y_train,
                epsilon=0.2):
    
    """
    Dynamically imports and returns a GP model instance based on the model name.
    """
    mean_module = gpytorch.means.ConstantMean().to(device)
    covar_module = gpytorch.kernels.ScaleKernel(LinopMaternKernel())
    beta = 1.0
    C = torch.quantile(train_y, 1 - epsilon)
    inducing_points = train_x[0: n_inducing_pts,:].clone()
    learn_inducing_locations = True

    if model_name == "RCaGP_expert": 
        mean_module = get_expert_mean_prior(train_x, 
                                        raw_y_train, 
                                        outliers_idx, 
                                        train_y_mean, 
                                        train_y_std, 
                                        device, 
                                        sigma_sq)
                                        
        weight_function = IMQ(mean_module, beta, C)
        model = RCaGP(
            train_inputs=train_x.to(device),
            train_targets=train_y.squeeze().to(device),
            weight_function=weight_function,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood.to(device),
            projection_dim=projection_dim,
            initialization="random",
        )

        mll = RCaGPVariationalELBO(likelihood, model)
    
    elif model_name == "RCaGP":
        const = train_y.mean()
        mean_module.constant = torch.tensor(const)
        
        weight_function = IMQ(mean_module, beta, C)
        model = RCaGP(
                train_inputs=train_x.to(device),
                train_targets=train_y.squeeze().to(device),
                weight_function=weight_function,
                mean_module=mean_module,
                covar_module=covar_module,
                likelihood=likelihood.to(device),
                projection_dim=projection_dim,
                initialization="random",
        )

        mll = RCaGPVariationalELBO(likelihood, model)
    
    elif model_name == "RCSVGP_expert":
        mean_module = get_expert_mean_prior(train_x, raw_y_train, outliers_idx, train_y_mean, train_y_std, device, sigma_sq)
        weight_function = IMQ(mean_module, beta, C)

        model = RCSVGP(
            inducing_points.to(device),
            train_x.to(device), 
            train_y.squeeze().to(device),
            mean_module,
            covar_module,
            likelihood.to(device),
            weight_function,
            learn_inducing_locations=learn_inducing_locations
        )

        mll = RCSVGPVariationalELBO(likelihood, model)

    elif model_name == "RCSVGP":
        const = train_y.mean()
        mean_module.constant = torch.tensor(const)
        
        weight_function = IMQ(mean_module, beta, C)
        model = RCSVGP(
            inducing_points.to(device),
            train_x.to(device), 
            train_y.squeeze().to(device),
            mean_module,
            covar_module,
            likelihood.to(device),
            weight_function,
            learn_inducing_locations=learn_inducing_locations
        )
        mll = RCSVGPVariationalELBO(likelihood, model)

    elif model_name == "CaGP":    
        model = CaGP(
            train_inputs=train_x.to(device),
            train_targets=train_y.squeeze().to(device),
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood.to(device),
            projection_dim=projection_dim,
            initialization="random",
        )
        mll = CaGPVariationalELBO(likelihood, model)

    else:
        weight_function = StandardWeight(likelihood.noise.to(device))
        model = RCSVGP(
            inducing_points.to(device),
            train_x.to(device), 
            train_y.squeeze().to(device),
            mean_module,
            covar_module,
            likelihood.to(device),
            weight_function,
            learn_inducing_locations=learn_inducing_locations,
        )
        mll = RCSVGPVariationalELBO(likelihood, model)

    return model, mll

def inject_noise(y, y_std, fraction, type):
    y = y.reshape(-1)
    n_data = y.shape[0]

    noise = np.random.uniform(low=3 * y_std, high=9 * y_std, size=n_data)
    mask = (np.random.rand(n_data) < fraction).astype(int)
    outliers_idx = np.where(mask == 1)[0]

    if type == "asymmetric":
        y_contaminated = y + noise * mask
    
    elif type == "uniform":
        rng = np.random.default_rng(seed=42)
        # Get a shuffled copy
        shuffle_y = rng.permutation(y)

        shuffle_y[:n_data // 2] += noise[: n_data // 2] * mask[:n_data // 2]
        shuffle_y[n_data // 2:] -= noise[n_data // 2:] * mask[n_data // 2:]
        y_contaminated = shuffle_y
    else:
        return y, []

    return y_contaminated, outliers_idx

def inject_focused(X, y, fraction):
    y = y.reshape(-1)
    n_data, dim = X.shape

    #noise = np.random.uniform(low=3 * y_std, high=9 * y_std, size=n_data)
    mask = (np.random.rand(n_data) < fraction).astype(int)
    outliers_idx = np.where(mask == 1)[0]
    n_outliers = int(np.sum(mask))
    
    med_x = np.median(X, axis=0).reshape(1, -1)
    noise_x = np.random.rand(n_outliers, dim)
    median_abs_x = np.median(np.absolute(X - med_x), axis=0).reshape(1, -1)
    replace_x = med_x + 0.1 * noise_x * median_abs_x
    X[outliers_idx] = replace_x

    y_median = np.median(y)
    y_abs_median = np.median(np.absolute(y - y_median))
    noise_y = np.random.rand(n_outliers)
    y[outliers_idx] -= 3 * y_median + 0.1 * noise_y * y_abs_median

    return X, y, outliers_idx

# --- SINGLE EXPERIMENT ---
def run_experiment(seed, 
                   dataset, 
                   outlier_params, 
                   device, 
                   model_name, 
                   epsilon,
                   obsvar):
    """
    Runs a single experiment with GP regression on the chosen UCI dataset.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load dataset
    X, y, projection_dim, n_inducing_pts = load_uci_dataset(dataset)
    sigma_sq = torch.tensor(obsvar).to(device)
    y_mean = y.mean()
    y_std = y.std()

    # Standardize features
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    raw_y_train = y_train

    # Inject outliers
    outliers_fraction = outlier_params["fraction"]
    outliers_type = outlier_params["type"]
    if outliers_type == "focused":
        X_train, y_train, outliers_idx = inject_focused(X_train, y_train, outliers_fraction)
    else:
        y_train, outliers_idx = inject_noise(y_train, y_std, outliers_fraction, outliers_type)

    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std 
    batch_size = y_train.shape[0]

    # Create datasets and dataloaders
    train_dataset = UCI_Dataset(X_train, y_train)
    test_dataset = UCI_Dataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # GP model setup
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    
    model, mll = get_gp_model(model_name, 
                        torch.tensor(X_train), 
                        torch.tensor(y_train), 
                        outliers_idx, 
                        torch.tensor(y_mean), 
                        torch.tensor(y_std), 
                        likelihood, 
                        device, 
                        sigma_sq, 
                        projection_dim, 
                        n_inducing_pts,
                        torch.tensor(raw_y_train),
                        epsilon)
    
    model = model.to(device)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start_time = time.time()

    # Training loop
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = -mll(output)
            loss.backward()
            optimizer.step()

    elapsed_time = time.time() - start_time

    # Evaluation
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(torch.tensor(X_test).to(device)))
        nll = - preds.log_prob(torch.tensor(y_test).to(device))
        mean = preds.mean
        var = preds.variance
        nll = torch.mean(0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((torch.tensor(y_test).to(device) - mean)**2 / var)).item()
        mae = torch.mean(torch.absolute(mean - torch.tensor(y_test).to(device))).item()

    return mae, nll, elapsed_time


# --- MAIN ENTRY POINT ---
def main():
    """
    Main function to execute the experiment.
    """
    parser = argparse.ArgumentParser(description="Run GPyTorch regression experiments on UCI datasets.")
    parser.add_argument("--dataset", type=str, default="boston", choices=["boston", "energy", "yacht", "parkinsons"],
                        help="UCI dataset to use")
    parser.add_argument("--seeds", type=int,
                        help="Random seeds for experiments")
    parser.add_argument("--outlier_type", type=str, default="additive", choices=["uniform", "focused", "asymmetric", "noiseless"],
                        help="Type of outlier to inject into training targets")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of outliers to inject")
    parser.add_argument("--models", type=str)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--obsvar", type=float, default=1.0)
    args = parser.parse_args()

    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_rmses, all_nlls, all_times = [], [], []
    model_name = args.models
    
    for seed in range(args.seeds):
        outlier_params = {
            'fraction': args.fraction,
            'type': args.outlier_type
        }
        mae, nll, t = run_experiment(
            seed=seed,
            dataset=args.dataset,
            outlier_params=outlier_params,
            device=device,
            model_name=model_name,
            epsilon=args.epsilon,
            obsvar=args.obsvar
        )
        print(f"[Model {model_name}, Seed {seed}] MAE: {mae:.4f}, NLL: {nll:.4f}, Time: {t:.2f} sec")
        all_rmses.append(mae)
        all_nlls.append(nll)
        all_times.append(t)

    print("\n== Summary ==")
    print(model_name)
    print(str(args.epsilon))
    print(str(args.fraction))
    print(f"MAE: {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f}")
    print(f"NLL:  {np.mean(all_nlls):.4f} ± {np.std(all_nlls):.4f}")
    print(f"Time: {np.mean(all_times):.2f} sec ± {np.std(all_times):.2f} sec")

if __name__ == "__main__":
    main()
