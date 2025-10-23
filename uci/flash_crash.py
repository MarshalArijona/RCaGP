import sys
import os
import argparse

sys.path.append("../")

import gpytorch
import torch
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, HourLocator
from uci.flash_utils import MyScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.variational.variational_ELBO import CaGPVariationalELBO, RCaGPVariationalELBO, RCSVGPVariationalELBO, RCGPVariationalELBO, RRPVariationalELBO, RRPVariationalELBOdiff
from src.models.SVGP import SVGP
from src.models.RCSVGP import RCSVGP
from src.models.RCaGP import RCaGP, CaGP

from utils.linop_matern_kernel import LinopMaternKernel
from utils.linop_rbf_kernel import LinopRBFKernel
from utils.weight_function import IMQ, StandardWeight

from torch.utils.data import TensorDataset, DataLoader
#from tueplots import bundles

def plot_and_save_report(
                        tensor_data,
                        tensor_dates,
                        data, 
                        dates,
                        model_RCaGP,
                        model_RCSVGP,
                        mean_RCaGP,
                        mean_SVGP,
                        mean_RCSVGP,
                        mean_CaGP,
                        ):

    dates = pd.to_datetime(dates.flatten())
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    #plt.rcParams['text.usetex'] = False
    CB_color_cycle = ['blue', 'brown', 'grey',
                    'cyan', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']
    
    #with plt.rc_context(bundles.aistats2023()):
    fig, ax = plt.subplots(
        ncols=2,
        sharex=True,
        figsize=(6.75, 2.5),  # (width, height) in inches
        gridspec_kw={'width_ratios': [1, 1]}
    )
    
    ax[0].plot(dates, mean_RCaGP, c=CB_color_cycle[0], lw=2, label= 'RCaGP')
    ax[0].plot(dates, mean_SVGP, c=CB_color_cycle[1], lw=2, label= 'SVGP')
    ax[0].plot(dates, mean_RCSVGP, c=CB_color_cycle[2], lw=2, label= 'RCSVGP')
    ax[0].plot(dates, mean_CaGP, c=CB_color_cycle[3], lw=2, label= 'CaGP')
    
    ax[0].hlines(np.median(data),dates[0],dates[-1], label='Prior mean', lw=2, ls='--', alpha=0.5, color='green')

    mask = np.ones(data.shape[0], bool)
    mask[[109,110]] = False
    ax[0].set_ylabel('DJIA index')
    ax[0].plot(dates[mask], data[mask], 'k.', ms=1, alpha=1.0)

    ax[0].scatter(dates[[109,110]], data[[109,110]], marker = 'x', alpha=1, color= 'r', s=10, label = 'Outliers')
    ax[0].legend(loc='lower right', fontsize=5)
    
    w_RCaGP = model_RCaGP.weight_function.W(tensor_dates, tensor_data).detach().cpu().numpy()
    w_RCSVGP = model_RCSVGP.weight_function.W(tensor_dates, tensor_data).detach().cpu().numpy()

    ax[1].plot(dates, w_RCaGP, lw=2, color=CB_color_cycle[0])
    ax[1].plot(dates, w_RCSVGP, lw=2, color=CB_color_cycle[2])
    ax[1].set_ylabel('w(x, y)')

    # --- Date axis formatting (applies to both)
    hour_locator = HourLocator(interval=1)
    formatter = DateFormatter("%H:%M")
    for a in ax:
        a.set_xlim(dates[0], dates[-1])
        a.xaxis.set_major_locator(hour_locator)
        a.xaxis.set_major_formatter(formatter)
        for label in a.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

    fig.text(0.5, 0.03, 'Time', ha='center', va='center')

    # --- Final layout adjustments
    plt.tight_layout(pad=0.2, rect=[0, 0.05, 1, 1])
    fig.savefig("flashcrash.pdf", bbox_inches="tight", dpi=600)
    plt.close(fig)

def plot_and_save_GP(dates, data, f_mean, f_std, model_name, color_idx):
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']
                    
    plt.plot(dates[:,0], f_mean, c=CB_color_cycle[color_idx], lw=2, label= model_name)
    plt.fill_between(
        dates[:,0],
        f_mean - 1.96 * f_std,
        f_mean + 1.96 * f_std,
        facecolor= CB_color_cycle[color_idx],
        alpha=0.5,
        label='95% IC')
    plt.plot(dates, data, 'k.', ms=7, alpha=0.5, label = 'Observations')
    plt.xlim(dates[0], dates[-1])
    plt.legend(ncol=4, frameon=True, shadow=False, loc=9, edgecolor='k')
    plt.tight_layout()

    plt.savefig(model_name, format='pdf')
    plt.close()


def train_approximateGP(train_loader, model_name, model, likelihood, mll, device):
    
    model = model.to(device)
    model.train()
    likelihood.train()
    

    #Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            
            if model_name == "SVGP":
                loss = -mll(output, y_batch.squeeze())
            else:
                loss = -mll(output)

            loss.backward()
            optimizer.step()

def predict_approximateGP(model, likelihood, X_test, y_test, device):
    model.eval()
    likelihood.eval()
        
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(torch.tensor(X_test).to(device)))
        mean = preds.mean
        var = preds.variance
        #nll = - preds.log_prob(torch.tensor(y_test).to(device))
        nll = torch.mean(0.5 * torch.log(2 * torch.pi * var[50:100].clone()) + 0.5 * ((torch.tensor(y_test[50:100].clone()).to(device) - mean[50:100].clone())**2 / var[50:100].clone())).item()
        mae = torch.mean(torch.absolute(mean[50:100].clone() - torch.tensor(y_test[50:100].clone()).to(device))).item()

    return mean, var, nll, mae

def run_experiment(
                   device,  
                   epsilon,
                   likelihood_type,
                   kernel_type,
                   projection_dim,
                   beta):

    DJI = pd.read_csv('DJI.txt', header = None)
    DJI.columns =['date', 'open', 'high', 'low', 'close']
    DJI['date'] = pd.to_datetime(DJI['date'], infer_datetime_format=True)
    flash_crash_dji = DJI[(DJI['date'].dt.date.astype(str) == '2013-04-23')]
                        
    data = flash_crash_dji['open'].values.reshape(-1,1)[1::2]
    dates = flash_crash_dji['date'].values.reshape(-1,1)[1::2]

    arr1inds = dates[:,0].argsort()
    data = data
    dates = dates

    print("data")
    print(data.mean())
    print(data.std())

    print("outliers")
    print(data[[109,110]])

    dates_float = dates.astype(np.float64)
    x_scaler = StandardScaler()
    y_scaler = MyScaler()
    dates_normalised = x_scaler.fit_transform(dates_float)
    y_scaler.fit(data)
    data_normalised = y_scaler.transform(data)

    train_x = torch.tensor(dates_normalised).to(device)
    train_y = torch.tensor(data_normalised).to(device)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=len(data_normalised), shuffle=True)

    C = torch.quantile(train_y, 1 - epsilon)
    n_inducing_pts = 100
    inducing_points = train_x[0: n_inducing_pts,:].clone()
    learn_inducing_locations = True
    const = train_y.mean()
    #----------------------------------------------------------------------------------------------------------
    if likelihood_type == "gaussian":
        likelihood_RCaGP = gpytorch.likelihoods.GaussianLikelihood().to(device)
    else:
        likelihood_RCaGP = gpytorch.likelihoods.StudentTLikelihood().to(device)
    
    mean_module_RCaGP = gpytorch.means.ConstantMean().to(device)        
    
    if kernel_type == "matern":
        covar_module_RCaGP = gpytorch.kernels.ScaleKernel(LinopMaternKernel())
    else:
        covar_module_RCaGP = gpytorch.kernels.ScaleKernel(LinopRBFKernel())
    
    mean_module_RCaGP.constant = torch.tensor(const)
    weight_function_RCaGP = IMQ(mean_module_RCaGP, beta, C)

    model_RCaGP = RCaGP(
            train_inputs=train_x.to(device),
            train_targets=train_y.squeeze().to(device),
            weight_function=weight_function_RCaGP,
            mean_module=mean_module_RCaGP,
            covar_module=covar_module_RCaGP,
            likelihood=likelihood_RCaGP.to(device),
            projection_dim=projection_dim,
            initialization="random",).  to(device)

    mll_RCaGP = RCaGPVariationalELBO(likelihood_RCaGP, model_RCaGP)
    train_approximateGP(train_loader, "RCaGP", model_RCaGP, likelihood_RCaGP, mll_RCaGP, device)
    mean_RCaGP, var_RCaGP, nll_RCaGP, mae_RCaGP = predict_approximateGP(model_RCaGP, likelihood_RCaGP, train_x, train_y, device)
    
    print("RCaGP")
    print(nll_RCaGP)
    print(mae_RCaGP)
    print()

    plot_and_save_GP(dates_normalised, data_normalised, mean_RCaGP.cpu().numpy(), torch.sqrt(var_RCaGP).cpu().numpy(), "RCaGP", 0)
    #--------------------------------------------------------------------------------------------------------------
    
    if likelihood_type == "gaussian":
        likelihood_SVGP = gpytorch.likelihoods.GaussianLikelihood().to(device)
    else:
        likelihood_SVGP = gpytorch.likelihoods.StudentTLikelihood().to(device)
    
    mean_module_SVGP = gpytorch.means.ConstantMean().to(device)        
    
    if kernel_type == "matern":
        covar_module_SVGP = gpytorch.kernels.ScaleKernel(LinopMaternKernel())
    else:
        covar_module_SVGP = gpytorch.kernels.ScaleKernel(LinopRBFKernel())
    
    mean_module_SVGP.constant = torch.tensor(const)

    model_SVGP = SVGP(
                mean_module_SVGP,
                covar_module_SVGP,
                inducing_points.to(device), 
                likelihood_SVGP.to(device), 
                learn_inducing_locations=True).to(device)
    
    mll_SVGP = gpytorch.mlls.VariationalELBO(likelihood_SVGP, model_SVGP, num_data=train_x.size(0))
    train_approximateGP(train_loader, "SVGP", model_SVGP, likelihood_SVGP, mll_SVGP, device)
    mean_SVGP, var_SVGP, nll_SVGP, mae_SVGP = predict_approximateGP(model_SVGP, likelihood_SVGP, train_x, train_y, device)
    
    print("SVGP")
    print(nll_SVGP)
    print(mae_SVGP)
    print()

    plot_and_save_GP(dates_normalised, data_normalised, mean_SVGP.cpu().numpy(), torch.sqrt(var_SVGP).cpu().numpy(), "SVGP", 1)
    #--------------------------------------------------------------------------------------------------------------
    
    if likelihood_type == "gaussian":
        likelihood_RCSVGP = gpytorch.likelihoods.GaussianLikelihood().to(device)
    else:
        likelihood_RCSVGP = gpytorch.likelihoods.StudentTLikelihood().to(device)
    
    mean_module_RCSVGP = gpytorch.means.ConstantMean().to(device)       
    
    if kernel_type == "matern":
        covar_module_RCSVGP = gpytorch.kernels.ScaleKernel(LinopMaternKernel())
    else:
        covar_module_RCSVGP = gpytorch.kernels.ScaleKernel(LinopRBFKernel())
    
    mean_module_RCSVGP.constant = torch.tensor(const)

    weight_function_RCSVGP = IMQ(mean_module_RCSVGP, beta, C)

    model_RCSVGP = RCSVGP(
            inducing_points.to(device),
            train_x.to(device), 
            train_y.squeeze().to(device),
            mean_module_RCSVGP,
            covar_module_RCSVGP,
            likelihood_RCSVGP.to(device),
            weight_function_RCSVGP,
            learn_inducing_locations=learn_inducing_locations
        ).to(device)
    
    mll_RCSVGP = RCSVGPVariationalELBO(likelihood_RCSVGP, model_RCSVGP)
    train_approximateGP(train_loader, "RCSVGP", model_RCSVGP, likelihood_RCSVGP, mll_RCSVGP, device)
    mean_RCSVGP, var_RCSVGP, nll_RCSVGP, mae_RCSVGP = predict_approximateGP(model_RCSVGP, likelihood_RCSVGP, train_x, train_y, device)

    print("RCSVGP")
    print(nll_RCSVGP)
    print(mae_RCSVGP)
    print()

    plot_and_save_GP(dates_normalised, data_normalised, mean_RCSVGP.cpu().numpy(), torch.sqrt(var_RCSVGP).cpu().numpy(), "RCSVGP", 2)
    #--------------------------------------------------------------------------------------------------------------
    
    if likelihood_type == "gaussian":
        likelihood_CaGP = gpytorch.likelihoods.GaussianLikelihood().to(device)
    else:
        likelihood_CaGP = gpytorch.likelihoods.StudentTLikelihood().to(device)
    
    mean_module_CaGP = gpytorch.means.ConstantMean().to(device)        
    
    if kernel_type == "matern":
        covar_module_CaGP = gpytorch.kernels.ScaleKernel(LinopMaternKernel())
    else:
        covar_module_CaGP = gpytorch.kernels.ScaleKernel(LinopRBFKernel())
    
    mean_module_CaGP.constant = torch.tensor(const)
    
    model_CaGP = CaGP(
            train_inputs=train_x.to(device),
            train_targets=train_y.squeeze().to(device),
            mean_module=mean_module_CaGP,
            covar_module=covar_module_CaGP,
            likelihood=likelihood_CaGP.to(device),
            projection_dim=projection_dim,
            initialization="random",
        ).to(device)
    
    mll_CaGP = CaGPVariationalELBO(likelihood_CaGP, model_CaGP)
    train_approximateGP(train_loader, "CaGP", model_CaGP, likelihood_CaGP, mll_CaGP, device)
    mean_CaGP, var_CaGP, nll_CaGP, mae_CaGP = predict_approximateGP(model_CaGP, likelihood_CaGP, train_x, train_y, device)
    
    print("CaGP")
    print(nll_CaGP)
    print(mae_CaGP)
    print()

    plot_and_save_GP(dates_normalised, data_normalised, mean_CaGP.cpu().numpy(), torch.sqrt(var_CaGP).cpu().numpy(), "CaGP", 3)
    #--------------------------------------------------------------------------------------------------------------

    print("mean")
    print(mean_RCaGP[[109, 110]].cpu().numpy())
    print(mean_SVGP[[109, 110]].cpu().numpy())
    print(mean_RCSVGP[[109, 110]].cpu().numpy())
    print(mean_CaGP[[109, 110]].cpu().numpy())

    print("variance")
    print(var_RCaGP[[109, 110]].cpu().numpy())
    print(var_SVGP[[109, 110]].cpu().numpy())
    print(var_RCSVGP[[109, 110]].cpu().numpy())
    print(var_CaGP[[109, 110]].cpu().numpy())  

    plot_and_save_report(train_y,
                        train_x,
                        data_normalised, 
                        dates,
                        model_RCaGP,
                        model_RCSVGP,
                        mean_RCaGP.cpu().numpy(),
                        mean_SVGP.cpu().numpy(),
                        mean_RCSVGP.cpu().numpy(),
                        mean_CaGP.cpu().numpy())
   
def main():
    """
    Main function to execute the experiment.
    """
    parser = argparse.ArgumentParser(description="Run GPyTorch regression experiments on UCI datasets.")
    parser.add_argument("--seeds", type=int,
                        help="Random seeds for experiments")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--likelihood_type", type=str, default="gaussian", choices=["gaussian", "student-t"])
    parser.add_argument("--kernel_type", type=str, default="matern", choices=["rbf", "matern"])
    parser.add_argument("--projection_dim", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    run_experiment(
        device=device,
        epsilon=args.epsilon,
        likelihood_type=args.likelihood_type,
        kernel_type=args.kernel_type,
        projection_dim=args.projection_dim,
        beta=args.beta
    )

if __name__ == "__main__":
    main()