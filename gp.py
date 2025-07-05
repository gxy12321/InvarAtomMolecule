#!/usr/bin/env python3
"""
Optimized Gaussian Process regression without feature/target scaling,
with simple train/test split, adaptive sample augmentation based on worst-case errors,
and kernel experiments. Saves updated datasets and models.
Prerequisites:
  pip install torch gpytorch pandas matplotlib
"""
import os
import numpy as np
import pandas as pd
import torch
import gpytorch
import sys
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, RQKernel, PiecewisePolynomialKernel, PeriodicKernel, LinearKernel,SpectralDeltaKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from lhsampling import load_data

global mode
mode = sys.argv[1]
if mode not in ['log', 'no_log']:
    raise ValueError('Mode is wrong: either "log" or "no_log"')
print(mode)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel)
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

def make_kernels(dim: int) -> dict:
    return {
        'Matern+RBF+RQ+Linear': MaternKernel(nu=1.5, ard_num_dims=dim) + MaternKernel(nu=2.5, ard_num_dims=dim) + RBFKernel(ard_num_dims=dim) + RQKernel(ard_num_dims=dim) + LinearKernel(ard_num_dims=dim),

    }


def train_evaluate_augment(
    name: str,
    kernel: gpytorch.kernels.Kernel,
    X_np: np.ndarray,
    y_np: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: str,
    data_dir: str,
    n_iter: int = 10000
):
    """
    Train GP, evaluate RMSE, augment train set with worst-case errors,
    retrain, and save datasets and model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # Initial train/test split
    X_train_np, y_train_np = X_np[train_idx], y_np[train_idx]
    X_test_np,  y_test_np  = X_np[test_idx],  y_np[test_idx]

    def tensorify(x): return torch.from_numpy(x).to(device)
    X_train, y_train = tensorify(X_train_np), tensorify(y_train_np)
    X_test,  y_test  = tensorify(X_test_np),  tensorify(y_test_np)

    print(f"Data loaded: {X_train.size(0)} train, {X_test.size(0)} test on {device}")

    # Build model
    likelihood = GaussianLikelihood().to(device)
    model = ExactGPModel(X_train, y_train, likelihood, kernel).to(device)
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    n_iterations = n_iter
    # Train
    for i in range(1, n_iterations+1):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward(); optimizer.step()
        if i % 100 == 0:
            print(f"[{name}] Iter {i}/{n_iterations} - Loss {loss.item():.4f}")
    
    # Eval
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_log = likelihood(model(X_test)).mean.cpu().numpy()
    print(pred_log)
    if mode == 'log':
        y_pred = np.exp(pred_log) - 1e-8
        y_true = np.exp(y_test.cpu().numpy()) - 1e-8
    elif mode == 'no_log':
        y_pred = pred_log
        y_true = y_test.cpu().numpy()

    # Compute RMSE and identify worst errors
    errors = np.abs(y_pred - y_true)
    rmse_initial = np.sqrt(np.mean(errors**2))
    
    errors_not_abs = y_pred - y_true
    abs_error_percent = errors/y_true * 100
    error_percent = (y_pred - y_true)/y_true * 100
    
    print(f"{name} RMSE: {rmse_initial:.4f}, {np.mean(abs_error_percent)} %. Max error: {errors_not_abs[np.argmax(abs_error_percent)]}, in {error_percent[np.argmax(abs_error_percent)]} %")

    # Save model
    torch.save(model, os.path.join(output_dir, f'model.pth'))
    print(f"Saved final model for {name}")

    np.save(os.path.join(data_dir, f'X_train.npy'), X_train_np)
    np.save(os.path.join(data_dir, f'y_train.npy'), y_train_np)
    np.save(os.path.join(data_dir, f'X_test.npy'),  X_test_np)
    np.save(os.path.join(data_dir, f'y_test.npy'),  y_test_np)
    np.save(os.path.join(data_dir, f'y_pred.npy'),  y_pred)

   

def main():
    main_dir = './'
    data_path = os.path.join(main_dir, 'dataset.csv')
    gases = ['H2','N2']
    for gas in gases:
        print(gas)
        X_np, y_np, train_idx, test_idx = load_data(
                csv_path= data_path,
                gas= 'N2',
                feature_cols = None,
                target_col= 'tot_rate',
                n_train= 1500,
                mode = mode
        )


        kernels = make_kernels(X_np.shape[1])
        for name, kernel in kernels.items():
            data_dir = os.path.join(main_dir, f'dataset/{mode}/{gas}')
            os.makedirs(data_dir, exist_ok=True)
            output_dir = os.path.join(main_dir, f'results/{mode}/{gas}/{name}')
            os.makedirs(output_dir, exist_ok=True)
 
            train_evaluate_augment(name, kernel,
                                X_np, y_np,
                                train_idx.copy(), test_idx.copy(),
                                output_dir, data_dir, n_iter = 10000)
    print("All experiments done.")

if __name__ == '__main__':
    main()
