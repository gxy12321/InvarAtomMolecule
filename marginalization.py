import os
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch import settings

Re_mp = {
    'H2': {0: 6.392774468860418, 2: 5.517380194079086},
    'N2': {0: 6.319050033666612, 2: 5.8449690023854055},
}
De_mp = {
    'H2':{0:7.27039879296524,  2:3.505674060033286 },
    'N2':{0:19.97132478219435, 2:11.933923088199135},
}

# --- GP utilities ---
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x)
        )

def gp_predict_torch(model, pts: torch.Tensor):
    """
    pts: (batch_size, 6) float Tensor on the correct device.
    Returns a tuple of Tensors: (mean, variance), each of shape (batch_size,).
    """
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), settings.fast_pred_var():
        out = model.likelihood(model(pts))
    return out.mean, out.variance  # Both Tensors on same device

# --- Configuration ---
main_dir    = './'
csv_path    = os.path.join(main_dir, 'dataset.csv')
gases       = ['H2', 'N2']
M           = 10000  # Number of Monte Carlo samples
n_pts       = 500   # Grid points per dimension
batch_size  = 1000  # Number of grid points to process per batch
feature_cols= ['de0', 're0', 'de2', 're2', 'c62', 'b']
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def tensorify(x): return torch.from_numpy(x).to(device)
# Set GPyTorch settings to mitigate previous CG warnings


# Load full dataset once
df_all = pd.read_csv(csv_path)

name = 'Matern+RBF+RQ+Linear'
for gas in gases:
    torch.cuda.empty_cache()
    # 1) Load model onto GPU
    output_dir = f'results/log/{gas}/{name}'
    os.makedirs(output_dir,exist_ok=True)
    model_path = os.path.join(output_dir, f'model.pth')
    x_path = os.path.join(output_dir, f'X_train.npy')
    y_path = os.path.join(output_dir, f'y_train.npy')
    X_train_np = np.load(x_path)
    y_train_np = np.load(y_path)

    try:
        model = torch.load(model_path, map_location=device, weights_only=False).to(device)
        model.set_train_data(
            inputs=tensorify(X_train_np),
            targets=tensorify(y_train_np),
            strict=False
        )
        print(model)
    except RuntimeError as e:
        print(f"Error loading model for {gas}: {e}")
        continue

    # 2) Subset & scale data to find feature ranges
    df = (df_all[df_all['gas'] == gas]
          .sort_values('tot_rate')
          .reset_index(drop=True))
    
    df = df[df['de0'] <= 110]
    df = df[df['de2'] <= 110]
    df = df[df['re2'] <= 7]

    if gas == 'H2':
        df = df[df['c62'] <= 156]
    elif gas == 'N2':
        df = df[df['c62'] <= 350.04]
    X = df[feature_cols].values.astype(np.float32)
    mins, maxs = X.min(axis=0), X.max(axis=0)

    # 3) Loop over dimensions to grid (dim_x = 0, dim_y = 2)
    
    # Define fixed dimensions
    fixed_dims = [1]  # define the dimension you want to fix 
    fixed_values = [Re_mp[gas][0]] # give the value for fixed dimensions

    
    dim_to_grid = 0 # dim_to_grid is the dim that you want to marginalize over
  

    # Define background dimensions to sample once (consistent across grid points)
    bg_dims = [ i  for i in range(0, X.shape[1]+1) if i not in [dim_to_grid] + fixed_dims]

    lower_bg = torch.tensor(mins[bg_dims], device=device)
    upper_bg = torch.tensor(maxs[bg_dims], device=device)
    bg = torch.rand(M, len(bg_dims), device=device) * (upper_bg - lower_bg) + lower_bg

    # Create 1D grid over the current dimension
    grid = torch.linspace(mins[dim_to_grid], maxs[dim_to_grid], n_pts, device=device)

    # Initialize output arrays
    num_grid = n_pts
    preds_np = np.empty((num_grid, M), dtype=np.float32)
    vars_np = np.empty((num_grid, M), dtype=np.float32)

    # Process grid points in batches
    for start_idx in range(0, num_grid, batch_size):
        end_idx = min(start_idx + batch_size, num_grid)
        batch_grid = grid[start_idx:end_idx]  # (batch_size,)

        batch_preds = torch.empty(end_idx - start_idx, M, device=device)
        batch_vars = torch.empty(end_idx - start_idx, M, device=device)

        for i, v in enumerate(batch_grid):
            
            # Build input tensor (M, 6)
            pts = torch.zeros(M, 6, device=device, dtype=torch.float32)
            pts[:, dim_to_grid] = v
            pts[:, fixed_dims[0]] = fixed_values[0]
            pts[:, bg_dims] = bg  # Broadcast background dims
            try:
                mean, variance = gp_predict_torch(model, pts)
                batch_preds[i] = mean
                batch_vars[i] = variance
            except RuntimeError as e:
                print(f"CUDA error at grid point {start_idx + i}/{num_grid} for dim {dim_to_grid}: {e}")
                batch_preds[i] = torch.full((M,), float('nan'), device=device)
                batch_vars[i] = torch.full((M,), float('nan'), device=device)
            print(f'[{gas}] Dim {dim_to_grid}, Grid point {start_idx + i}/{num_grid}', end='\r')

        # Store batch results
        preds_np[start_idx:end_idx] = batch_preds.cpu().numpy()
        vars_np[start_idx:end_idx] = batch_vars.cpu().numpy()
        torch.cuda.empty_cache()  # Free GPU memory after each batch

    # Save results with dimension-specific filenames
    suffix = f'dim{dim_to_grid}'
    for i,d in enumerate(fixed_dims):
        if d == 1:
            continue # since we always fix Re,0
        suffix += '_' + feature_cols[d]+ f'{fixed_values[i]}'
    preds_file = os.path.join(output_dir, f'MC_preds_{name}_{suffix}.npy')
    bg_file = os.path.join(output_dir, f'MC_bgs_{name}_{suffix}.npy')
    vars_file = os.path.join(output_dir, f'MC_vars_{name}_{suffix}.npy')
    grid_file = os.path.join(output_dir, f'MC_grid_{name}_{suffix}.npy')
    np.save(preds_file, preds_np)
    np.save(vars_file, vars_np)
    np.save(grid_file, grid.cpu().numpy())
    np.save(bg_file, bg.cpu().numpy())
    print(f"[{gas}] Saved predictions, variances, and grid for dim {dim_to_grid} to {output_dir}")