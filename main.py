import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import time
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Choose device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic data
def generate_data(n_samples, noise_std=0.3, quality_factor=1.0):
    X = np.random.uniform(-2, 2, size=(n_samples, 1))
    y = np.sin(2 * np.pi * X) + (noise_std / quality_factor) * np.random.randn(n_samples, 1)
    return X, y

# Define a simple feed-forward network with two hidden layers
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# Function to train and evaluate the model for a given dataset size and quality level
def train_and_evaluate(n, quality):
    print(f"🚀 Training with {n} samples, Quality {quality}...")
    start_time = time.time()

    # Generate training and testing data
    X_train, y_train = generate_data(n, noise_std=0.1, quality_factor=quality)
    X_test, y_test = generate_data(500, noise_std=0.1, quality_factor=quality)

    # Convert data to torch tensors and move to the selected device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create model, loss function, and optimizer
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    batch_size = 32
    epochs = 50
    n_train = X_train.shape[0]

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mae = torch.mean(torch.abs(predictions - y_test)).item()

    elapsed_time = round(time.time() - start_time, 2)
    print(f"✅ Finished: MAE: {mae:.4f} | Time: {elapsed_time}s\n")
    return mae

import os

# Define parameter bounds for curve fitting
param_bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, 6.01])  # Ensures reasonable search space

# Experiment parameters
sample_sizes = np.array([50, 100, 200, 400, 800, 1600, 3200, 6400, 6400 * 2, 6400 * 4, 6400 * 8, 6400 * 16, 6400 * 32])
quality_levels = [1.0, 1.5, 2.0]
csv_filename = "generalization_error_results.csv"
# Small epsilon to prevent division by zero issues
epsilon = 1e-8
# import matplotlib
# matplotlib.use('Agg')

# Function to check for invalid values in log
def safe_log(y_values):
    invalid_mask = y_values <= 0
    if np.any(invalid_mask):
        print("Warning: Found invalid values in log computation:", y_values[invalid_mask])
        y_values[invalid_mask] = epsilon  # Replace invalid values with small positive number
    return np.log(y_values)

# Check if results CSV is present, if so load it; otherwise, run experiments
if os.path.exists(csv_filename):
    print(f"📂 Found existing results CSV: {csv_filename}. Loading results...")
    df_results = pd.read_csv(csv_filename)
else:
    print("📂 Results CSV not found. Running experiments...")
    data_records = []
    for quality in quality_levels:
        for n in sample_sizes:
            mae = train_and_evaluate(n, quality)
            data_records.append({"Samples": n, "Quality": quality, "MAE": mae})
    df_results = pd.DataFrame(data_records)
    df_results.to_csv(csv_filename, index=False)
    print(f"📂 Results saved to {csv_filename}")

# --- Prepare improved plotting with linear X and log Y ---
# --- Prepare plot with linear X and log Y ---
plt.figure(figsize=(10, 6))

for q in quality_levels:
    # Select and sort data for this quality level
    df_q = df_results[df_results['Quality'] == q].sort_values(by='Samples')
    x = df_q['Samples'].values
    y = df_q['MAE'].values

    # Plot experimental data as scatter points
    plt.scatter(x, y, label=f'Quality {q} data', alpha=0.7)

    # Fit power-law scaling: C + A / n^alpha
    def power_fit(n, C, A, alpha):
        return np.log(C + A / ((n + epsilon) ** (1/alpha+epsilon)))  # Avoid divide by zero

    # Check for invalid values before fitting
    safe_y = safe_log(y)

    try:
        popt, _ = curve_fit(power_fit, x, safe_y, bounds=param_bounds)  # Apply bounds
        C_fit = np.exp(popt[0])  # Convert back from log-space
        A_fit = popt[1]
        alpha_fit = popt[2]
    except Exception as e:
        print(f"Fit failed for quality {q}: {e}")
        continue

    print(f"Quality {q}: Fit -> C = {C_fit:.4f}, A = {A_fit:.4f}, alpha = {alpha_fit:.4f}")

    # Generate a smooth range for fitting
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.exp(power_fit(x_fit, *popt))  # Convert back to linear scale

    # Check for invalid values in power calculation
    invalid_mask = np.isnan(y_fit) | np.isinf(y_fit)
    if np.any(invalid_mask):
        print("Warning: Found invalid values in power computation:", y_fit[invalid_mask])
        y_fit[invalid_mask] = epsilon  # Replace invalid values with small positive number

    # Plot the fitted power-law curve
    plt.plot(x_fit, y_fit, linestyle='-.', label=f'C + A/n^α fit (q={q}, α={alpha_fit:.2f})')

# Add a "fun" equation using parameters from quality 2 but setting alpha=2
df_q2 = df_results[df_results['Quality'] == 1.5].sort_values(by='Samples')
x_q2 = df_q2['Samples'].values

try:
    param_bounds_fun = ([-np.inf, -np.inf,2], [np.inf, np.inf,2.01])  # No bounds on alpha
    popt_q2, _ = curve_fit(power_fit, x_q2, safe_log(df_q2['MAE'].values), bounds=param_bounds_fun)
    C_q2 = np.exp(popt_q2[0])
    A_q2 = popt_q2[1]
    alpha_fun = popt_q2[2] # Setting alpha to 2 just for fun

    # Generate a smooth range
    x_fit_fun = np.linspace(x_q2.min(), x_q2.max(), 200)
    y_fit_fun =np.exp(power_fit(x_fit, *popt_q2))

    # Plot the fun equation
    plt.plot(x_fit_fun, y_fit_fun, linestyle='--', color='black', label=f'Fun equation (C={C_q2:.4f}, A={A_q2:.4f}, α=2.00)')
except Exception as e:
    print(f"Fun equation fit failed: {e}")

plt.xscale('linear')  # Keep x-axis linear
plt.yscale('log')     # Log scale on y-axis
plt.xlabel('Number of Training Samples (n)')
plt.ylabel('MAE (log scale)')
plt.title('Generalization Error Scaling (Linear-X, Log-Y) with Fun Equation')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.show()
