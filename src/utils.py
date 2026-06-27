"""
Utility functions for the VAE Portfolio project
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


def calculate_log_returns(prices):
    """
    Calculate log returns from price series
    
    Args:
        prices: numpy array or pandas Series of prices
        
    Returns:
        log returns
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    return np.diff(np.log(prices))


def normalize_data(data, scaler=None, fit=False):
    """
    Normalize data using StandardScaler
    
    Args:
        data: numpy array (n_samples, n_features)
        scaler: StandardScaler object or None
        fit: if True, fit the scaler on the data
        
    Returns:
        normalized data, scaler object
    """
    if scaler is None:
        scaler = StandardScaler()
        if fit:
            data_normalized = scaler.fit_transform(data)
        else:
            data_normalized = scaler.transform(data)
    else:
        data_normalized = scaler.transform(data)
    
    return data_normalized, scaler


def denormalize_data(data_normalized, scaler):
    """
    Denormalize data back to original scale
    
    Args:
        data_normalized: normalized data
        scaler: StandardScaler object
        
    Returns:
        denormalized data
    """
    return scaler.inverse_transform(data_normalized)


def numpy_to_torch(data, device='cpu'):
    """
    Convert numpy array to torch tensor
    
    Args:
        data: numpy array
        device: 'cpu' or 'cuda'
        
    Returns:
        torch tensor
    """
    return torch.FloatTensor(data).to(device)


def torch_to_numpy(tensor):
    """
    Convert torch tensor to numpy array
    
    Args:
        tensor: torch tensor
        
    Returns:
        numpy array
    """
    return tensor.cpu().detach().numpy()


def sample_stressed_latent(n_samples, latent_dim, mean_shift=None, std_scale=1.0, device='cpu'):
    """
    Sample latent vectors from a stressed Gaussian distribution.

    Args:
        n_samples: number of latent samples
        latent_dim: latent space dimension
        mean_shift: optional array-like shift applied to the latent mean
        std_scale: multiplier applied to the latent standard deviation
        device: torch device

    Returns:
        torch tensor of shape (n_samples, latent_dim)
    """
    z = torch.randn(n_samples, latent_dim, device=device) * std_scale

    if mean_shift is not None:
        shift = torch.as_tensor(mean_shift, dtype=z.dtype, device=device)
        if shift.ndim == 0:
            shift = shift.repeat(latent_dim)
        z = z + shift

    return z


def select_top_stressed_scenarios(scenarios, weights, top_k=None, tail_fraction=None):
    """
    Rank scenarios by portfolio loss and keep the most stressed ones.

    Args:
        scenarios: numpy array of shape (n_scenarios, n_assets)
        weights: portfolio weights
        top_k: number of scenarios to keep
        tail_fraction: fraction of worst scenarios to keep if top_k is None

    Returns:
        selected_scenarios, portfolio_returns, stress_scores
    """
    weights = np.asarray(weights)
    portfolio_returns = np.dot(scenarios, weights)
    stress_scores = -portfolio_returns

    if top_k is None:
        if tail_fraction is None:
            tail_fraction = 0.1
        top_k = max(1, int(len(stress_scores) * tail_fraction))

    top_indices = np.argsort(stress_scores)[-top_k:]
    top_indices = top_indices[np.argsort(stress_scores[top_indices])[::-1]]

    return scenarios[top_indices], portfolio_returns[top_indices], stress_scores[top_indices]


class PortfolioConfig:
    """Configuration class for portfolio parameters"""
    
    def __init__(self):
        self.indices = ['^GSPC', '^STOXX50E', '^N225', '^HSI']  # S&P500, STOXX50, Nikkei, Hang Seng
        self.start_date = '2015-01-01'
        self.end_date = '2023-12-31'
        self.weights = np.array([0.4, 0.3, 0.2, 0.1])  # Portfolio weights
        
        # VAE parameters
        self.latent_dim = 10
        self.hidden_dim = 64
        self.n_epochs = 100
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # VaR parameters
        self.var_levels = [0.95, 0.99]
        self.n_scenarios = 10000
        self.stress_tail_fraction = 0.1
