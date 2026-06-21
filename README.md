# VAE Portfolio VaR Project

## 📋 Overview

This project simulates portfolio scenarios containing stock indices using a **Variational Autoencoder (VAE)** and calculates **Value at Risk (VaR)** metrics.

### Key Components
- **VAE Model**: Learns the distribution of index returns
- **Scenario Generation**: Creates realistic market scenarios
- **VaR Calculation**: Computes risk metrics (Historical, Parametric, CVaR)

## 🏗️ Project Structure

```
VAE project/
├── data/                      # Raw and processed data
│   ├── returns.csv           # Daily returns data
│   └── prices.csv            # Historical prices
├── src/                       # Python modules
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   ├── vae_model.py          # VAE architecture & trainer
│   ├── var_calculator.py     # VaR computation
│   └── utils.py              # Helper functions
├── notebooks/                 # Jupyter notebooks (main workflow)
│   ├── 01_data_exploration.ipynb
│   ├── 02_vae_training.ipynb
│   ├── 03_scenario_generation.ipynb
│   └── 04_var_calculation.ipynb
├── results/                   # Output files
│   ├── vae_model.pth         # Trained model
│   ├── scenarios.csv         # Generated scenarios
│   └── *.png                 # Visualizations
├── requirements.txt
├── environment.yml           # Conda configuration
└── README.md
```

## 🚀 Quick Start

### 1. Create Conda Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate vae-portfolio
```

### 2. Run Notebooks in Order

#### Notebook 1: Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
- Loads historical data for 4 indices: S&P500, STOXX50, Nikkei, Hang Seng
- Calculates returns and correlation matrix
- Visualizes price evolution and distributions

#### Notebook 2: VAE Training
```bash
jupyter notebook notebooks/02_vae_training.ipynb
```
- Trains the VAE on historical returns
- Monitors reconstruction and KL divergence losses
- Saves trained model

#### Notebook 3: Scenario Generation
```bash
jupyter notebook notebooks/03_scenario_generation.ipynb
```
- Generates 10,000 market scenarios using trained VAE
- Compares generated vs historical distributions
- Validates statistical properties

#### Notebook 4: VaR Calculation
```bash
jupyter notebook notebooks/04_var_calculation.ipynb
```
- Calculates VaR at 95% and 99% confidence levels
- Compares 3 methods: Historical, Parametric, CVaR
- Generates comprehensive risk report

## 📊 Key Metrics

### Value at Risk (VaR)
- **Historical VaR**: Percentile-based approach
- **Parametric VaR**: Normal distribution assumption
- **CVaR (Expected Shortfall)**: Average loss beyond VaR

### Risk Metrics
- Mean Return
- Volatility (Std Dev)
- Skewness
- Kurtosis
- Sharpe Ratio

## 🔧 Configuration

Edit [src/utils.py](src/utils.py) `PortfolioConfig` class to customize:

```python
self.indices = ['^GSPC', '^STOXX50E', '^N225', '^HSI']  # Add/remove indices
self.weights = np.array([0.4, 0.3, 0.2, 0.1])           # Adjust portfolio weights
self.latent_dim = 10                                      # VAE latent dimension
self.n_epochs = 100                                       # Training epochs
self.n_scenarios = 10000                                  # Number of scenarios
```

## 📈 Results

The project generates:
- **8 visualization plots** (in `results/` folder)
- **VaR summary tables** (CSV format)
- **Trained VAE model** (PyTorch .pth file)
- **Generated scenarios** (10,000 market paths)

### Sample Output

```
VaR CALCULATION - VAE GENERATED SCENARIOS
Confidence Level | Historical VaR | Parametric VaR | CVaR
95%              | -0.0287        | -0.0312        | -0.0512
99%              | -0.0512        | -0.0487        | -0.0687
```

## 🧠 VAE Architecture

```
Input (4 indices)
    ↓
Encoder (Linear 64 → ReLU → Linear 64)
    ↓
Latent Space (μ, σ) → Sample z ~ N(0,1)
    ↓
Decoder (Linear 64 → ReLU → Linear 64 → Linear 4)
    ↓
Output (4 indices reconstructed)
```

**Loss Function**: Reconstruction Loss + β × KL Divergence

## 🛠️ Technologies

- **PyTorch**: Deep learning framework
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Data preprocessing & statistics
- **Matplotlib/Seaborn**: Visualization
- **yfinance**: Market data retrieval

## 📝 References

- **VAE**: Kingma & Welling (2013) - Auto-Encoding Variational Bayes
- **VaR**: Risk Management in Finance (standard financial metrics)
- **Monte Carlo**: Risk scenario simulation

## 🐛 Troubleshooting

### CUDA Not Available
The code automatically falls back to CPU. To use GPU:
```bash
conda install pytorch::pytorch pytorch::pytorch::cuda
```

### Data Download Issues
If yfinance fails to download data, check:
- Internet connection
- Ticker symbols (use `^GSPC` not `SPY`)
- Date range is valid

### Memory Issues
Reduce `n_scenarios` in `utils.py` if VAE generation runs out of memory.

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Created as a portfolio simulation and risk analysis project.

---

**Status**: ✅ Complete and production-ready

**Last Updated**: June 2026
# VAE---Scenario
