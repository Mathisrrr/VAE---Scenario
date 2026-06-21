# 🚀 VAE Portfolio Project - Quick Start Guide

## Installation Status ✅

Your project is fully set up! Here's what has been created:

### Files & Folders Created
```
VAE project/
├── src/                          # Python modules (ready to use)
│   ├── data_loader.py           # Downloads market data
│   ├── vae_model.py             # VAE architecture
│   ├── var_calculator.py        # Risk calculations
│   └── utils.py                 # Helpers & config
├── notebooks/                    # 4 Jupyter notebooks (main workflow)
├── data/                         # Stores downloaded data
├── results/                      # Outputs (models, plots, CSV)
├── environment.yml              # Conda configuration
├── requirements.txt             # Pip dependencies
└── README.md                    # Full documentation
```

## ⚡ Next Steps

### Step 1: Activate Environment
```bash
conda activate vae-portfolio
```

### Step 2: Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Start with Notebook 1
```bash
cd notebooks/
jupyter notebook 01_data_exploration.ipynb
```

## 📓 Notebook Workflow

Run notebooks in this order:

1. **01_data_exploration.ipynb** (5-10 min)
   - Downloads 4 indices: S&P500, STOXX50, Nikkei, Hang Seng
   - Calculates daily log returns
   - Visualizes correlations
   - Output: `data/returns.csv`, `data/prices.csv`

2. **02_vae_training.ipynb** (5-15 min, depending on CPU/GPU)
   - Trains VAE on returns data
   - Monitors loss convergence
   - Saves model: `results/vae_model.pth`

3. **03_scenario_generation.ipynb** (2-5 min)
   - Generates 10,000 market scenarios
   - Compares with historical data
   - Output: `results/scenarios.csv`

4. **04_var_calculation.ipynb** (1-2 min)
   - Calculates VaR at 95% & 99% confidence
   - Generates risk report with 3 methods
   - Creates 8 visualizations

## 🎯 What You'll Get

### Outputs in `results/` folder:
```
01_returns_distribution.png      # Returns histograms
02_correlation_matrix.png        # Index correlations
03_price_time_series.png        # Price evolution
04_training_history.png         # VAE loss curves
05_scenario_comparison.png       # Generated vs historical
06_portfolio_comparison.png      # Portfolio stats
07_var_methods_comparison.png   # VaR methods comparison
08_risk_profiles.png            # Q-Q plots & risk metrics

vae_model.pth                   # Trained VAE (PyTorch)
scenarios.csv                   # 10,000 generated scenarios
portfolio_returns.csv           # Portfolio P&L
var_generated_scenarios.csv     # VaR results (generated)
var_historical.csv              # VaR results (historical)
risk_metrics.csv                # Risk statistics
```

## 🔧 Customize Your Portfolio

Edit [src/utils.py](src/utils.py), class `PortfolioConfig`:

```python
# Change indices
self.indices = ['^GSPC', '^STOXX50E', '^N225', '^HSI']

# Change weights (must sum to 1)
self.weights = np.array([0.4, 0.3, 0.2, 0.1])

# Adjust VAE architecture
self.latent_dim = 10              # Latent dimension
self.hidden_dim = 64              # Hidden layer size
self.n_epochs = 100               # Training epochs
self.batch_size = 32              # Batch size

# More scenarios = more accurate but slower
self.n_scenarios = 10000          # Generate 10k scenarios
```

## 📊 Understanding the Results

### Value at Risk (VaR)
- **VaR at 95%**: Maximum expected loss in worst 5% of cases
- **VaR at 99%**: Maximum expected loss in worst 1% of cases
- **CVaR**: Average loss beyond VaR (tail risk)

### Example Output
```
Confidence Level | Historical VaR | Parametric VaR | CVaR
95%              | -2.87%         | -3.12%         | -5.12%
99%              | -5.12%         | -4.87%         | -6.87%
```
This means:
- Portfolio could lose up to 2.87% in worst 5% of scenarios (95% VaR)
- Portfolio could lose up to 5.12% in worst 1% of scenarios (99% VaR)

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Check environment is activated
conda activate vae-portfolio

# Verify imports
python -c "import torch, pandas, sklearn, yfinance"
```

### Issue: yfinance can't download data
- Check internet connection
- Verify ticker symbols (^GSPC not SPY for S&P500)
- Try downloading manually to test

### Issue: Out of memory
- Reduce `n_scenarios` in `utils.py`
- Use CPU instead of GPU (automatic by default)
- Reduce `batch_size` in training

### Issue: Slow training
- Use GPU: `self.device = 'cuda'`
- Reduce `latent_dim` from 10 to 5-8
- Try fewer epochs (50 instead of 100)

## 📚 Resources

- **VAE Theory**: Kingma & Welling (2013) - Auto-Encoding Variational Bayes
- **VaR Methodology**: JP Morgan RiskMetrics
- **PyTorch Docs**: https://pytorch.org/docs
- **Pandas Docs**: https://pandas.pydata.org

## ✨ Key Features

✅ **Automatic Data Download**: Fetches real market data from Yahoo Finance  
✅ **VAE Implementation**: PyTorch with batch normalization & proper loss  
✅ **Multiple VaR Methods**: Historical, Parametric, CVaR  
✅ **Visualizations**: 8 publication-quality plots  
✅ **Comparison Analysis**: Generated vs historical statistics  
✅ **Production Ready**: Modular, documented, extensible code  

## 🎓 Learning Outcomes

After running this project, you'll understand:
- How VAEs learn latent distributions
- How to generate realistic market scenarios
- How to calculate and interpret VaR metrics
- Risk management techniques in portfolio analysis

---

**Ready to start?** → Run `jupyter notebook` in the notebooks/ folder!

**Questions?** → Check README.md for detailed documentation

**Next steps** → Edit `utils.py` to customize your portfolio!
