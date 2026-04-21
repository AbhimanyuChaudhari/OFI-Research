# Order Flow Imbalance (OFI) Research

A research-grade implementation of the OFI framework from **Cont, Kukanov & Stoikov (2014)** — applied to real LOBSTER-format limit order book data for AAPL.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Data](https://img.shields.io/badge/Data-LOBSTER%20LOB-orange?style=flat-square)
![Research](https://img.shields.io/badge/Research-Microstructure-green?style=flat-square)

---

## Research Question

*Does Order Flow Imbalance in the limit order book have statistically significant predictive power for short-term price movements, and do deeper LOB levels add incremental information beyond the best bid/ask?*

---

## What This Implements

### Three OFI Variants (Cont et al. 2014)

**Best-Level OFI** — the change in net order pressure at the best bid/ask:
```
OFI_t = ΔBid_t - ΔAsk_t
```
Where ΔBid_t = +q_b if price improved, q_b - q_b_prev if unchanged, -q_b if deteriorated.

**Multi-Level OFI** — extends to all 10 LOB levels with depth normalization:
```
OFI_t^m = (ΔBid_t^m - ΔAsk_t^m) / avg_depth^m
```
Normalization makes levels comparable across liquidity regimes.

**Integrated OFI** — PCA compression of the multi-level vector:
```
OFI_integrated = w^T × [OFI^1, OFI^2, ..., OFI^M]
```
The first principal component captures the dominant direction of order flow across all levels.

### Price Impact Regression

Core regression with **Newey-West HAC standard errors** (critical for tick data):
```
Δp_{t+1} = α + β × OFI_t + ε_t
```
Naive OLS on tick data produces inflated t-statistics due to autocorrelation. Newey-West correction is the standard in microstructure research.

### Analysis Pipeline
- Multi-horizon impact (how many periods does OFI predict?)
- R² decomposition by LOB level (which levels matter?)
- Rolling regression (is impact time-varying?)
- Permanent vs temporary price impact decomposition
- LOB depth heatmap visualization

---

## Improvements Over Original Notebook

| Original | This Version |
|---|---|
| Python `for` loop over rows | Vectorized NumPy — 100x faster |
| No price impact regression | OLS + Newey-West t-stats |
| No statistical significance | p-values, R², confidence intervals |
| Tick-level only | Multi-frequency: 1s / 5s / 30s |
| No visualization | LOB heatmap, decay charts, rolling β |
| No decay analysis | Multi-horizon + permanent/temporary |

---

## Project Structure

```
OFI-Research/
├── src/
│   ├── ofi.py            # OFI computation (vectorized), LOB utils
│   ├── price_impact.py   # Regressions, Newey-West, rolling, decomposition
│   └── visualization.py  # All Plotly charts
├── notebooks/
│   ├── 01_microstructure.ipynb  # LOB analysis, spread, depth heatmap
│   ├── 02_ofi_computation.ipynb # All three OFI variants + aggregation
│   ├── 03_price_impact.ipynb    # Regressions + decay analysis
│   └── 04_research.ipynb        # Summary findings + conclusions
├── data/
│   └── first_25000_rows.csv     # AAPL LOB data (LOBSTER format)
├── requirements.txt
└── README.md
```

---

## Key Findings

*(Run notebooks to generate results)*

The pipeline produces:
- Whether OFI has statistically significant price impact (p-value from Newey-West regression)
- Whether integrated OFI outperforms best-level OFI
- Which LOB depth levels carry the most price-relevant information
- The decay structure of OFI price impact across horizons
- The split between permanent (information) and temporary (noise) impact

---

## Reference

Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*. Journal of Financial Econometrics, 12(1), 47–88.

---

## Data Format

LOBSTER-format LOB data with 10 bid/ask levels:
- `bid_px_00` ... `bid_px_09`: bid prices at each level
- `bid_sz_00` ... `bid_sz_09`: bid sizes at each level
- `ask_px_00` ... `ask_px_09`: ask prices at each level
- `ask_sz_00` ... `ask_sz_09`: ask sizes at each level
- `action`: A (add), C (cancel), T (trade)
- `side`: B (bid), A (ask)

---

## Getting Started

```bash
pip install -r requirements.txt
jupyter lab
```

Run notebooks in order: **01 → 02 → 03 → 04**. Notebook 02 saves `data/ofi_data.pkl` which notebooks 03 and 04 load.

---

## Contact

**Abhimanyu Chaudhari** — MS Financial Technologies, NJIT
[LinkedIn](http://www.linkedin.com/in/abhimanyu-chaudhari16) · [GitHub](https://github.com/AbhimanyuChaudhari) · abhimanyuchaudhari16@gmail.com
