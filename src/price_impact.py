"""
price_impact.py
---------------
OFI price impact models — does OFI predict future price changes?

Models:
  1. Linear regression (OLS) with Newey-West standard errors
  2. Multi-horizon regression (how many periods forward does OFI predict?)
  3. Impulse response function (OFI decay analysis)
  4. R² decomposition by LOB level
  5. Rolling regression (time-varying impact)

These are the core research questions in market microstructure.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
import warnings
warnings.filterwarnings('ignore')


# ── OLS with Newey-West standard errors ──────────────────────────────────────

def ofi_price_impact(df: pd.DataFrame,
                      ofi_col: str = 'ofi_best',
                      return_col: str = 'fwd_return',
                      n_lags: int = 5) -> dict:
    """
    Regress forward mid-price returns on OFI.

    Model: Δp_{t+1} = α + β × OFI_t + ε_t

    Uses Newey-West (HAC) standard errors to account for
    autocorrelation and heteroscedasticity in tick data.
    This is the CORRECT approach for time-series microstructure regressions.

    Parameters
    ----------
    df         : aggregated LOB DataFrame with OFI and forward returns
    ofi_col    : column name for OFI
    return_col : column name for forward return
    n_lags     : Newey-West lag truncation

    Returns dict with coefficient, t-stat, p-value, R²
    """
    sub = df[[ofi_col, return_col]].dropna()
    if len(sub) < 30:
        return {'error': 'Not enough observations'}

    X = sm.add_constant(sub[ofi_col].values)
    y = sub[return_col].values

    model = sm.OLS(y, X)
    res   = model.fit()

    # Newey-West HAC covariance (correct for time-series)
    nw_cov = cov_hac(res, nlags=n_lags)
    nw_se  = np.sqrt(np.diag(nw_cov))
    nw_t   = res.params / nw_se
    nw_p   = 2 * (1 - sm.stats.stattools.durbin_watson(res.resid))  # approximate

    # Proper p-values from t-distribution
    from scipy import stats
    nw_p_vals = 2 * stats.t.sf(np.abs(nw_t), df=len(y) - 2)

    return {
        'alpha':        round(res.params[0], 8),
        'beta':         round(res.params[1], 8),
        'beta_se_nw':   round(nw_se[1], 8),
        't_stat_nw':    round(nw_t[1], 4),
        'p_value_nw':   round(nw_p_vals[1], 4),
        'r_squared':    round(res.rsquared, 4),
        'r_squared_adj':round(res.rsquared_adj, 4),
        'n_obs':        len(y),
        'significant':  nw_p_vals[1] < 0.05,
    }


# ── Multi-horizon regression ──────────────────────────────────────────────────

def multi_horizon_impact(df: pd.DataFrame,
                          ofi_col: str,
                          horizons: list = None) -> pd.DataFrame:
    """
    Test OFI predictability at multiple forward horizons.

    Runs: Δp_{t+h} = α + β_h × OFI_t for each h in horizons.

    This reveals the decay structure of OFI's price impact —
    how many periods forward does the signal remain predictive?
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 20, 30]

    results = []
    log_mid = df['log_mid'] if 'log_mid' in df.columns else np.log(df['mid_price'])

    for h in horizons:
        fwd_ret = log_mid.diff(h).shift(-h)
        sub = pd.DataFrame({'ofi': df[ofi_col], 'ret': fwd_ret}).dropna()

        if len(sub) < 30:
            continue

        X = sm.add_constant(sub['ofi'].values)
        y = sub['ret'].values
        res = sm.OLS(y, X).fit()

        try:
            nw_cov = cov_hac(res, nlags=min(h + 2, len(y) // 4))
            nw_se  = np.sqrt(np.diag(nw_cov))
            from scipy import stats
            t_stat = res.params[1] / nw_se[1]
            p_val  = 2 * stats.t.sf(abs(t_stat), df=len(y)-2)
        except Exception:
            t_stat = res.tvalues[1]
            p_val  = res.pvalues[1]

        results.append({
            'horizon':    h,
            'beta':       round(res.params[1], 8),
            't_stat':     round(t_stat, 3),
            'p_value':    round(p_val, 4),
            'r_squared':  round(res.rsquared, 4),
            'significant':p_val < 0.05,
        })

    return pd.DataFrame(results)


# ── Level-by-level R² decomposition ──────────────────────────────────────────

def level_r2_decomposition(df: pd.DataFrame,
                             level_cols: list,
                             return_col: str = 'fwd_return') -> pd.DataFrame:
    """
    Test each LOB level's OFI independently against forward returns.

    Answers: which depth levels carry the most price-relevant information?
    Best level only? Or do deeper levels add signal?
    """
    results = []
    sub_ret = df[return_col].dropna()

    for col in level_cols:
        sub = pd.DataFrame({'ofi': df[col], 'ret': df[return_col]}).dropna()
        if len(sub) < 30:
            continue

        X = sm.add_constant(sub['ofi'].values)
        res = sm.OLS(sub['ret'].values, X).fit()

        results.append({
            'level':      col,
            'beta':       round(res.params[1], 8),
            'r_squared':  round(res.rsquared, 4),
            't_stat':     round(res.tvalues[1], 3),
            'p_value':    round(res.pvalues[1], 4),
        })

    return pd.DataFrame(results).sort_values('r_squared', ascending=False)


# ── Rolling regression ─────────────────────────────────────────────────────────

def rolling_price_impact(df: pd.DataFrame,
                          ofi_col: str,
                          return_col: str = 'fwd_return',
                          window: int = 100) -> pd.DataFrame:
    """
    Rolling OLS: how does the OFI→price impact coefficient change over time?

    A time-varying beta suggests the market's response to order flow
    changes throughout the trading day (common pattern: stronger impact
    near open and close).
    """
    sub = df[[ofi_col, return_col, 'ts_event']].dropna().reset_index(drop=True)
    betas = []
    r2s   = []
    dates = []

    for i in range(window, len(sub)):
        win = sub.iloc[i - window: i]
        X   = sm.add_constant(win[ofi_col].values)
        res = sm.OLS(win[return_col].values, X).fit()
        betas.append(res.params[1])
        r2s.append(res.rsquared)
        dates.append(sub['ts_event'].iloc[i])

    return pd.DataFrame({
        'ts_event': dates,
        'beta':     betas,
        'r_squared':r2s,
    })


# ── Permanent vs temporary impact ─────────────────────────────────────────────

def permanent_temporary_impact(df: pd.DataFrame,
                                 ofi_col: str,
                                 horizons: list = None) -> dict:
    """
    Decompose price impact into permanent and temporary components.

    Permanent impact: long-run price change per unit OFI
    Temporary impact: short-run price change that subsequently reverses

    A high temporary/permanent ratio suggests more noise trading.
    """
    if horizons is None:
        horizons = [1, 5, 10, 30, 60]

    log_mid = np.log(df['mid_price'])
    results = {}

    for h in horizons:
        fwd_ret = log_mid.diff(h).shift(-h)
        sub = pd.DataFrame({'ofi': df[ofi_col], 'ret': fwd_ret}).dropna()
        if len(sub) < 30:
            continue
        X   = sm.add_constant(sub['ofi'].values)
        res = sm.OLS(sub['ret'].values, X).fit()
        results[h] = round(res.params[1], 8)

    if not results:
        return {}

    betas  = list(results.values())
    max_b  = max(betas)
    final_b= betas[-1]

    return {
        'betas_by_horizon':  results,
        'peak_impact':       round(max_b, 8),
        'permanent_impact':  round(final_b, 8),
        'temporary_impact':  round(max_b - final_b, 8),
        'reversion_ratio':   round((max_b - final_b) / (abs(max_b) + 1e-10), 4),
    }
