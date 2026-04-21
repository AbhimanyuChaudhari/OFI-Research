"""
ofi.py
------
Order Flow Imbalance (OFI) computations for limit order book data.

Implements the framework from:
  Cont, Kukanov & Stoikov (2014) — "The Price Impact of Order Book Events"
  Journal of Financial Econometrics

Three OFI variants:
  1. Best-level OFI  — uses only the best bid/ask
  2. Multi-level OFI — uses top N levels, depth-normalized
  3. Integrated OFI  — PCA compression of multi-level vector

All functions are vectorized using NumPy — no Python loops.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ── Data loading & preprocessing ─────────────────────────────────────────────

def load_lob_data(path: str) -> pd.DataFrame:
    """
    Load LOBSTER-format LOB data.
    Computes mid-price, spread, and standardizes timestamps.
    """
    df = pd.read_csv(path)
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
    df = df.sort_values('ts_event').reset_index(drop=True)

    # Mid-price and spread
    df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
    df['spread']    = df['ask_px_00'] - df['bid_px_00']
    df['spread_bps'] = df['spread'] / df['mid_price'] * 10_000

    # Log mid-price for return computation
    df['log_mid'] = np.log(df['mid_price'])

    return df


def get_lob_levels(df: pd.DataFrame) -> int:
    """Detect how many LOB levels are available in the data."""
    levels = 0
    for i in range(20):
        if f'bid_px_{i:02d}' in df.columns:
            levels = i + 1
        else:
            break
    return levels


# ── Best-level OFI (vectorized) ───────────────────────────────────────────────

def compute_best_ofi(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized best-level OFI computation.

    OFI_t = ΔBid_t - ΔAsk_t

    Where:
        ΔBid_t = +q_b if P_b increases
                  q_b - q_b_prev if P_b unchanged
                 -q_b if P_b decreases

    Reference: Cont et al. (2014), Equation (1)
    """
    bp = df['bid_px_00'].values
    bs = df['bid_sz_00'].values
    ap = df['ask_px_00'].values
    as_ = df['ask_sz_00'].values

    # Bid contribution
    delta_bid = np.where(
        bp[1:] > bp[:-1],  bs[1:],
        np.where(bp[1:] == bp[:-1], bs[1:] - bs[:-1], -bs[1:])
    )

    # Ask contribution (note: ask increases = negative OFI)
    delta_ask = np.where(
        ap[1:] > ap[:-1], -as_[1:],
        np.where(ap[1:] == ap[:-1], as_[1:] - as_[:-1], as_[1:])
    )

    ofi = delta_bid - delta_ask
    return pd.Series(np.concatenate([[0], ofi]), index=df.index, name='ofi_best')


# ── Multi-level OFI (vectorized) ──────────────────────────────────────────────

def compute_multilevel_ofi(df: pd.DataFrame,
                             max_levels: int = 10,
                             normalize: bool = True) -> pd.DataFrame:
    """
    Compute depth-normalized OFI for each LOB level.

    Normalization divides each level's OFI by the average depth
    at that level, making levels comparable across different liquidity regimes.

    Returns DataFrame with columns ofi_level_00 ... ofi_level_{N-1}
    """
    available = get_lob_levels(df)
    n_levels  = min(max_levels, available)
    ofi_levels = {}

    for m in range(n_levels):
        tag = f'{m:02d}'
        bp  = df[f'bid_px_{tag}'].values
        bs  = df[f'bid_sz_{tag}'].values
        ap  = df[f'ask_px_{tag}'].values
        as_ = df[f'ask_sz_{tag}'].values

        delta_bid = np.where(
            bp[1:] > bp[:-1],  bs[1:],
            np.where(bp[1:] == bp[:-1], bs[1:] - bs[:-1], -bs[1:])
        )
        delta_ask = np.where(
            ap[1:] > ap[:-1], -as_[1:],
            np.where(ap[1:] == ap[:-1], as_[1:] - as_[:-1], as_[1:])
        )

        ofi_raw = delta_bid - delta_ask
        ofi_raw = np.concatenate([[0], ofi_raw])

        if normalize:
            avg_depth = (bs + as_).mean() / 2
            ofi_raw   = ofi_raw / (avg_depth + 1e-10)

        ofi_levels[f'ofi_level_{tag}'] = ofi_raw

    return pd.DataFrame(ofi_levels, index=df.index)


# ── Integrated OFI via PCA ────────────────────────────────────────────────────

def compute_integrated_ofi(ofi_df: pd.DataFrame,
                             level_cols: list = None) -> tuple:
    """
    Compress multi-level OFI into a single integrated signal via PCA.

    The first principal component captures the dominant direction of
    order flow across all levels.

    Returns (integrated_ofi Series, weights array, explained_variance_ratio)

    Reference: Cont et al. (2014), Section 3.3
    """
    if level_cols is None:
        level_cols = [c for c in ofi_df.columns if c.startswith('ofi_level_')]

    X = ofi_df[level_cols].fillna(0).values

    pca = PCA(n_components=1)
    pca.fit(X)

    w = pca.components_[0]
    w_norm = w / (np.sum(np.abs(w)) + 1e-10)  # L1 normalize

    integrated = X @ w_norm
    evr = pca.explained_variance_ratio_[0]

    return (
        pd.Series(integrated, index=ofi_df.index, name='ofi_integrated'),
        w_norm,
        evr,
    )


# ── Temporal aggregation ──────────────────────────────────────────────────────

def aggregate_ofi(df: pd.DataFrame,
                   ofi_cols: list,
                   freq: str = '1s') -> pd.DataFrame:
    """
    Aggregate tick-level OFI to fixed time intervals.

    Aggregation: sum of OFI within each interval.
    Also computes forward mid-price return for each interval.

    freq: pandas offset string — '1s', '5s', '30s', '1min'
    """
    df = df.copy()
    df = df.set_index('ts_event')

    agg_dict = {col: 'sum' for col in ofi_cols}
    agg_dict['mid_price'] = 'last'
    agg_dict['spread']    = 'mean'
    agg_dict['log_mid']   = 'last'

    agg = df[list(agg_dict.keys())].resample(freq).agg(agg_dict)
    agg = agg.dropna()

    # Forward log return
    agg['fwd_return'] = agg['log_mid'].diff(1).shift(-1)

    return agg.reset_index()


# ── LOB snapshot ──────────────────────────────────────────────────────────────

def lob_snapshot(df: pd.DataFrame, idx: int, n_levels: int = 10) -> dict:
    """Extract LOB snapshot at a given row index."""
    row = df.iloc[idx]
    bids = []
    asks = []
    for m in range(n_levels):
        tag = f'{m:02d}'
        bids.append({'price': row[f'bid_px_{tag}'], 'size': row[f'bid_sz_{tag}']})
        asks.append({'price': row[f'ask_px_{tag}'], 'size': row[f'ask_sz_{tag}']})
    return {
        'timestamp': row['ts_event'],
        'mid_price': row['mid_price'],
        'spread':    row['spread'],
        'bids':      bids,
        'asks':      asks,
    }


# ── Microstructure metrics ────────────────────────────────────────────────────

def microstructure_metrics(df: pd.DataFrame) -> dict:
    """
    Compute key microstructure metrics from LOB data.

    Includes: spread statistics, depth, order arrival rates,
    queue imbalance, and price volatility.
    """
    n = len(df)

    # Time elapsed
    t_start = df['ts_event'].iloc[0]
    t_end   = df['ts_event'].iloc[-1]
    duration_s = (t_end - t_start).total_seconds()

    # Spread metrics
    spread = df['spread']

    # Depth metrics (best level)
    total_depth = df['bid_sz_00'] + df['ask_sz_00']
    queue_imb   = (df['bid_sz_00'] - df['ask_sz_00']) / (df['bid_sz_00'] + df['ask_sz_00'] + 1e-10)

    # Order arrival rate
    if 'action' in df.columns:
        n_adds    = (df['action'] == 'A').sum()
        n_cancels = (df['action'] == 'C').sum()
        n_trades  = (df['action'] == 'T').sum()
        arrival_rate = n_adds / duration_s
    else:
        n_adds = n_cancels = n_trades = None
        arrival_rate = None

    # Price volatility (mid-price)
    mid_returns = df['mid_price'].pct_change().dropna()
    vol_ann     = mid_returns.std() * np.sqrt(252 * 6.5 * 3600)  # annualized

    return {
        'n_events':          n,
        'duration_s':        round(duration_s, 1),
        'arrival_rate_hz':   round(arrival_rate, 2) if arrival_rate else None,
        'n_adds':            int(n_adds) if n_adds is not None else None,
        'n_cancels':         int(n_cancels) if n_cancels is not None else None,
        'n_trades':          int(n_trades) if n_trades is not None else None,
        'mean_spread':       round(spread.mean(), 4),
        'median_spread':     round(spread.median(), 4),
        'mean_spread_bps':   round(df['spread_bps'].mean(), 2),
        'mean_depth':        round(total_depth.mean(), 1),
        'mean_queue_imb':    round(queue_imb.mean(), 4),
        'mid_price_vol_ann': round(vol_ann * 100, 2),
        'price_range':       round(df['mid_price'].max() - df['mid_price'].min(), 4),
    }
