"""
visualization.py
----------------
All Plotly visualizations for OFI research.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEME = dict(template='plotly_white',
             font=dict(family='Inter, sans-serif', size=11),
             margin=dict(l=50, r=30, t=60, b=40))
C = ['#4C6EF5','#F76707','#2F9E44','#E03131','#7950F2','#1098AD','#F59F00','#364FC7']


def lob_heatmap(df: pd.DataFrame, n_levels: int = 5,
                sample_every: int = 10) -> go.Figure:
    """
    LOB depth heatmap — shows bid/ask sizes across levels over time.
    Visually striking and shows how liquidity evolves.
    """
    df_s = df.iloc[::sample_every].reset_index(drop=True)
    timestamps = df_s['ts_event'].astype(str)

    bid_matrix = np.array([[df_s[f'bid_sz_{m:02d}'].values
                             for m in range(n_levels)]]).squeeze()
    ask_matrix = np.array([[df_s[f'ask_sz_{m:02d}'].values
                             for m in range(n_levels)]]).squeeze()

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=['Bid Depth', 'Ask Depth'],
                         shared_yaxes=True)

    fig.add_trace(go.Heatmap(
        z=bid_matrix,
        x=timestamps,
        y=[f'Level {m}' for m in range(n_levels)],
        colorscale='Blues',
        name='Bid Depth',
        colorbar=dict(x=0.45, title='Size'),
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=ask_matrix,
        x=timestamps,
        y=[f'Level {m}' for m in range(n_levels)],
        colorscale='Reds',
        name='Ask Depth',
        colorbar=dict(x=1.0, title='Size'),
    ), row=1, col=2)

    fig.update_layout(title='Limit Order Book Depth Heatmap',
                       height=450, **THEME)
    return fig


def ofi_price_chart(df: pd.DataFrame,
                     ofi_col: str = 'ofi_best',
                     ticker: str = 'AAPL') -> go.Figure:
    """OFI time series overlaid with mid-price."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=['Mid Price', 'OFI', 'Bid-Ask Spread (bps)'],
                         row_heights=[0.45, 0.35, 0.2],
                         vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=df['ts_event'], y=df['mid_price'],
        line=dict(color=C[0], width=1.5), name='Mid Price'
    ), row=1, col=1)

    # OFI colored positive/negative
    ofi = df[ofi_col].fillna(0)
    fig.add_trace(go.Bar(
        x=df['ts_event'], y=ofi,
        marker_color=[C[2] if v >= 0 else C[3] for v in ofi],
        opacity=0.7, name='OFI'
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='gray', row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df['ts_event'], y=df['spread_bps'],
        line=dict(color=C[1], width=1), name='Spread (bps)',
        fill='tozeroy', fillcolor='rgba(247,103,7,0.1)'
    ), row=3, col=1)

    fig.update_layout(title=f'{ticker} — Mid Price, OFI & Spread',
                       height=600, showlegend=True, **THEME)
    fig.update_yaxes(title_text='Price ($)',   row=1)
    fig.update_yaxes(title_text='OFI',         row=2)
    fig.update_yaxes(title_text='Spread (bps)',row=3)
    return fig


def multilevel_ofi_chart(ofi_df: pd.DataFrame,
                           level_cols: list,
                           timestamps: pd.Series) -> go.Figure:
    """Stacked area chart of OFI across all LOB levels."""
    fig = go.Figure()
    for i, col in enumerate(level_cols):
        fig.add_trace(go.Scatter(
            x=timestamps, y=ofi_df[col],
            name=col.replace('ofi_level_', 'Level '),
            line=dict(width=1),
            opacity=0.7,
        ))

    fig.update_layout(
        title='Multi-Level OFI — All LOB Levels',
        xaxis_title='Time', yaxis_title='Normalized OFI',
        height=450, **THEME
    )
    return fig


def pca_weights_chart(weights: np.ndarray,
                       level_cols: list,
                       evr: float) -> go.Figure:
    """Bar chart of PCA weights — which levels matter most?"""
    labels = [c.replace('ofi_level_', 'Level ') for c in level_cols]
    colors = [C[2] if w >= 0 else C[3] for w in weights]

    fig = go.Figure(go.Bar(
        x=labels, y=weights,
        marker_color=colors, opacity=0.85,
        text=[f'{w:.3f}' for w in weights],
        textposition='outside',
    ))
    fig.add_hline(y=0, line_dash='dot', line_color='gray')
    fig.update_layout(
        title=f'PCA Weights for Integrated OFI<br>Explained Variance: {evr*100:.1f}%',
        yaxis_title='Weight (L1 normalized)',
        height=400, **THEME
    )
    return fig


def price_impact_horizon_chart(horizon_df: pd.DataFrame) -> go.Figure:
    """Beta coefficient and R² across forecast horizons."""
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=['β coefficient vs Horizon',
                                         'R² vs Horizon'])

    colors = [C[2] if row['significant'] else C[1]
              for _, row in horizon_df.iterrows()]

    fig.add_trace(go.Bar(
        x=horizon_df['horizon'], y=horizon_df['beta'],
        marker_color=colors, opacity=0.85,
        error_y=dict(type='data',
                      array=(horizon_df['beta'].abs() / horizon_df['t_stat'].abs()).fillna(0),
                      visible=True),
        name='Beta',
        text=[f't={r["t_stat"]:.2f}' for _, r in horizon_df.iterrows()],
        textposition='outside',
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='gray', row=1, col=1)

    fig.add_trace(go.Bar(
        x=horizon_df['horizon'], y=horizon_df['r_squared'],
        marker_color=C[0], opacity=0.85, name='R²',
    ), row=1, col=2)

    fig.update_layout(
        title='OFI Price Impact Across Horizons<br>(green=significant at 5%, orange=not significant)',
        height=400, showlegend=False, **THEME
    )
    fig.update_xaxes(title_text='Horizon (ticks)', row=1, col=1)
    fig.update_xaxes(title_text='Horizon (ticks)', row=1, col=2)
    fig.update_yaxes(title_text='β (price impact)', row=1, col=1)
    fig.update_yaxes(title_text='R²',               row=1, col=2)
    return fig


def level_r2_chart(level_r2_df: pd.DataFrame) -> go.Figure:
    """R² by LOB level — which levels predict price?"""
    df = level_r2_df.sort_values('r_squared', ascending=True)
    labels = [c.replace('ofi_level_', 'Level ') for c in df['level']]
    colors = [C[2] if p < 0.05 else C[1] for p in df['p_value']]

    fig = go.Figure(go.Bar(
        x=df['r_squared'], y=labels,
        orientation='h',
        marker_color=colors, opacity=0.85,
        text=[f'R²={r:.4f}' for r in df['r_squared']],
        textposition='outside',
    ))
    fig.update_layout(
        title='R² by LOB Level<br>(green=significant at 5%)',
        xaxis_title='R²', height=400, **THEME
    )
    return fig


def rolling_impact_chart(rolling_df: pd.DataFrame) -> go.Figure:
    """Time-varying price impact coefficient."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=['Rolling Beta (Price Impact)', 'Rolling R²'],
                         row_heights=[0.6, 0.4], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=rolling_df['ts_event'], y=rolling_df['beta'],
        line=dict(color=C[0], width=2), name='Beta',
        fill='tozeroy', fillcolor='rgba(76,110,245,0.1)'
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='gray', row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rolling_df['ts_event'], y=rolling_df['r_squared'],
        line=dict(color=C[2], width=1.5), name='R²'
    ), row=2, col=1)

    fig.update_layout(
        title='Time-Varying Price Impact (Rolling OLS)',
        height=500, **THEME
    )
    fig.update_yaxes(title_text='β',  row=1)
    fig.update_yaxes(title_text='R²', row=2)
    return fig


def spread_depth_chart(df: pd.DataFrame) -> go.Figure:
    """Spread and depth distribution analysis."""
    fig = make_subplots(rows=2, cols=2,
                         subplot_titles=['Spread Distribution',
                                         'Best Bid/Ask Depth',
                                         'Queue Imbalance Over Time',
                                         'Spread vs Time'])

    fig.add_trace(go.Histogram(
        x=df['spread_bps'], nbinsx=50,
        marker_color=C[0], opacity=0.8, name='Spread'
    ), row=1, col=1)

    total_depth = df['bid_sz_00'] + df['ask_sz_00']
    fig.add_trace(go.Histogram(
        x=total_depth.clip(0, total_depth.quantile(0.99)),
        nbinsx=40, marker_color=C[1], opacity=0.8, name='Depth'
    ), row=1, col=2)

    queue_imb = (df['bid_sz_00'] - df['ask_sz_00']) / (df['bid_sz_00'] + df['ask_sz_00'] + 1)
    fig.add_trace(go.Scatter(
        x=df['ts_event'], y=queue_imb,
        line=dict(color=C[4], width=0.8), name='Queue Imb'
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='gray', row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df['ts_event'], y=df['spread_bps'],
        line=dict(color=C[3], width=0.8), name='Spread (bps)'
    ), row=2, col=2)

    fig.update_layout(title='Microstructure Analysis — Spread & Depth',
                       height=550, showlegend=False, **THEME)
    fig.update_xaxes(title_text='Spread (bps)', row=1, col=1)
    fig.update_xaxes(title_text='Total Depth',  row=1, col=2)
    return fig
