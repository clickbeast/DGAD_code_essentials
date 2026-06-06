
import os
import re
import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
from collections import defaultdict
from datetime import datetime

#plotly visualiztion
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np




@dataclass
class ViolinParams:
    """Configuration for violin plot rendering."""
    # One or more numeric columns to visualize; multiple columns render as subplots
    value_column: Optional[List[str]] = None  # default resolved to ['rmsd'] in function
    show_violin: bool = True
    show_box: bool = True              # inner box (quartiles) inside the violin
    show_points: bool = False          # show individual observations
    show_mean: bool = True             # mean marker and meanline inside violin
    show_median: bool = True           # explicit median marker (box already shows a median line)
    show_extrema: bool = True          # min/max markers
    show_quartiles: bool = True        # convenience toggle; maps to inner box visibility
    show_whiskers: bool = True         # draw whisker lines using Tukey method
    use_probability_density: bool = True  # maps to scalemode; True -> equal width (density), False -> count
    # Sizing controls
    box_width: float = 0.35            # width of the overlaid box (0..1 of category span)
    whisker_width: float = 0.25        # width of the whisker caps
    violin_width: Optional[float] = None  # if set, controls violin width (0..1 of category span)


def plot_violin_distributions(
    df_dict: Dict[str, pd.DataFrame],
    cdr_target: str = 'H_CDR3',
    title: str = "Distribution Comparison (Violin)",
    params: Optional[ViolinParams] = None,
    confidence_level: Optional[float] = None,
    value_range: Optional[Tuple[float, float]] = None,
    width: int = 1000,
    height: int = 600,
    category_order: Optional[List[str]] = None,
    rename_dict: Optional[Dict[str, str]] = None,
):
    """
    Create grouped violin plots for multiple datasets (and optional multiple metrics) using Plotly.

    Arguments
    - df_dict: {dataset_name: DataFrame}
    - cdr_target: filter on df['cdr_target']
    - title: plot title
    - params: ViolinParams config; value_column may be a list of columns
    - confidence_level: optional CI to trim outliers (Tukey whiskers are still computed on remaining data)
    - value_range: optional (min, max) to clip values before plotting
    - width/height: figure size
    - category_order: optional ordering of datasets on the x-axis

    Returns
    - fig: plotly.graph_objects.Figure
    """

    if params is None:
        params = ViolinParams()
    value_cols = params.value_column or ['rmsd']

    # Resolve dataset order
    dataset_names = list(df_dict.keys())
    if category_order:
        # Keep only the ones present and in given order
        dataset_names = [d for d in category_order if d in df_dict]
    default_colors = px.colors.qualitative.Plotly

    # Helper to get display name
    def get_display_name(dn):
        if rename_dict and dn in rename_dict:
            return rename_dict[dn]
        return dn

    # Build subplots if multiple metrics
    from plotly.subplots import make_subplots
    ncols = len(value_cols)
    fig = make_subplots(
        rows=1,
        cols=ncols,
        shared_yaxes=False,
        subplot_titles=[(f"{vc.upper()} (Å)" if vc == 'rmsd' else vc) for vc in value_cols],
        horizontal_spacing=0.08 if ncols > 1 else 0.02,
    )

    # Helper to darken a hex color slightly for lines/whiskers
    def darker(hex_color: str, delta: int = 50) -> str:
        r = max(0, int(hex_color[1:3], 16) - delta)
        g = max(0, int(hex_color[3:5], 16) - delta)
        b = max(0, int(hex_color[5:7], 16) - delta)
        return f'rgb({r}, {g}, {b})'

    # Pre-filter and cache data per dataset and metric
    filtered_data: dict[str, dict[str, np.ndarray]] = {dn: {} for dn in dataset_names}

    for dn in dataset_names:
        df = df_dict[dn]
        for vc in value_cols:
            df_f = df[(df['cdr_target'] == cdr_target) & (df[vc].notna())].copy()
            values = df_f[vc].values
            original_count = len(values)
            if original_count == 0:
                continue

            # Range filtering
            if value_range is not None:
                lo, hi = value_range
                values = values[(values >= lo) & (values <= hi)]

            # CI filtering
            if confidence_level is not None and values.size:
                alpha = 1 - confidence_level
                lo_p = (alpha / 2) * 100
                hi_p = (1 - alpha / 2) * 100
                lo_v = np.percentile(values, lo_p)
                hi_v = np.percentile(values, hi_p)
                values = values[(values >= lo_v) & (values <= hi_v)]

            if values.size:
                filtered_data[dn][vc] = values

    # Add traces per metric (subplot) and dataset
    for col_idx, vc in enumerate(value_cols, start=1):
        x_axis_label = f"{vc.upper()} (Å)" if vc == 'rmsd' else vc
        for i, dn in enumerate(dataset_names):
            if vc not in filtered_data.get(dn, {}):
                continue
            values = filtered_data[dn][vc]
            color = default_colors[i % len(default_colors)]
            line_color = darker(color, 60)

            # Stats
            mean_val = float(np.mean(values))
            median_val = float(np.median(values))
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            whisk_low = max(values.min(), q1 - 1.5 * iqr)
            whisk_high = min(values.max(), q3 + 1.5 * iqr)
            min_v, max_v = float(np.min(values)), float(np.max(values))

            display_name = get_display_name(dn)

            if params.show_violin:
                fig.add_trace(
                    go.Violin(
                        y=values,
                        x=[display_name] * len(values),
                        name=f"{display_name} (n={len(values)})",
                        box=dict(visible=(params.show_box or params.show_quartiles)),
                        meanline=dict(visible=params.show_mean),
                        points='all' if params.show_points else False,
                        marker=dict(color=color, opacity=0.6),
                        line=dict(color=line_color, width=1),
                        legendgroup=display_name,
                        alignmentgroup=display_name,
                        offsetgroup=display_name,
                        width=params.violin_width,
                        scalemode='width' if params.use_probability_density else 'count',
                        hovertemplate=(
                            f"<b>{display_name}</b><br>"
                            f"Count: {len(values)}<br>"
                            f"Mean: {mean_val:.3f}{' Å' if vc == 'rmsd' else ''}<br>"
                            f"Median: {median_val:.3f}{' Å' if vc == 'rmsd' else ''}<extra></extra>"
                        ),
                        showlegend=True if col_idx == 1 else False,
                    ),
                    row=1,
                    col=col_idx,
                )

            # Add a transparent Box trace to render whiskers aligned with the violin
            if params.show_whiskers or (params.show_quartiles and not params.show_violin):
                fig.add_trace(
                    go.Box(
                        y=values,
                        x=[display_name] * len(values),
                        name=f"{display_name} whiskers",
                        legendgroup=display_name,
                        alignmentgroup=display_name,
                        offsetgroup=display_name,
                        boxpoints=False,
                        whiskerwidth=params.whisker_width,
                        width=params.box_width,
                        marker=dict(opacity=0),
                        fillcolor='rgba(0,0,0,0)',
                        line=dict(color=line_color, width=2),
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    row=1,
                    col=col_idx,
                )

            # Mean and median markers
            if params.show_mean:
                fig.add_trace(
                    go.Scatter(
                        x=[display_name],
                        y=[mean_val],
                        mode='markers',
                        marker=dict(symbol='circle', size=8, color=line_color),
                        name=f"{display_name} mean",
                        showlegend=False,
                        hovertemplate=f"<b>{display_name} mean</b>: {mean_val:.3f}{' Å' if vc == 'rmsd' else ''}<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )
            if params.show_median:
                fig.add_trace(
                    go.Scatter(
                        x=[display_name],
                        y=[median_val],
                        mode='markers',
                        marker=dict(symbol='diamond', size=7, color=color),
                        name=f"{display_name} median",
                        showlegend=False,
                        hovertemplate=f"<b>{display_name} median</b>: {median_val:.3f}{' Å' if vc == 'rmsd' else ''}<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

            # Extrema markers
            if params.show_extrema:
                fig.add_trace(
                    go.Scatter(
                        x=[display_name, display_name],
                        y=[min_v, max_v],
                        mode='markers',
                        marker=dict(symbol='x', size=7, color=line_color),
                        name=f"{display_name} min/max",
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{display_name} min/max</b><br>"
                            f"min: {min_v:.3f}{' Å' if vc == 'rmsd' else ''}<br>"
                            f"max: {max_v:.3f}{' Å' if vc == 'rmsd' else ''}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col_idx,
                )

        # X/Y titles per subplot
        # Use display names for x-axis
        display_names = [get_display_name(dn) for dn in dataset_names]
        fig.update_xaxes(
            title_text="Dataset" if ncols == 1 else "",
            categoryorder='array',
            categoryarray=display_names,
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            title_text=x_axis_label if ncols == 1 else x_axis_label,
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title={
            'text': f"{title} - {cdr_target}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16},
        },
        width=width,
        height=height,
        template='plotly_white',
        violingap=0.25,
        violinmode='group',
        legend=dict(
            orientation='v',
            yanchor='top', y=1,
            xanchor='left', x=1.02,
            title="Legend",
            itemsizing='constant',
        ),
    )

    # Add custom legend entries for mean and median markers
    # These are invisible scatter traces just for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(symbol='circle', size=8, color='black'),
        name='Mean',
        showlegend=True,
        legendgroup='mean',
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(symbol='diamond', size=7, color='gray'),
        name='Median',
        showlegend=True,
        legendgroup='median',
    ))

    fig.show()
    return fig


def plot_rmsd_distributions_overlay(df_dict, cdr_target='H_CDR3', title="RMSD Distribution Comparison", 
                                   value_column='rmsd', show_histogram=True, show_kde=True, fill_kde=True, 
                                   use_probability_density=True, nbins=50, confidence_level=None, value_range=None):
    """
    Create an overlaid histogram/density plot of distributions using Plotly.
    
    Parameters:
    - df_dict: Dictionary with dataset names as keys and DataFrames as values
    - cdr_target: CDR region to filter for
    - title: Plot title
    - value_column: Column name to analyze (default: 'rmsd')
    - show_histogram: Boolean to show/hide histogram bars (default: True)
    - show_kde: Boolean to show/hide KDE overlay (default: True)
    - fill_kde: Boolean to fill area under KDE curve (default: True)
    - use_probability_density: Boolean to use probability density vs. absolute counts (default: True)
    - nbins: Number of bins for histogram (default: 50)
    - confidence_level: Float (e.g., 0.95) to trim outliers. None shows full distribution (default: None)
    - value_range: Tuple (min, max) to filter values. None shows all values (default: None)
    """
    
    fig = go.Figure()
    
    # Let plotly pick default colors automatically
    default_colors = px.colors.qualitative.Plotly
    
    # Set histogram normalization based on user preference
    histnorm = 'probability density' if use_probability_density else None
    y_axis_title = "Probability Density" if use_probability_density else "Count"
    
    # Determine axis labels based on value column
    x_axis_label = f"{value_column.upper()} (Å)" if value_column == 'rmsd' else value_column
    
    # First pass: collect all filtered data to determine consistent bin edges
    all_filtered_data = {}
    global_min, global_max = float('inf'), float('-inf')
    
    for dataset_name, df in df_dict.items():
        # Filter for CDR target and remove NaN values
        df_filtered = df[(df['cdr_target'] == cdr_target) & (df[value_column].notna())].copy()
        
        if len(df_filtered) == 0:
            print(f"No valid data for {dataset_name}")
            continue
            
        values = df_filtered[value_column].values
        original_count = len(values)
        
        # Apply value range filtering first (if specified)
        if value_range is not None:
            min_val, max_val = value_range
            values = values[(values >= min_val) & (values <= max_val)]
            range_filtered_count = len(values)
            print(f"\n{dataset_name} - Range filtering [{min_val}, {max_val}]: {original_count} → {range_filtered_count} samples")
        
        # Apply confidence interval filtering (if specified)
        if confidence_level is not None:
            alpha = 1 - confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            ci_lower = np.percentile(values, lower_percentile)
            ci_upper = np.percentile(values, upper_percentile)
            
            values = values[(values >= ci_lower) & (values <= ci_upper)]
            ci_filtered_count = len(values)
            print(f"{dataset_name} - CI filtering {confidence_level*100:.0f}% [{ci_lower:.3f}, {ci_upper:.3f}]: {range_filtered_count if value_range else original_count} → {ci_filtered_count} samples")
        
        if len(values) == 0:
            print(f"No data remaining for {dataset_name} after filtering")
            continue
        
        # Store filtered data and update global range
        all_filtered_data[dataset_name] = values
        global_min = min(global_min, values.min())
        global_max = max(global_max, values.max())
    
    # Calculate consistent bin edges for all datasets
    if all_filtered_data and show_histogram:
        bin_edges = np.linspace(global_min, global_max, nbins + 1)
    
    # Second pass: create plots with consistent binning
    for i, (dataset_name, values) in enumerate(all_filtered_data.items()):
        
        # Calculate statistics on filtered data
        mean_val = np.mean(values)
        median_val = np.median(values)
        
        print(f"\n{dataset_name} Final Statistics:")
        print(f"  Count: {len(values)}")
        value_unit = " Å" if value_column == 'rmsd' else ""
        print(f"  Mean: {mean_val:.3f}{value_unit}")
        print(f"  Median: {median_val:.3f}{value_unit}")
        print(f"  Std: {np.std(values):.3f}{value_unit}")
        print(f"  Range: [{np.min(values):.3f}, {np.max(values):.3f}]{value_unit}")
        
        color = default_colors[i % len(default_colors)]
        
        # Create a darker version of the same color for the outline
        # Convert hex to RGB, darken by reducing each component by 50
        r = max(0, int(color[1:3], 16) - 50)
        g = max(0, int(color[3:5], 16) - 50)
        b = max(0, int(color[5:7], 16) - 50)
        darker_color = f'rgb({r}, {g}, {b})'
        
        # Add histogram with consistent binning if requested
        if show_histogram:
            fig.add_trace(go.Histogram(
                x=values,
                name=f'{dataset_name} (n={len(values)})',
                opacity=0.7,  # Semi-transparent for overlaying
                histnorm=histnorm,
                xbins=dict(
                    start=bin_edges[0],
                    end=bin_edges[-1],
                    size=(bin_edges[-1] - bin_edges[0]) / nbins
                ),
                marker=dict(
                    color=color,
                    line=dict(
                        color=darker_color,  # Darker version of the same color
                        width=1
                    )
                ),
                showlegend=True
            ))
        
        # Add KDE if requested
        if show_kde:
            try:
                from scipy import stats
                # Create KDE
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                kde_values = kde(x_range)
                
                # Scale KDE values if using absolute counts instead of probability density
                if not use_probability_density:
                    # Scale KDE to match histogram scale (approximate)
                    bin_width = (values.max() - values.min()) / nbins
                    kde_values = kde_values * len(values) * bin_width
                
                # Determine KDE name and legend visibility
                kde_name = f'{dataset_name} KDE' if not show_histogram else f'{dataset_name} KDE'
                kde_show_legend = not show_histogram  # Show KDE in legend only when histogram is off
                
                if fill_kde:
                    # Add filled KDE curve
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde_values,
                        fill='tozeroy',  # Fill to x-axis
                        mode='lines',
                        name=f'{dataset_name} (n={len(values)})' if not show_histogram else kde_name,
                        line=dict(color=color, width=2),
                        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)',  # 30% opacity
                        showlegend=kde_show_legend
                    ))
                else:
                    # Add KDE line only (no fill)
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde_values,
                        mode='lines',
                        name=f'{dataset_name} (n={len(values)})' if not show_histogram else kde_name,
                        line=dict(color=color, width=2),
                        showlegend=kde_show_legend
                    ))
            except ImportError:
                print("Warning: scipy not available for KDE. Install scipy to enable KDE overlay.")
                if not show_histogram:
                    # If no histogram and no KDE possible, add a dummy trace for legend
                    fig.add_trace(go.Scatter(
                        x=[],
                        y=[],
                        mode='markers',
                        name=f'{dataset_name} (n={len(values)})',
                        marker=dict(color=color),
                        showlegend=True
                    ))
        
        # Add mean line with hover tooltip (no annotation text)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color=color,
            line_width=2
        )
        
        # Add invisible scatter point for hover tooltip on mean line
        fig.add_trace(go.Scatter(
            x=[mean_val],
            y=[0],
            mode='markers',
            marker=dict(size=0.1, color=color, opacity=0),
            hovertemplate=f"<b>{dataset_name} Mean</b><br>{value_column}: {mean_val:.3f}{' Å' if value_column == 'rmsd' else ''}<extra></extra>",
            showlegend=False,
            name=f'{dataset_name}_mean_hover'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'{title} - {cdr_target}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_title,
        barmode='overlay',
        width=900,
        height=600,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Show the plot
    fig.show()
    
    return fig

# Create the overlay plot with different options

# Option 1: Full distribution with filled KDE and probability density (default - RMSD)
# fig = plot_rmsd_distributions_overlay(
#     {'SIMS': df_sims, 'Original': df_original}, 
#     cdr_target='H_CDR3',
#     title="RMSD Distribution Comparison - Full Data",
#     value_column='rmsd',  # Default, can be omitted
#     show_histogram=True,
#     show_kde=True,
#     fill_kde=True,
#     use_probability_density=True,
#     nbins=40,
#     #value_range=(0,20)
# )

# Option 2: Analyze a different column (uncomment to use)
# fig = plot_rmsd_distributions_overlay(
#     {'SIMS': df_sims, 'Original': df_original}, 
#     cdr_target='H_CDR3',
#     title="Sample ID Distribution Comparison",
#     value_column='sample_id',  # Analyze sample_id instead of rmsd
#     show_histogram=True,
#     show_kde=True,
#     fill_kde=True,
#     use_probability_density=True,
#     nbins=40
# )

# Option 3: Absolute counts instead of probability density (uncomment to use)
# fig = plot_rmsd_distributions_overlay(
#     {'SIMS': df_sims, 'Original': df_original}, 
#     cdr_target='H_CDR3',
#     title="RMSD Distribution Comparison - Absolute Counts",
#     value_column='rmsd',
#     show_histogram=True,
#     show_kde=True,
#     fill_kde=True,
#     use_probability_density=False,
#     nbins=40
# )

# Option 4: KDE lines without fill (uncomment to use)
# fig = plot_rmsd_distributions_overlay(
#     {'SIMS': df_sims, 'Original': df_original}, 
#     cdr_target='H_CDR3',
#     title="RMSD Distribution Comparison - KDE Lines Only",
#     value_column='rmsd',
#     show_histogram=True,
#     show_kde=True,
#     fill_kde=False,
#     use_probability_density=True,
#     nbins=40
# )

# Option 5: Histograms only (no KDE) (uncomment to use)
# fig = plot_rmsd_distributions_overlay(
#     {'SIMS': df_sims, 'Original': df_original}, 
#     cdr_target='H_CDR3',
#     title="RMSD Distribution Comparison - Histograms Only",
#     value_column='rmsd',
#     show_histogram=True,
#     show_kde=False,
#     nbins=40
# )

# Option 6: KDE only (no histograms) (uncomment to use)
# fig = plot_rmsd_distributions_overlay(
#     {'SIMS': df_sims, 'Original': df_original}, 
#     cdr_target='H_CDR3',
#     title="RMSD Distribution Comparison - KDE Only",
#     value_column='rmsd',
#     show_histogram=False,
#     show_kde=True,
#     fill_kde=True,
#     use_probability_density=True,
#     nbins=40
# )

# Option 7: With filtering options (uncomment to use)
# fig = plot_rmsd_distributions_overlay(
#     {'SIMS': df_sims, 'Original': df_original}, 
#     cdr_target='H_CDR3',
#     title="RMSD Distribution Comparison - Range [0, 8] + 95% CI",
#     value_column='rmsd',
#     show_histogram=True,
#     show_kde=True,
#     fill_kde=True,
#     use_probability_density=True,
#     nbins=40,
#     value_range=(0, 8),
#     confidence_level=0.95
# )


#if __name__ == '__main__':
    #Example usage
    # fig = plot_rmsd_distributions_overlay(
    # df_all, 
    # cdr_target='H_CDR3',
    # title="RMSD Distribution Comparison - Full Data",
    # value_column='rmsd',  # Default, can be omitted
    # show_histogram=False,
    # show_kde=True,
    # fill_kde=True,
    # use_probability_density=True,
    # nbins=40#
    # )



def plot_rmsd_summary_bars(
    df_dict,
    cdr_target='H_CDR3',
    title="Experiment Summary (min/mean/max)",
    value_columns=('rmsd',),
    confidence_level=None,            # e.g. 0.95 → trim to central 95% BEFORE computing stats
    value_range=None,                 # (lo, hi) clip BEFORE trimming
    cap_percentile=0.95,              # cap max at 95th and min at 5th; true extrema get markers
    errorbar_percentiles=(5, 95),     # mean error bars span these percentiles (asymmetric)
    show_outlier_markers=True,
    width=1200,
    height=500,
    category_order=None,              # order of experiments on x-axis
    reverse_y=False,                  # True if you want “lower is better” visually emphasized
):
    """
    For each experiment (key in df_dict), draw three grouped bars:
      - MIN (outline only), capped at lower cap percentile
      - MEAN (filled), with asymmetric error bars spanning errorbar_percentiles
      - MAX (outline only), capped at upper cap percentile
    True extrema beyond the caps are shown as triangle markers.

    Multiple metrics are supported via value_columns; each metric gets its own subplot column.
    Colors encode experiments; min/max are transparent fill with colored outlines; mean is filled.
    """

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import pandas as pd

    # ---- helpers ----
    def darker(hex_color: str, delta: int = 50) -> str:
        if hex_color.startswith("rgb"):
            return hex_color
        r = max(0, int(hex_color[1:3], 16) - delta)
        g = max(0, int(hex_color[3:5], 16) - delta)
        b = max(0, int(hex_color[5:7], 16) - delta)
        return f"rgb({r},{g},{b})"

    # Validate percentiles
    lo_p, hi_p = errorbar_percentiles
    if not (0 <= lo_p < hi_p <= 100):
        raise ValueError("errorbar_percentiles must be an increasing pair within [0, 100].")
    if not (0 < cap_percentile <= 1):
        raise ValueError("cap_percentile must be in (0, 1].")

    datasets = list(df_dict.keys())
    if category_order:
        datasets = [dn for dn in category_order if dn in df_dict] + [dn for dn in df_dict if dn not in set(category_order)]

    value_cols = list(value_columns) if isinstance(value_columns, (list, tuple)) else [value_columns]
    ncols = len(value_cols)

    palette = px.colors.qualitative.Plotly
    color_map = {dn: palette[i % len(palette)] for i, dn in enumerate(datasets)}

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=value_cols, horizontal_spacing=0.08)

    for ci, vc in enumerate(value_cols, start=1):
        x_cats = datasets

        # Collect stats per dataset
        means, true_mins, true_maxs = [], [], []
        p_los, p_his = [], []
        capped_min, capped_max = [], []

        for dn in datasets:
            df = df_dict[dn]
            # robust selection: if 'cdr_target' column absent, .get() returns None; compare safely
            s = df[vc].astype(float)
            if 'cdr_target' in df.columns:
                s = s[df['cdr_target'] == cdr_target]
            s = s[s.notna()]

            # Clip numeric range if requested
            if value_range is not None:
                lo, hi = value_range
                s = s.clip(lower=lo, upper=hi)

            # Trim to central confidence interval if requested (like overlay function)
            if confidence_level is not None and 0 < confidence_level < 1:
                alpha = (1 - confidence_level) / 2.0
                lo_q, hi_q = s.quantile(alpha), s.quantile(1 - alpha)
                s = s[(s >= lo_q) & (s <= hi_q)]

            vals = s.to_numpy()
            if vals.size == 0:
                means.append(None); true_mins.append(None); true_maxs.append(None)
                p_los.append(None); p_his.append(None)
                capped_min.append(None); capped_max.append(None)
                continue

            # Core stats
            mn = float(np.min(vals))
            mx = float(np.max(vals))
            mean = float(np.mean(vals))
            p_lo = float(np.percentile(vals, lo_p))
            p_hi = float(np.percentile(vals, hi_p))

            # Capping for visual stability
            cap_hi = float(np.percentile(vals, cap_percentile * 100))   # e.g., 95th
            cap_lo = float(np.percentile(vals, (1 - cap_percentile) * 100))  # e.g., 5th

            means.append(mean)
            true_mins.append(mn); true_maxs.append(mx)
            p_los.append(p_lo); p_his.append(p_hi)
            capped_min.append(max(mn, cap_lo))
            capped_max.append(min(mx, cap_hi))

        # Asymmetric error for MEAN (percentiles lo_p–hi_p)
        err_up = []
        err_dn = []
        for m, plo, phi in zip(means, p_los, p_his):
            if m is None or plo is None or phi is None:
                err_dn.append(0); err_up.append(0)
            else:
                err_dn.append(max(0, m - plo))
                err_up.append(max(0, phi - m))

        # Colors per experiment
        colors = [color_map[dn] for dn in x_cats]
        line_colors = [darker(color_map[dn], 60) for dn in x_cats]

        # MIN (outline only, capped)
        fig.add_bar(
            row=1, col=ci,
            x=x_cats, y=capped_min,
            name="min",
            marker=dict(color='rgba(0,0,0,0)', line=dict(color=colors, width=2)),
            hovertemplate="<b>%{x}</b><br>min (capped): %{y:.4f}"
                          "<br><i>true min</i>: %{customdata:.4f}<extra></extra>",
            customdata=true_mins,
            offsetgroup="min",
            legendgroup="stats",
            showlegend=(ci == 1)
        )

        # MEAN (filled, with asymmetric error)
        fig.add_bar(
            row=1, col=ci,
            x=x_cats, y=means,
            name="mean",
            marker=dict(color=colors, line=dict(color=line_colors, width=1.5)),
            error_y=dict(type='data', array=err_up, arrayminus=err_dn, visible=True),
            hovertemplate=(f"<b>%{{x}}</b><br>mean: %{{y:.4f}}"
                           f"<br>{lo_p:g}–{hi_p:g} pct: %{{customdata}}<extra></extra>"),
            customdata=[f"{plo:.4f} – {phi:.4f}" if (plo is not None and phi is not None) else "n/a"
                        for plo, phi in zip(p_los, p_his)],
            offsetgroup="mean",
            legendgroup="stats",
            showlegend=(ci == 1)
        )

        # MAX (outline only, capped)
        fig.add_bar(
            row=1, col=ci,
            x=x_cats, y=capped_max,
            name="max",
            marker=dict(color='rgba(0,0,0,0)', line=dict(color=colors, width=2)),
            hovertemplate="<b>%{x}</b><br>max (capped): %{y:.4f}"
                          "<br><i>true max</i>: %{customdata:.4f}<extra></extra>",
            customdata=true_maxs,
            offsetgroup="max",
            legendgroup="stats",
            showlegend=(ci == 1)
        )

        # True extrema markers if capped
        if show_outlier_markers:
            # True MAX markers (triangle-up) where mx > capped_max
            x_max_mark = [x for x, mx, cx in zip(x_cats, true_maxs, capped_max)
                          if (mx is not None and cx is not None and mx > cx)]
            y_max_mark = [mx for mx, cx in zip(true_maxs, capped_max)
                          if (mx is not None and cx is not None and mx > cx)]
            c_max_mark = [color_map[x] for x in x_max_mark]
            if x_max_mark:
                fig.add_scatter(
                    row=1, col=ci,
                    x=x_max_mark, y=y_max_mark,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color=c_max_mark,
                                line=dict(color='black', width=1)),
                    name="true max",
                    hovertemplate="<b>%{x}</b><br>true max: %{y:.4f}<extra></extra>",
                    legendgroup="extrema",
                    showlegend=(ci == 1)
                )

            # True MIN markers (triangle-down) where mn < capped_min
            x_min_mark = [x for x, mn, cn in zip(x_cats, true_mins, capped_min)
                          if (mn is not None and cn is not None and mn < cn)]
            y_min_mark = [mn for mn, cn in zip(true_mins, capped_min)
                          if (mn is not None and cn is not None and mn < cn)]
            c_min_mark = [color_map[x] for x in x_min_mark]
            if x_min_mark:
                fig.add_scatter(
                    row=1, col=ci,
                    x=x_min_mark, y=y_min_mark,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color=c_min_mark,
                                line=dict(color='black', width=1)),
                    name="true min",
                    hovertemplate="<b>%{x}</b><br>true min: %{y:.4f}<extra></extra>",
                    legendgroup="extrema",
                    showlegend=(ci == 1)
                )

        # Add invisible dummy traces so the legend shows color→experiment mapping once
        if ci == ncols:
            for dn in datasets:
                fig.add_scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color=color_map[dn]),
                    name=dn,
                    legendgroup=f"exp-{dn}",
                    showlegend=True
                )

    fig.update_layout(
        barmode='group',
        title=title,
        width=width,
        height=height,
        legend_title_text="Legend",
        bargap=0.25,
        bargroupgap=0.08,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Experiment", categoryorder="array", categoryarray=datasets)
    fig.update_yaxes(title_text="Value", autorange="reversed" if reverse_y else True)

    return fig


def plot_rmsd_summary_bars_seaborn(
    df_dict,
    cdr_target='H_CDR3',
    title="Experiment Summary (min/mean/max)",
    value_columns=('rmsd',),
    confidence_level=None,            # trim ONLY for CI whiskers; mean is raw
    value_range=None,                 # (lo, hi) clip BEFORE trimming
    cap_percentile=0.95,              # cap max at 95th and min at 5th; true extrema get markers
    errorbar_percentiles=(5, 95),     # mean error bars span these percentiles (asymmetric)
    show_outlier_markers=True,        # (back-compat) kept; see show_extrema_markers below
    width_per_metric=8.0,             # wider default so long labels fit
    height=5.0,
    category_order=None,              # order of experiments on x-axis
    reverse_y=False,                  # invert y-axis (useful if “lower is better”)
    dpi=120,
    xtick_rotation=35,                # angle for long experiment names
    xtick_fontsize=9,                 # smaller font size for long labels

    # --- Flexible toggles (all default ON) ---
    show_min_max_bars=True,
    show_mean_errorbars=True,
    show_extrema_markers=None,        # if None, falls back to show_outlier_markers
    show_experiment_names=True,
    show_mean_value=True,             # NEW: show raw mean as text above each mean bar
    show_yaxis_label=True,            # NEW: show y-axis "Value" label
    magnify=False,                    # NEW: zoom into the mean value region
):
    """
    Seaborn/Matplotlib grouped summary:
      - 3 bars per experiment: MIN (outline), MEAN (filled + percentile error bars), MAX (outline)
      - Min/max bars are capped at (1-cap_percentile)/cap_percentile (e.g., 5th/95th)
      - True extrema beyond caps can be shown as triangle markers
      - Multiple metrics -> horizontal subplots (one per value column)
      - Colors encode experiments; same color for the 3 bars of an experiment
      - NEW: `magnify=True` creates a broken y-axis to zoom in on the mean values.

    Conventions:
      - Bar height & printed value = RAW arithmetic mean (after optional value_range clipping, NO trimming).
      - Error bars (whiskers) = percentiles computed on data *optionally trimmed* by confidence_level.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec

    # --- theme ---
    sns.set_theme(style="whitegrid", context="talk")

    # --- validate inputs ---
    lo_p, hi_p = errorbar_percentiles
    if not (0 <= lo_p < hi_p <= 100):
        raise ValueError("errorbar_percentiles must be an increasing pair within [0, 100].")
    if not (0 < cap_percentile <= 1):
        raise ValueError("cap_percentile must be in (0, 1].")
    # Resolve back-compat flag
    if show_extrema_markers is None:
        show_extrema_markers = bool(show_outlier_markers)

    # --- figure layout ---
    value_cols = list(value_columns) if isinstance(value_columns, (list, tuple)) else [value_columns]
    ncols = len(value_cols)
    
    fig_height = height * 1.8 if magnify else height
    fig = plt.figure(figsize=(max(6.0, width_per_metric * ncols), fig_height), dpi=dpi)
    
    gs = GridSpec(1, ncols, figure=fig)

    # --- experiments order & colors ---
    experiments = list(df_dict.keys())
    if category_order:
        experiments = [dn for dn in category_order if dn in df_dict] + [dn for dn in df_dict if dn not in set(category_order)]

    palette = sns.color_palette("colorblind", n_colors=max(7, len(experiments)))
    color_map = {exp: palette[i % len(palette)] for i, exp in enumerate(experiments)}

    # --- helpers ---
    def _series_raw(df, metric):
        s = df[metric].astype(float)
        if 'cdr_target' in df.columns:
            s = s[df['cdr_target'] == cdr_target]
        s = s.dropna()
        if value_range is not None:
            lo, hi = value_range
            s = s.clip(lower=lo, upper=hi)
        return s

    def _series_for_ci(s):
        if confidence_level is None or not (0 < confidence_level < 1):
            return s
        alpha = (1 - confidence_level) / 2.0
        lo_q, hi_q = s.quantile(alpha), s.quantile(1 - alpha)
        return s[(s >= lo_q) & (s <= hi_q)]

    bar_width = 0.25
    offsets = (-bar_width, 0.0, bar_width)
    x_labels = experiments if show_experiment_names else [str(i) for i in range(1, len(experiments) + 1)]
    legend_labels = experiments if show_experiment_names else [f"{i+1}) {name}" for i, name in enumerate(experiments)]

    for i, metric in enumerate(value_cols):
        rows = []
        for exp in experiments:
            df = df_dict[exp]
            s_raw = _series_raw(df, metric)
            if s_raw.empty:
                rows.append(dict(exp=exp, mean=np.nan, p_lo=np.nan, p_hi=np.nan,
                                 true_min=np.nan, true_max=np.nan,
                                 cap_min=np.nan, cap_max=np.nan))
                continue
            mean = float(s_raw.mean())
            s_ci = _series_for_ci(s_raw)
            p_lo = float(np.percentile(s_ci, lo_p)) if len(s_ci) else np.nan
            p_hi = float(np.percentile(s_ci, hi_p)) if len(s_ci) else np.nan
            mn = float(np.min(s_raw))
            mx = float(np.max(s_raw))
            cap_lo = float(np.percentile(s_raw, (1 - cap_percentile) * 100))
            cap_hi = float(np.percentile(s_raw, cap_percentile * 100))
            rows.append(dict(exp=exp, mean=mean, p_lo=p_lo, p_hi=p_hi,
                             true_min=mn, true_max=mx,
                             cap_min=max(mn, cap_lo), cap_max=min(mx, cap_hi)))

        stats = pd.DataFrame(rows).dropna(subset=['mean'])
        if stats.empty:
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
            continue

        x_idx = np.arange(len(stats))
        exp_names = stats["exp"].values
        exp_colors = [color_map[e] for e in exp_names]

        # --- auto-rotation for x-ticks ---
        current_x_labels = [l for l, e in zip(x_labels, experiments) if e in exp_names]
        max_label_len = max(len(str(l)) for l in current_x_labels) if current_x_labels else 0
        rotation = xtick_rotation if max_label_len > 3 else 0
        ha = "right" if rotation > 0 else "center"

        def plot_bars(ax_plot):
            min_positions = x_idx + offsets[0]
            if show_min_max_bars:
                ax_plot.bar(min_positions, stats["cap_min"].values, width=bar_width,
                            edgecolor=exp_colors, facecolor="none", linewidth=2, label="min (capped)")
            mean_positions = x_idx + offsets[1]
            ax_plot.bar(mean_positions, stats["mean"].values, width=bar_width,
                        color=exp_colors, edgecolor="black", linewidth=0.8, label="mean")
            if show_mean_errorbars:
                err_dn = np.clip(stats["mean"].values - stats["p_lo"].values, a_min=0, a_max=None)
                err_up = np.clip(stats["p_hi"].values - stats["mean"].values, a_min=0, a_max=None)
                ax_plot.errorbar(mean_positions, stats["mean"].values, yerr=[err_dn, err_up],
                                 fmt="none", elinewidth=1.6, capsize=5, ecolor='black')
            if show_mean_value:
                for x, y in zip(mean_positions, stats["mean"].values):
                    if np.isfinite(y):
                        ax_plot.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points",
                                         ha="center", va="bottom", fontsize=9)
            max_positions = x_idx + offsets[2]
            if show_min_max_bars:
                ax_plot.bar(max_positions, stats["cap_max"].values, width=bar_width,
                            edgecolor=exp_colors, facecolor="none", linewidth=2, label="max (capped)")
            if show_extrema_markers:
                mask_max = stats["true_max"].values > stats["cap_max"].values
                if np.any(mask_max):
                    ax_plot.scatter(max_positions[mask_max], stats["true_max"].values[mask_max], marker="^", s=56,
                                    c=[color_map[exp_names[i]] for i, m in enumerate(mask_max) if m],
                                    edgecolors="black", linewidths=0.6, label="true max")
                mask_min = stats["true_min"].values < stats["cap_min"].values
                if np.any(mask_min):
                    ax_plot.scatter(min_positions[mask_min], stats["true_min"].values[mask_min], marker="v", s=56,
                                    c=[color_map[exp_names[i]] for i, m in enumerate(mask_min) if m],
                                    edgecolors="black", linewidths=0.6, label="true min")

        if magnify:
            sub_gs = gs[0, i].subgridspec(3, 1, hspace=0.1, height_ratios=[1, 4, 1])
            ax_top = fig.add_subplot(sub_gs[0])
            ax_middle = fig.add_subplot(sub_gs[1], sharex=ax_top)
            ax_bottom = fig.add_subplot(sub_gs[2], sharex=ax_top)
            axes = [ax_top, ax_middle, ax_bottom]

            min_mean = stats['mean'].min()
            max_mean = stats['mean'].max()
            data_range = max_mean - min_mean
            slack = data_range * 0.5 
            
            magnify_bottom = min_mean - slack
            magnify_top = max_mean + slack

            full_min = stats['cap_min'].min()
            full_max = stats['cap_max'].max()

            if not reverse_y:
                ax_top.set_ylim(bottom=magnify_top, top=full_max + (full_max-magnify_top)*0.1)
                ax_middle.set_ylim(bottom=magnify_bottom, top=magnify_top)
                ax_bottom.set_ylim(bottom=full_min - (magnify_bottom-full_min)*0.1, top=magnify_bottom)
            else:
                ax_top.set_ylim(top=magnify_top, bottom=full_max + (full_max-magnify_top)*0.1)
                ax_middle.set_ylim(top=magnify_bottom, bottom=magnify_top)
                ax_bottom.set_ylim(top=full_min - (magnify_bottom-full_min)*0.1, bottom=magnify_bottom)

            for ax_plot in axes:
                plot_bars(ax_plot)

            ax_top.spines['bottom'].set_visible(False)
            ax_middle.spines['top'].set_visible(False)
            ax_middle.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            
            plt.setp(ax_top.get_xticklabels(), visible=False)
            plt.setp(ax_middle.get_xticklabels(), visible=False)
            ax_top.tick_params(axis='x', which='both', bottom=False)
            ax_middle.tick_params(axis='x', which='both', bottom=False)

            d = .015
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=ax_middle.transAxes)
            ax_middle.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_middle.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            ax_middle.plot((-d, +d), (-d, +d), **kwargs)
            ax_middle.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=ax_bottom.transAxes)
            ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            
            rect = Rectangle((ax_middle.get_xlim()[0]-0.5, magnify_bottom if not reverse_y else magnify_top),
                             width=ax_middle.get_xlim()[1] - ax_middle.get_xlim()[0]+1,
                             height=abs(magnify_top - magnify_bottom),
                             transform=ax_middle.transData, facecolor='grey', alpha=0.1, zorder=-100,
                             edgecolor='grey', linestyle='--')
            ax_middle.add_patch(rect)

            ax_bottom.set_xticks(x_idx)
            ax_bottom.set_xticklabels(current_x_labels, rotation=rotation, ha=ha, fontsize=xtick_fontsize)
            ax_top.set_title(str(metric))
            if show_yaxis_label:
                ax_middle.set_ylabel("Value")
            if reverse_y:
                for ax_plot in axes:
                    ax_plot.invert_yaxis()
            sns.despine(ax=ax_top)
            sns.despine(ax=ax_middle)
            sns.despine(ax=ax_bottom)
        else:
            ax = fig.add_subplot(gs[0, i])
            plot_bars(ax)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(current_x_labels, rotation=rotation, ha=ha, fontsize=xtick_fontsize)
            ax.set_title(str(metric))
            if show_yaxis_label:
                ax.set_ylabel("Value")
            if reverse_y:
                ax.invert_yaxis()
            sns.despine(ax=ax)

    # -------- Legends OUTSIDE --------
    stat_handles = []
    if show_min_max_bars:
        stat_handles.append(Patch(facecolor="none", edgecolor="black", linewidth=2, label="min (capped)"))
    stat_handles.append(Patch(facecolor="lightgray", edgecolor="black", linewidth=0.8, label="mean"))
    if show_min_max_bars:
        stat_handles.append(Patch(facecolor="none", edgecolor="black", linewidth=2, label="max (capped)"))
    if show_extrema_markers:
        stat_handles.append(Line2D([0], [0], marker="^", linestyle="None", color="black", label="true max"))
        stat_handles.append(Line2D([0], [0], marker="v", linestyle="None", color="black", label="true min"))

    if stat_handles:
        fig.legend(handles=stat_handles, title="Statistic",
                   loc="lower left", bbox_to_anchor=(0.01, 0.01), frameon=False, ncol=len(stat_handles))

    exp_handles = [Patch(facecolor=color_map[e], edgecolor="black", label=lbl)
                   for e, lbl in zip(experiments, legend_labels)]
    fig.legend(handles=exp_handles, title="Experiment",
               loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0.05, 0.1, 0.85, 0.95])
    fig.subplots_adjust(bottom=0.25)

    return fig

