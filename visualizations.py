"""
Charts and text formatting for the e-commerce analysis.

Every chart uses the same palette and takes a ``period_label`` (see
``data_loader.describe_period``) so the date range always appears in the title.
Functions return the figure and do not call ``show()``, so callers decide how to
display or save it.
"""

import calendar
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Shared palette: one primary hue for the analysis period, a neutral grey for
# the comparison period, and a single-hue blue ramp for magnitude (maps).
COLORS = {
    'primary': '#2a78d6',
    'comparison': '#9a9893',
    'text': '#0b0b0b',
    'text_secondary': '#52514e',
    'grid': '#e4e3df',
}
SEQUENTIAL_BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as US dollars, e.g. ``$1,234.50``."""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2, signed: bool = False) -> str:
    """Format a number already expressed in percent, e.g. ``-2.46%``; NaN becomes ``n/a``."""
    if pd.isna(value):
        return 'n/a'
    return f"{value:+.{decimals}f}%" if signed else f"{value:.{decimals}f}%"


def _currency_axis(axis) -> None:
    """Show tick labels as whole dollars."""
    axis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))


def _style_axes(ax, grid_axis: str = 'y') -> None:
    """Apply the shared look: recessive grid, no top/right spines, grey labels."""
    ax.grid(axis=grid_axis, color=COLORS['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text_secondary'])
    ax.xaxis.label.set_color(COLORS['text_secondary'])
    ax.yaxis.label.set_color(COLORS['text_secondary'])
    ax.title.set_color(COLORS['text'])


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_monthly_revenue(monthly: pd.DataFrame,
                         period_label: str,
                         comparison_monthly: Optional[pd.DataFrame] = None,
                         comparison_label: Optional[str] = None,
                         figsize=(11, 5)) -> plt.Figure:
    """
    Line chart of monthly revenue, optionally overlaid with the comparison period.

    Months are plotted by position (1st, 2nd, ... month of the period) so a
    comparison period of the same length lines up month for month.

    Args:
        monthly: Output of ``BusinessMetricsCalculator.calculate_monthly_trends``.
        period_label: Analysis period label for the title and legend.
        comparison_monthly: Same table for the comparison period, or None.
        comparison_label: Comparison period label for the legend.

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    positions = range(len(monthly))
    ax.plot(positions, monthly['revenue'], color=COLORS['primary'], linewidth=2,
            marker='o', markersize=6, label=period_label)

    if comparison_monthly is not None and len(comparison_monthly):
        ax.plot(range(len(comparison_monthly)), comparison_monthly['revenue'],
                color=COLORS['comparison'], linewidth=2, linestyle='--',
                marker='o', markersize=5, label=comparison_label or 'Comparison period')
        ax.legend(frameon=False, loc='lower left')

    tick_labels = [f"{calendar.month_abbr[m]} {y}" for y, m in zip(monthly['year'], monthly['month'])]
    ax.set_xticks(list(positions))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    _currency_axis(ax.yaxis)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Monthly Revenue, {period_label}", fontsize=14, fontweight='bold', loc='left')
    ax.set_xlabel('Purchase month')
    ax.set_ylabel('Revenue (USD)')
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_category_revenue(categories: pd.DataFrame,
                          period_label: str,
                          figsize=(11, 6)) -> plt.Figure:
    """
    Horizontal bar chart of revenue by product category, largest at the top.

    Args:
        categories: Output of ``BusinessMetricsCalculator.analyze_product_performance``.
        period_label: Analysis period label for the title.

    Returns:
        plt.Figure
    """
    data = categories.sort_values('total_revenue')
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(data['product_category_name'], data['total_revenue'], color=COLORS['primary'], height=0.7)
    for y, (value, share) in enumerate(zip(data['total_revenue'], data['revenue_share'])):
        ax.text(value, y, f"  ${value / 1000:,.0f}K ({share:.1f}%)", va='center',
                fontsize=9, color=COLORS['text_secondary'])
    _currency_axis(ax.xaxis)
    ax.set_xlim(right=data['total_revenue'].max() * 1.2)
    ax.set_title(f"Revenue by Product Category, {period_label}", fontsize=14, fontweight='bold', loc='left')
    ax.set_xlabel('Revenue (USD)')
    ax.set_ylabel('Product category')
    _style_axes(ax, grid_axis='x')
    fig.tight_layout()
    return fig


def plot_revenue_by_state(states: pd.DataFrame, period_label: str) -> go.Figure:
    """
    US choropleth map of revenue by customer state.

    Args:
        states: Output of ``BusinessMetricsCalculator.analyze_geographic_performance``.
        period_label: Analysis period label for the title.

    Returns:
        go.Figure: Interactive plotly figure (hover shows revenue, orders and AOV).
    """
    fig = px.choropleth(
        states,
        locations='state',
        color='revenue',
        locationmode='USA-states',
        scope='usa',
        color_continuous_scale=SEQUENTIAL_BLUES,
        hover_data={'orders': ':,', 'avg_order_value': ':$,.2f', 'revenue': ':$,.0f'},
        labels={'revenue': 'Revenue (USD)', 'orders': 'Orders',
                'avg_order_value': 'Avg order value (USD)', 'state': 'State'},
    )
    fig.update_layout(
        title=dict(text=f"Revenue by Customer State, {period_label}", x=0.02),
        coloraxis_colorbar=dict(title='Revenue (USD)', tickprefix='$', tickformat=',.0f'),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_review_distribution(score_distribution: pd.Series,
                             period_label: str,
                             figsize=(9, 4.5)) -> plt.Figure:
    """
    Horizontal bar chart of the share of reviewed orders at each review score.

    Args:
        score_distribution: Score -> percent of reviewed orders (the
            ``score_distribution`` entry of ``analyze_customer_satisfaction``).
        period_label: Analysis period label for the title.

    Returns:
        plt.Figure
    """
    data = score_distribution.sort_index()
    labels = [f"{int(score)} star{'s' if score != 1 else ''}" for score in data.index]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, data.values, color=COLORS['primary'], height=0.65)
    for y, value in enumerate(data.values):
        ax.text(value, y, f"  {value:.1f}%", va='center', fontsize=9, color=COLORS['text_secondary'])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlim(right=data.max() * 1.15)
    ax.set_title(f"Review Score Distribution, {period_label}", fontsize=14, fontweight='bold', loc='left')
    ax.set_xlabel('Share of reviewed orders (%)')
    ax.set_ylabel('Review score')
    _style_axes(ax, grid_axis='x')
    fig.tight_layout()
    return fig


def plot_review_score_by_delivery(scores: pd.DataFrame,
                                  group_col: str,
                                  period_label: str,
                                  xlabel: str,
                                  figsize=(9, 4.5)) -> plt.Figure:
    """
    Bar chart of average review score per delivery-time group.

    Args:
        scores: Output of ``BusinessMetricsCalculator.review_score_by_delivery``.
        group_col: The grouping column in ``scores``.
        period_label: Analysis period label for the title.
        xlabel: Axis label describing the groups, including units.

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = scores[group_col].astype(str)
    ax.bar(labels, scores['avg_review_score'], color=COLORS['primary'], width=0.65)
    if len(scores) <= 6:
        for x, value in enumerate(scores['avg_review_score']):
            ax.text(x, value, f"{value:.2f}", ha='center', va='bottom', fontsize=9,
                    color=COLORS['text_secondary'])
    ax.set_ylim(0, 5)
    ax.set_title(f"Average Review Score by Delivery Time, {period_label}",
                 fontsize=14, fontweight='bold', loc='left')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Average review score (1-5)')
    _style_axes(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_metrics_summary(report: Dict[str, object], period_label: str,
                          comparison_label: Optional[str] = None) -> None:
    """
    Print the headline metrics of a report as plain text.

    Args:
        report: Output of ``BusinessMetricsCalculator.generate_comprehensive_report``.
        period_label: Analysis period label.
        comparison_label: Comparison period label, if the report has comparison values.
    """
    revenue = report['revenue_metrics']
    print('=' * 60)
    print(f"BUSINESS METRICS SUMMARY: {period_label}")
    if comparison_label:
        print(f"Compared with: {comparison_label}")
    print('=' * 60)
    print(f"Total revenue:          {format_currency(revenue['total_revenue'])}")
    print(f"Total orders:           {revenue['total_orders']:,}")
    print(f"Average order value:    {format_currency(revenue['average_order_value'])}")
    if 'revenue_growth_rate' in revenue:
        print(f"Revenue change:         {format_percentage(revenue['revenue_growth_rate'], signed=True)}")
        print(f"Order count change:     {format_percentage(revenue['order_growth_rate'], signed=True)}")
        print(f"AOV change:             {format_percentage(revenue['aov_growth_rate'], signed=True)}")
    print(f"Average MoM growth:     {format_percentage(report['average_mom_growth'], signed=True)}")
    if 'customer_satisfaction' in report:
        print(f"Average review score:   {report['customer_satisfaction']['avg_review_score']:.2f} / 5")
    if 'delivery_performance' in report:
        print(f"Average delivery time:  {report['delivery_performance']['avg_delivery_days']:.1f} days")
    print('=' * 60)
