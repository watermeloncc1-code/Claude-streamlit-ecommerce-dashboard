"""
E-commerce sales dashboard (Streamlit).

Interactive version of EDA_Refactored.ipynb. All data preparation comes from
data_loader.py and all metrics from business_metrics.py, so the numbers shown
here match the notebook for the same period.

Run with:
    streamlit run dashboard.py
"""

import calendar
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from business_metrics import BusinessMetricsCalculator
from data_loader import (
    add_delivery_speed_category,
    describe_period,
    get_comparison_period,
    load_and_process_data,
)
from visualizations import COLORS, SEQUENTIAL_BLUES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = 'ecommerce_data'
ORDER_STATUS = 'delivered'

# Delivery-time buckets (right-inclusive edges in days), same as the notebook
DELIVERY_BIN_EDGES = [0, 3, 7, np.inf]
DELIVERY_BIN_LABELS = ['1-3 days', '4-7 days', '8+ days']

DEFAULT_YEAR = 2023
DEFAULT_START_MONTH = 1
DEFAULT_END_MONTH = 12

TREND_POSITIVE = '#1a7f37'
TREND_NEGATIVE = '#cf222e'
TREND_NEUTRAL = '#6e6d68'
STAR_FILLED = '#e3a008'
STAR_EMPTY = '#d6d4cf'

CHART_HEIGHT = 380

st.set_page_config(page_title='E-commerce Sales Dashboard', layout='wide',
                   initial_sidebar_state='collapsed')

st.markdown(f"""
<style>
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 3.5rem; padding-bottom: 2rem; }}

    .card {{
        background: #ffffff;
        border: 1px solid #e4e3df;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        padding: 1rem 1.25rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }}
    .kpi-card {{ height: 132px; }}
    .bottom-card {{ height: 170px; align-items: center; text-align: center; }}

    .card-label {{ font-size: 0.9rem; color: #52514e; margin: 0 0 0.35rem 0; }}
    .card-value {{ font-size: 1.9rem; font-weight: 700; color: #0b0b0b; margin: 0; line-height: 1.2; }}
    .card .card-value-large {{ font-size: 2.6rem; }}
    .card-trend {{ font-size: 0.85rem; margin: 0.4rem 0 0 0; color: {TREND_NEUTRAL}; }}
    .trend-positive {{ color: {TREND_POSITIVE}; font-weight: 600; }}
    .trend-negative {{ color: {TREND_NEGATIVE}; font-weight: 600; }}

    .stars {{ position: relative; display: inline-block; font-size: 1.6rem;
              letter-spacing: 2px; color: {STAR_EMPTY}; line-height: 1.2; }}
    .stars-fill {{ position: absolute; top: 0; left: 0; overflow: hidden;
                   white-space: nowrap; color: {STAR_FILLED}; }}

    .action-banner {{
        background: #fff8e6;
        border: 1px solid #f0d58a;
        border-left: 5px solid {STAR_FILLED};
        border-radius: 10px;
        padding: 0.9rem 1.25rem;
        margin: 0.5rem 0 1rem 0;
    }}
    .action-headline {{ font-size: 1.2rem; font-weight: 700; color: #0b0b0b; }}
    .action-detail {{ font-size: 0.95rem; color: #3d3c39; margin-top: 0.25rem; }}

    .dashboard-subtitle {{ color: #52514e; margin-top: -0.5rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_resource
def get_loader():
    """Load and clean the CSV files once per server process."""
    loader, _ = load_and_process_data(DATA_PATH, verbose=False)
    return loader


def available_years(loader) -> List[int]:
    """Years that contain at least one order, most recent first."""
    years = loader.processed_data['orders']['purchase_year'].unique()
    return sorted((int(y) for y in years), reverse=True)


def month_label(year: int, month: int) -> str:
    """Label such as 'Jan 2023'."""
    return f"{calendar.month_abbr[month]} {year}"


@st.cache_data
def build_period_data(start: Tuple[int, int], end: Tuple[int, int]) -> Dict[str, object]:
    """
    Build the datasets and metrics for one analysis period and its comparison
    period (the same months one year earlier).

    Returns:
        Dict with ``period``, ``comparison_period``, ``current_sales``,
        ``report``, ``score_by_bucket`` and ``comparison_delivery``.
    """
    loader = get_loader()
    period = (*start, *end)
    comparison_period = get_comparison_period(*period)

    current = loader.create_sales_dataset(*period, status_filter=ORDER_STATUS)
    current = add_delivery_speed_category(current, DELIVERY_BIN_EDGES, DELIVERY_BIN_LABELS)
    comparison = loader.create_sales_dataset(*comparison_period, status_filter=ORDER_STATUS)

    # Only compare when the earlier period has data in as many months as the
    # current one; otherwise trends would compare unequal amounts of time.
    def months_with_data(sales):
        return sales[['purchase_year', 'purchase_month']].drop_duplicates().shape[0]

    comparison_status = 'ok'
    if comparison.empty:
        comparison, comparison_status = None, 'missing'
    elif months_with_data(comparison) < months_with_data(current):
        comparison, comparison_status = None, 'incomplete'

    result = {
        'period': period,
        'comparison_period': comparison_period if comparison is not None else None,
        'comparison_status': comparison_status,
        'requested_comparison_period': comparison_period,
        'current_sales': current,
        'report': None,
        'score_by_bucket': None,
        'comparison_delivery': None,
    }
    if current.empty:
        return result

    calculator = BusinessMetricsCalculator(current, comparison)
    result['report'] = calculator.generate_comprehensive_report()
    result['score_by_bucket'] = calculator.review_score_by_delivery('delivery_speed_category')
    if comparison is not None:
        result['comparison_delivery'] = (
            BusinessMetricsCalculator(comparison).analyze_delivery_performance()['avg_delivery_days']
        )
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _trim(value: float, decimals: int) -> str:
    """Format with up to ``decimals`` decimals, dropping trailing zeros."""
    text = f"{value:.{decimals}f}"
    return text.rstrip('0').rstrip('.') if '.' in text else text


def format_compact_currency(value: float) -> str:
    """Format dollars compactly: $950, $300K, $2M, $3.36M."""
    sign = '-' if value < 0 else ''
    value = abs(value)
    if value >= 1e6:
        return f"{sign}${_trim(value / 1e6, 2)}M"
    if value >= 1e3:
        return f"{sign}${_trim(value / 1e3, 1)}K"
    return f"{sign}${value:,.0f}"


def compact_currency_ticks(max_value: float, n: int = 5) -> Tuple[List[float], List[str]]:
    """
    Round tick positions from 0 to just above ``max_value`` with compact labels.

    Plotly's SI format prints a lowercase 'k', so currency axes use explicit
    tick values and labels instead.
    """
    if max_value <= 0:
        return [0], ['$0']
    raw_step = max_value / n
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw_step)
    ticks = list(np.arange(0, max_value + step, step))
    return ticks, [format_compact_currency(t) for t in ticks]


def trend_html(change_pct: Optional[float], comparison_label: Optional[str],
               higher_is_better: bool = True) -> str:
    """
    Trend line for a card: arrow for direction, color for whether the change
    is favourable, two decimals.
    """
    if change_pct is None or pd.isna(change_pct) or comparison_label is None:
        return 'No comparison data'
    if change_pct == 0:
        return f"0.00% vs {comparison_label}"
    arrow = '&#9650;' if change_pct > 0 else '&#9660;'
    favourable = (change_pct > 0) == higher_is_better
    css_class = 'trend-positive' if favourable else 'trend-negative'
    return (f'<span class="{css_class}">{arrow} {abs(change_pct):.2f}%</span> '
            f'vs {comparison_label}')


def card(label: str, value: str, trend: str, css_class: str = 'kpi-card',
         value_class: str = '') -> str:
    """HTML for a metric card."""
    return f"""
    <div class="card {css_class}">
        <div class="card-label">{label}</div>
        <div class="card-value {value_class}">{value}</div>
        <div class="card-trend">{trend}</div>
    </div>
    """


def priority_action(report: Dict[str, object],
                    score_by_bucket: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    The one action to act on: speed up the slowest deliveries.

    Compares the average review score of the slowest delivery bucket with the
    fastest one, and how many orders fall in the slowest bucket. Returns a
    (headline, detail) pair, or None when either bucket has no reviewed orders.
    """
    fast_label, slow_label = DELIVERY_BIN_LABELS[0], DELIVERY_BIN_LABELS[-1]
    scores = score_by_bucket.assign(
        bucket=score_by_bucket['delivery_speed_category'].astype(str)
    ).set_index('bucket')['avg_review_score']
    if fast_label not in scores.index or slow_label not in scores.index:
        return None

    gap = scores[fast_label] - scores[slow_label]
    slow_share = report['delivery_performance'].get(
        'category_distribution', pd.Series(dtype=float)).get(slow_label, np.nan)
    if pd.isna(gap) or pd.isna(slow_share) or gap <= 0:
        return None

    headline = f"Priority: speed up the {slow_share:.0f}% of orders that take {slow_label}"
    detail = (f"Customers who wait {slow_label} give an average of {scores[slow_label]:.2f} stars, "
              f"compared with {scores[fast_label]:.2f} for {fast_label} "
              f"({gap:.2f} lower). Start with the carriers or regions behind the slowest orders.")
    return headline, detail


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    """Shared chart styling: white background, light grid, compact margins."""
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16, color=COLORS['text'])),
        height=CHART_HEIGHT,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=20, t=60, b=10),
        font=dict(color=COLORS['text_secondary']),
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['grid'], linecolor=COLORS['grid'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['grid'], linecolor=COLORS['grid'])
    return fig


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def revenue_trend_chart(report: Dict[str, object], period_label: str,
                        comparison_label: Optional[str]) -> go.Figure:
    """Monthly revenue: current period solid, previous period dashed."""
    monthly = report['monthly_trends']
    labels = [month_label(y, m) for y, m in zip(monthly['year'], monthly['month'])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=monthly['revenue'], name=period_label, mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2.5), marker=dict(size=8),
        hovertemplate='%{x}: %{customdata}<extra></extra>',
        customdata=[format_compact_currency(v) for v in monthly['revenue']],
    ))

    max_revenue = monthly['revenue'].max()
    previous = report.get('comparison_monthly_trends')
    if previous is not None and not previous.empty:
        # Align each previous-period month with the same month one year later
        previous = previous.assign(label=[month_label(y + 1, m)
                                          for y, m in zip(previous['year'], previous['month'])])
        previous = previous[previous['label'].isin(labels)]
        fig.add_trace(go.Scatter(
            x=previous['label'], y=previous['revenue'], name=comparison_label,
            mode='lines+markers', line=dict(color=COLORS['comparison'], width=2, dash='dash'),
            marker=dict(size=6),
            customdata=[[month_label(y, m), format_compact_currency(v)]
                        for y, m, v in zip(previous['year'], previous['month'], previous['revenue'])],
            hovertemplate='%{customdata[0]}: %{customdata[1]}<extra></extra>',
        ))
        max_revenue = max(max_revenue, previous['revenue'].max())

    tickvals, ticktext = compact_currency_ticks(max_revenue)
    _base_layout(fig, f"Monthly Revenue, {period_label}")
    fig.update_layout(hovermode='x unified',
                      legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='left', x=0),
                      margin=dict(b=40))
    # Ticks show the month only (the year is in the title); hover keeps the full label.
    # Month labels are self-explanatory; the legend sits where an axis title would.
    fig.update_xaxes(title=None, categoryorder='array', categoryarray=labels,
                     tickvals=labels, ticktext=[calendar.month_abbr[m] for m in monthly['month']],
                     tickangle=0)
    fig.update_yaxes(title='Revenue (USD)', tickvals=tickvals, ticktext=ticktext,
                     range=[0, tickvals[-1]])
    return fig


def category_chart(report: Dict[str, object], period_label: str) -> go.Figure:
    """Top 10 product categories by revenue, largest first, blue gradient by value."""
    top = report['product_performance'].head(10)
    tickvals, ticktext = compact_currency_ticks(top['total_revenue'].max() * 1.15)

    fig = go.Figure(go.Bar(
        x=top['total_revenue'], y=top['product_category_name'], orientation='h',
        marker=dict(color=top['total_revenue'], colorscale=SEQUENTIAL_BLUES,
                    cmin=0, cmax=top['total_revenue'].max()),
        text=[format_compact_currency(v) for v in top['total_revenue']],
        textposition='outside', cliponaxis=False,
        customdata=top['revenue_share'],
        hovertemplate='%{y}<br>Revenue: %{text}<br>Share: %{customdata:.1f}%<extra></extra>',
    ))
    _base_layout(fig, f"Top 10 Categories by Revenue, {period_label}")
    fig.update_xaxes(title='Revenue (USD)', tickvals=tickvals, ticktext=ticktext,
                     range=[0, tickvals[-1]])
    fig.update_yaxes(title=None, autorange='reversed', showgrid=False)
    return fig


def state_map(report: Dict[str, object], period_label: str) -> go.Figure:
    """US choropleth of revenue by customer state, blue gradient."""
    states = report['geographic_performance']
    tickvals, ticktext = compact_currency_ticks(states['revenue'].max())

    fig = go.Figure(go.Choropleth(
        locations=states['state'], z=states['revenue'], locationmode='USA-states',
        colorscale=SEQUENTIAL_BLUES, zmin=0, marker_line_color='white',
        colorbar=dict(title=dict(text='Revenue'), tickvals=tickvals, ticktext=ticktext),
        customdata=np.column_stack([
            [format_compact_currency(v) for v in states['revenue']],
            states['orders'],
            [f"${v:,.2f}" for v in states['avg_order_value']],
        ]),
        hovertemplate=('%{location}<br>Revenue: %{customdata[0]}<br>Orders: %{customdata[1]}'
                       '<br>AOV: %{customdata[2]}<extra></extra>'),
    ))
    _base_layout(fig, f"Revenue by State, {period_label}")
    fig.update_layout(geo=dict(scope='usa', bgcolor='white', lakecolor='white'))
    return fig


def satisfaction_chart(score_by_bucket: pd.DataFrame, period_label: str) -> go.Figure:
    """Average review score per delivery-time bucket."""
    fig = go.Figure(go.Bar(
        x=score_by_bucket['delivery_speed_category'].astype(str),
        y=score_by_bucket['avg_review_score'],
        marker=dict(color=COLORS['primary']),
        text=[f"{v:.2f}" for v in score_by_bucket['avg_review_score']],
        textposition='outside', cliponaxis=False,
        customdata=score_by_bucket['orders'],
        hovertemplate='%{x}<br>Average score: %{y:.2f}<br>Orders: %{customdata:,}<extra></extra>',
    ))
    _base_layout(fig, f"Review Score by Delivery Time, {period_label}")
    fig.update_xaxes(title='Delivery time', showgrid=False)
    fig.update_yaxes(title='Average review score (1-5)', range=[0, 5.4], tickvals=[0, 1, 2, 3, 4, 5])
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    loader = get_loader()
    years = available_years(loader)
    month_names = list(calendar.month_abbr)[1:]

    # Header: title left, date range filters (year, from month, to month) right
    title_col, filter_col = st.columns([3, 2], vertical_alignment='bottom')
    with filter_col:
        year_col, from_col, to_col = st.columns(3)
        year = year_col.selectbox(
            'Year', years,
            index=years.index(DEFAULT_YEAR) if DEFAULT_YEAR in years else 0, key='year')
        start_month = from_col.selectbox(
            'From month', range(1, 13), index=DEFAULT_START_MONTH - 1,
            format_func=lambda m: month_names[m - 1], key='start_month')
        # Only months on or after the start month, so the range is always valid
        end_options = list(range(start_month, 13))
        end_default = max(DEFAULT_END_MONTH, start_month)
        end_month = to_col.selectbox(
            'To month', end_options, index=end_options.index(end_default),
            format_func=lambda m: month_names[m - 1], key='end_month')
    start, end = (year, start_month), (year, end_month)

    data = build_period_data(start, end)
    period_label = describe_period(*data['period'])
    comparison_label = describe_period(*data['comparison_period']) if data['comparison_period'] else None

    with title_col:
        st.title('E-commerce Sales Dashboard')
        st.markdown(f'<p class="dashboard-subtitle">Delivered orders, {period_label}</p>',
                    unsafe_allow_html=True)
    requested_label = describe_period(*data['requested_comparison_period'])
    with filter_col:
        if data['comparison_status'] == 'ok':
            st.caption(f"Compared with {comparison_label}")
        elif data['comparison_status'] == 'incomplete':
            st.caption(f"No comparison: {requested_label} has data for fewer months")
        else:
            st.caption(f"No comparison: no data for {requested_label}")

    report = data['report']
    if report is None:
        st.warning(f"No {ORDER_STATUS} orders in {period_label}. Choose a different date range.")
        st.stop()

    # The one thing to act on, before the numbers
    action = priority_action(report, data['score_by_bucket'])
    if action:
        st.markdown(f"""
        <div class="action-banner">
            <div class="action-headline">{action[0]}</div>
            <div class="action-detail">{action[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI row
    revenue = report['revenue_metrics']
    mom = report['average_mom_growth']
    kpi_cols = st.columns(4)
    kpi_cards = [
        card('Total Revenue', format_compact_currency(revenue['total_revenue']),
             trend_html(revenue.get('revenue_growth_rate'), comparison_label)),
        card('Monthly Growth', 'n/a' if pd.isna(mom) else f"{mom:.2f}%",
             'Needs 2+ months' if pd.isna(mom) else 'Average month-over-month change'),
        card('Average Order Value', f"${revenue['average_order_value']:,.2f}",
             trend_html(revenue.get('aov_growth_rate'), comparison_label)),
        card('Total Orders', f"{revenue['total_orders']:,}",
             trend_html(revenue.get('order_growth_rate'), comparison_label)),
    ]
    for col, html in zip(kpi_cols, kpi_cards):
        col.markdown(html, unsafe_allow_html=True)

    st.write('')

    # Charts grid (2x2)
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)
    top_left.plotly_chart(revenue_trend_chart(report, period_label, comparison_label), width='stretch')
    top_right.plotly_chart(category_chart(report, period_label), width='stretch')
    bottom_left.plotly_chart(state_map(report, period_label), width='stretch')
    bottom_right.plotly_chart(satisfaction_chart(data['score_by_bucket'], period_label), width='stretch')

    # Bottom row
    delivery_days = report['delivery_performance']['avg_delivery_days']
    previous_delivery = data['comparison_delivery']
    delivery_change = (
        (delivery_days - previous_delivery) / previous_delivery * 100
        if previous_delivery else None
    )
    review_score = report['customer_satisfaction']['avg_review_score']
    fill_pct = 0 if pd.isna(review_score) else review_score / 5 * 100

    delivery_col, review_col = st.columns(2)
    delivery_col.markdown(card(
        'Average Delivery Time', f"{delivery_days:.2f} days",
        trend_html(delivery_change, comparison_label, higher_is_better=False),
        css_class='bottom-card', value_class='card-value-large',
    ), unsafe_allow_html=True)
    review_col.markdown(f"""
    <div class="card bottom-card">
        <div class="card-value card-value-large">{'n/a' if pd.isna(review_score) else f'{review_score:.2f}'}</div>
        <div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;<div class="stars-fill"
             style="width: {fill_pct:.1f}%">&#9733;&#9733;&#9733;&#9733;&#9733;</div></div>
        <div class="card-trend">Average Review Score</div>
    </div>
    """, unsafe_allow_html=True)


main()
