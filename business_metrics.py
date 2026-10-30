"""
Business metric calculations for the e-commerce analysis.

This module only calculates numbers; it does not filter by date, plot or
print. Inputs are item-level sales datasets produced by
``data_loader.EcommerceDataLoader.create_sales_dataset`` that have already been
filtered to the analysis period (and, optionally, a comparison period).

Definitions used throughout:
    Revenue: sum of item ``price`` (freight excluded).
    Order: a unique ``order_id``.
    Average order value (AOV): mean of per-order revenue.
    Percent change: (current - previous) / previous * 100; undefined (NaN)
        when the previous value is zero or missing.

Typical usage:

    from business_metrics import BusinessMetricsCalculator

    calc = BusinessMetricsCalculator(current_sales, comparison_sales)
    report = calc.generate_comprehensive_report()
"""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _percent_change(current: float, previous: float) -> float:
    """
    Percent change from ``previous`` to ``current``.

    Returns:
        float: Change in percent, or NaN if ``previous`` is zero or missing.
    """
    if previous is None or pd.isna(previous) or previous == 0:
        return np.nan
    return (current - previous) / previous * 100


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise ValueError listing any of ``columns`` missing from ``df``."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _revenue_summary(sales: pd.DataFrame) -> Dict[str, float]:
    """Revenue, order count, AOV and items sold for one sales dataset."""
    order_revenue = sales.groupby('order_id')['price'].sum()
    return {
        'total_revenue': sales['price'].sum(),
        'total_orders': sales['order_id'].nunique(),
        'average_order_value': order_revenue.mean() if len(order_revenue) else np.nan,
        'total_items_sold': len(sales),
    }


def _order_level(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per order (order attributes are repeated on every item row)."""
    return sales.drop_duplicates('order_id')


class BusinessMetricsCalculator:
    """
    Calculate revenue, product, geographic and customer experience metrics.

    Both datasets must already be filtered to their respective periods; this
    class never filters by date itself, so it works for any period length.
    """

    def __init__(self, current_data: pd.DataFrame,
                 comparison_data: Optional[pd.DataFrame] = None):
        """
        Args:
            current_data: Item-level sales for the analysis period. Requires
                ``order_id`` and ``price``; individual methods need more columns.
            comparison_data: Item-level sales for the comparison period, or None.
        """
        _require_columns(current_data, ['order_id', 'price'])
        if comparison_data is not None:
            _require_columns(comparison_data, ['order_id', 'price'])
        self.current_data = current_data.copy()
        self.comparison_data = comparison_data.copy() if comparison_data is not None else None

    # -- Revenue -----------------------------------------------------------

    def calculate_revenue_metrics(self) -> Dict[str, float]:
        """
        Headline revenue metrics for the analysis period.

        Returns:
            Dict[str, float]: ``total_revenue``, ``total_orders``,
            ``average_order_value`` and ``total_items_sold``. When comparison
            data is available, also ``previous_*`` values and
            ``revenue_growth_rate``, ``order_growth_rate`` and
            ``aov_growth_rate`` (percent).
        """
        metrics = _revenue_summary(self.current_data)
        if self.comparison_data is not None:
            previous = _revenue_summary(self.comparison_data)
            metrics.update({
                'previous_revenue': previous['total_revenue'],
                'previous_orders': previous['total_orders'],
                'previous_aov': previous['average_order_value'],
                'previous_items_sold': previous['total_items_sold'],
                'revenue_growth_rate': _percent_change(metrics['total_revenue'], previous['total_revenue']),
                'order_growth_rate': _percent_change(metrics['total_orders'], previous['total_orders']),
                'aov_growth_rate': _percent_change(metrics['average_order_value'], previous['average_order_value']),
            })
        return metrics

    @staticmethod
    def _monthly_table(sales: pd.DataFrame) -> pd.DataFrame:
        """Revenue, orders and AOV per (year, month), with month-over-month change."""
        _require_columns(sales, ['purchase_year', 'purchase_month'])
        keys = ['purchase_year', 'purchase_month']
        per_order = sales.groupby(keys + ['order_id'])['price'].sum().reset_index()
        monthly = per_order.groupby(keys).agg(
            revenue=('price', 'sum'),
            orders=('order_id', 'nunique'),
            avg_order_value=('price', 'mean'),
        ).reset_index().rename(columns={'purchase_year': 'year', 'purchase_month': 'month'})

        monthly['revenue_growth'] = monthly['revenue'].pct_change() * 100
        monthly['order_growth'] = monthly['orders'].pct_change() * 100
        monthly['aov_growth'] = monthly['avg_order_value'].pct_change() * 100
        return monthly

    def calculate_monthly_trends(self) -> pd.DataFrame:
        """
        Month-by-month metrics for the analysis period.

        Months are grouped by (year, month), so periods that cross a year
        boundary stay in chronological order. Growth columns are the
        month-over-month percent change; the first month is NaN.

        Returns:
            pd.DataFrame: Columns ``year``, ``month``, ``revenue``, ``orders``,
            ``avg_order_value``, ``revenue_growth``, ``order_growth``,
            ``aov_growth``.
        """
        return self._monthly_table(self.current_data)

    def calculate_comparison_monthly_trends(self) -> Optional[pd.DataFrame]:
        """
        Month-by-month metrics for the comparison period.

        Returns:
            pd.DataFrame or None: Same columns as ``calculate_monthly_trends``,
            or None if no comparison data was given.
        """
        if self.comparison_data is None:
            return None
        return self._monthly_table(self.comparison_data)

    def calculate_average_mom_growth(self) -> float:
        """
        Mean of the month-over-month revenue growth rates in the analysis period.

        Returns:
            float: Average monthly revenue growth in percent (NaN if fewer than
            two months of data).
        """
        return self.calculate_monthly_trends()['revenue_growth'].mean()

    # -- Products ----------------------------------------------------------

    def analyze_product_performance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Revenue by product category.

        Args:
            top_n: Return only the ``top_n`` categories by revenue; None returns all.

        Returns:
            pd.DataFrame: Columns ``product_category_name``, ``total_revenue``,
            ``avg_item_price``, ``items_sold``, ``unique_orders`` and
            ``revenue_share`` (percent of total revenue), sorted by revenue.
        """
        _require_columns(self.current_data, ['product_category_name'])
        categories = self.current_data.groupby('product_category_name').agg(
            total_revenue=('price', 'sum'),
            avg_item_price=('price', 'mean'),
            items_sold=('price', 'count'),
            unique_orders=('order_id', 'nunique'),
        ).reset_index()
        categories['revenue_share'] = categories['total_revenue'] / categories['total_revenue'].sum() * 100
        categories = categories.sort_values('total_revenue', ascending=False).reset_index(drop=True)
        return categories.head(top_n) if top_n is not None else categories

    # -- Geography ---------------------------------------------------------

    def analyze_geographic_performance(self) -> pd.DataFrame:
        """
        Revenue, orders and AOV by customer state.

        Returns:
            pd.DataFrame: Columns ``state``, ``revenue``, ``orders``,
            ``avg_order_value`` and ``revenue_share`` (percent), sorted by revenue.
        """
        _require_columns(self.current_data, ['customer_state'])
        per_order = self.current_data.groupby(['customer_state', 'order_id'])['price'].sum().reset_index()
        states = per_order.groupby('customer_state').agg(
            revenue=('price', 'sum'),
            orders=('order_id', 'nunique'),
            avg_order_value=('price', 'mean'),
        ).reset_index().rename(columns={'customer_state': 'state'})
        states['revenue_share'] = states['revenue'] / states['revenue'].sum() * 100
        return states.sort_values('revenue', ascending=False).reset_index(drop=True)

    # -- Customer experience -----------------------------------------------

    def analyze_customer_satisfaction(self) -> Dict[str, object]:
        """
        Review score metrics at the order level (orders without a review are excluded).

        Returns:
            Dict[str, object]: ``avg_review_score``, ``total_reviews``,
            ``reviewed_order_share`` (percent of orders with a review) and
            ``score_distribution`` (pd.Series: score -> percent of reviewed orders).
        """
        _require_columns(self.current_data, ['review_score'])
        orders = _order_level(self.current_data)
        scores = orders['review_score'].dropna()
        distribution = scores.value_counts(normalize=True).sort_index() * 100
        return {
            'avg_review_score': scores.mean(),
            'total_reviews': len(scores),
            'reviewed_order_share': len(scores) / len(orders) * 100 if len(orders) else np.nan,
            'score_distribution': distribution,
        }

    def analyze_delivery_performance(self,
                                     category_col: str = 'delivery_speed_category') -> Dict[str, object]:
        """
        Delivery time metrics at the order level (undelivered orders are excluded).

        Args:
            category_col: Optional delivery bucket column created by
                ``data_loader.add_delivery_speed_category``.

        Returns:
            Dict[str, object]: ``avg_delivery_days``, ``median_delivery_days``,
            ``delivered_orders`` and, if ``category_col`` exists,
            ``category_distribution`` (pd.Series: bucket -> percent of orders).
        """
        _require_columns(self.current_data, ['delivery_days'])
        orders = _order_level(self.current_data).dropna(subset=['delivery_days'])
        metrics = {
            'avg_delivery_days': orders['delivery_days'].mean(),
            'median_delivery_days': orders['delivery_days'].median(),
            'delivered_orders': len(orders),
        }
        if category_col in orders.columns:
            metrics['category_distribution'] = (
                orders[category_col].value_counts(normalize=True, sort=False) * 100
            )
        return metrics

    def review_score_by_delivery(self, group_col: str = 'delivery_days') -> pd.DataFrame:
        """
        Average review score grouped by delivery time.

        Only orders that were delivered and reviewed are included.

        Args:
            group_col: ``'delivery_days'`` for one row per day, or a bucket
                column such as ``'delivery_speed_category'``.

        Returns:
            pd.DataFrame: Columns ``group_col``, ``avg_review_score`` and ``orders``.
        """
        _require_columns(self.current_data, [group_col, 'review_score', 'delivery_days'])
        orders = _order_level(self.current_data).dropna(subset=['review_score', 'delivery_days'])
        return orders.groupby(group_col, observed=True).agg(
            avg_review_score=('review_score', 'mean'),
            orders=('order_id', 'nunique'),
        ).reset_index()

    # -- Report ------------------------------------------------------------

    def generate_comprehensive_report(self) -> Dict[str, object]:
        """
        Run every metric that the available columns allow.

        Returns:
            Dict[str, object]: Keys ``revenue_metrics``, ``monthly_trends``,
            ``comparison_monthly_trends``, ``average_mom_growth``, and, when the
            needed columns exist, ``product_performance``,
            ``geographic_performance``, ``customer_satisfaction`` and
            ``delivery_performance``.
        """
        columns = self.current_data.columns
        report = {
            'revenue_metrics': self.calculate_revenue_metrics(),
            'monthly_trends': self.calculate_monthly_trends(),
            'comparison_monthly_trends': self.calculate_comparison_monthly_trends(),
            'average_mom_growth': self.calculate_average_mom_growth(),
        }
        if 'product_category_name' in columns:
            report['product_performance'] = self.analyze_product_performance()
        if 'customer_state' in columns:
            report['geographic_performance'] = self.analyze_geographic_performance()
        if 'review_score' in columns:
            report['customer_satisfaction'] = self.analyze_customer_satisfaction()
        if 'delivery_days' in columns:
            report['delivery_performance'] = self.analyze_delivery_performance()
        return report


def calculate_order_status_distribution(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Share of orders in each status.

    Args:
        orders: Order-level data (one row per order, every status), already
            filtered to the period of interest. Requires ``order_status``.

    Returns:
        pd.DataFrame: Columns ``order_status``, ``orders`` and ``share``
        (percent of all orders), sorted by count.
    """
    _require_columns(orders, ['order_status'])
    counts = orders['order_status'].value_counts()
    return pd.DataFrame({
        'order_status': counts.index,
        'orders': counts.values,
        'share': counts.values / counts.sum() * 100,
    })
