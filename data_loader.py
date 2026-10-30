"""
Data loading, cleaning and transformation for the e-commerce analysis.

This module is responsible for everything that happens before a metric is
calculated: reading the raw CSV files, parsing timestamps, joining the tables
into an item-level sales dataset, and filtering that dataset to a configurable
period (an inclusive range of months, which may span several years).

Typical usage:

    from data_loader import load_and_process_data

    loader, processed = load_and_process_data('ecommerce_data')
    sales = loader.create_sales_dataset(start_year=2023, start_month=1,
                                        end_year=2023, end_month=12)
"""

import calendar
import os
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import pandas as pd


# Raw file names, keyed by the name used throughout the analysis.
FILE_MAPPINGS = {
    'orders': 'orders_dataset.csv',
    'order_items': 'order_items_dataset.csv',
    'products': 'products_dataset.csv',
    'customers': 'customers_dataset.csv',
    'reviews': 'order_reviews_dataset.csv',
    'payments': 'order_payments_dataset.csv',
}

ORDER_TIMESTAMP_COLUMNS = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
]

REVIEW_TIMESTAMP_COLUMNS = ['review_creation_date', 'review_answer_timestamp']


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def _validate_period(start_year: int, start_month: int,
                     end_year: int, end_month: int) -> None:
    """Raise ValueError if the month range is malformed or reversed."""
    for month in (start_month, end_month):
        if not 1 <= month <= 12:
            raise ValueError(f"Month must be between 1 and 12, got {month}")
    if (start_year, start_month) > (end_year, end_month):
        raise ValueError(
            f"Period start {start_year}-{start_month:02d} is after "
            f"period end {end_year}-{end_month:02d}"
        )


def filter_by_period(df: pd.DataFrame,
                     start_year: int,
                     start_month: int,
                     end_year: int,
                     end_month: int,
                     date_col: str = 'order_purchase_timestamp') -> pd.DataFrame:
    """
    Keep only rows whose date falls in an inclusive range of whole months.

    The range may span year boundaries, e.g. July 2022 through June 2023.

    Args:
        df: DataFrame with a datetime column named ``date_col``.
        start_year, start_month: First month of the period (inclusive).
        end_year, end_month: Last month of the period (inclusive).
        date_col: Name of the datetime column to filter on.

    Returns:
        pd.DataFrame: A filtered copy of ``df``.
    """
    _validate_period(start_year, start_month, end_year, end_month)
    period = df[date_col].dt.to_period('M')
    start = pd.Period(year=start_year, month=start_month, freq='M')
    end = pd.Period(year=end_year, month=end_month, freq='M')
    return df[(period >= start) & (period <= end)].copy()


def describe_period(start_year: int, start_month: int,
                    end_year: int, end_month: int) -> str:
    """
    Build a human-readable label for a period, for use in titles and captions.

    Args:
        start_year, start_month: First month of the period.
        end_year, end_month: Last month of the period.

    Returns:
        str: For example ``'Jan 2023 - Dec 2023'`` or ``'Mar 2023'``.
    """
    start = f"{calendar.month_abbr[start_month]} {start_year}"
    end = f"{calendar.month_abbr[end_month]} {end_year}"
    return start if start == end else f"{start} - {end}"


def get_comparison_period(start_year: int, start_month: int,
                          end_year: int, end_month: int,
                          years_back: int = 1) -> Tuple[int, int, int, int]:
    """
    Shift a period back by a whole number of years (same months, earlier year).

    Args:
        start_year, start_month, end_year, end_month: The analysis period.
        years_back: How many years to shift back.

    Returns:
        Tuple[int, int, int, int]: (start_year, start_month, end_year, end_month)
        of the comparison period.
    """
    return (start_year - years_back, start_month, end_year - years_back, end_month)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def add_delivery_speed_category(df: pd.DataFrame,
                                bin_edges: Sequence[float],
                                labels: Sequence[str],
                                days_col: str = 'delivery_days',
                                category_col: str = 'delivery_speed_category') -> pd.DataFrame:
    """
    Bucket delivery days into labelled categories.

    The bucket boundaries are a business decision, so they are not defined in
    this module; callers must pass them explicitly. Bins are right-inclusive,
    so ``bin_edges=[0, 3, 7, inf]`` produces (0, 3], (3, 7] and (7, inf).

    Args:
        df: DataFrame with a numeric ``days_col`` column.
        bin_edges: Monotonically increasing bucket edges (len(labels) + 1 values).
        labels: Label for each bucket.
        days_col: Column holding delivery time in days.
        category_col: Name of the column to create.

    Returns:
        pd.DataFrame: A copy of ``df`` with an ordered categorical ``category_col``.
        Rows with missing delivery days get a missing category.
    """
    if len(bin_edges) != len(labels) + 1:
        raise ValueError("bin_edges must contain exactly one more value than labels")
    result = df.copy()
    result[category_col] = pd.cut(result[days_col], bins=list(bin_edges),
                                  labels=list(labels), right=True)
    return result


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class EcommerceDataLoader:
    """
    Load, clean and join the e-commerce CSV files.

    Attributes:
        data_path: Directory containing the CSV files.
        raw_data: Dataset name -> DataFrame exactly as read from disk.
        processed_data: Dataset name -> cleaned DataFrame (parsed timestamps,
            derived date columns).
    """

    def __init__(self, data_path: str = 'ecommerce_data'):
        """
        Args:
            data_path: Directory containing the CSV files.
        """
        self.data_path = data_path
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}

    def load_raw_data(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Read every CSV listed in FILE_MAPPINGS. Missing files are skipped.

        Args:
            verbose: Print the number of records loaded per file.

        Returns:
            Dict[str, pd.DataFrame]: The raw datasets.
        """
        for key, filename in FILE_MAPPINGS.items():
            path = os.path.join(self.data_path, filename)
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping")
                continue
            self.raw_data[key] = pd.read_csv(path)
            if verbose:
                print(f"Loaded {key}: {len(self.raw_data[key]):,} records")
        return self.raw_data

    def clean_orders_data(self) -> pd.DataFrame:
        """
        Parse order timestamps and add purchase year and month columns.

        Returns:
            pd.DataFrame: Orders with datetime columns plus ``purchase_year``,
            ``purchase_month`` and ``purchase_date``.
        """
        orders = self.raw_data['orders'].copy()
        for col in ORDER_TIMESTAMP_COLUMNS:
            if col in orders.columns:
                orders[col] = pd.to_datetime(orders[col])

        orders['purchase_year'] = orders['order_purchase_timestamp'].dt.year
        orders['purchase_month'] = orders['order_purchase_timestamp'].dt.month
        orders['purchase_date'] = orders['order_purchase_timestamp'].dt.date
        return orders

    def clean_order_items_data(self) -> pd.DataFrame:
        """
        Parse the shipping limit date and add the item value including freight.

        Returns:
            pd.DataFrame: Order items with ``total_item_value`` (price + freight).
        """
        order_items = self.raw_data['order_items'].copy()
        if 'shipping_limit_date' in order_items.columns:
            order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date'])
        order_items['total_item_value'] = order_items['price'] + order_items['freight_value']
        return order_items

    def clean_reviews_data(self) -> pd.DataFrame:
        """
        Parse review timestamps.

        Returns:
            pd.DataFrame: Reviews with datetime columns.
        """
        reviews = self.raw_data['reviews'].copy()
        for col in REVIEW_TIMESTAMP_COLUMNS:
            if col in reviews.columns:
                reviews[col] = pd.to_datetime(reviews[col])
        return reviews

    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Clean every loaded dataset. Loads the raw data first if needed.

        Returns:
            Dict[str, pd.DataFrame]: The processed datasets.
        """
        if not self.raw_data:
            self.load_raw_data()

        self.processed_data['orders'] = self.clean_orders_data()
        self.processed_data['order_items'] = self.clean_order_items_data()
        if 'reviews' in self.raw_data:
            self.processed_data['reviews'] = self.clean_reviews_data()
        for key in ('products', 'customers', 'payments'):
            if key in self.raw_data:
                self.processed_data[key] = self.raw_data[key].copy()
        return self.processed_data

    def filter_orders_by_period(self,
                                start_year: int,
                                start_month: int,
                                end_year: int,
                                end_month: int) -> pd.DataFrame:
        """
        Return all orders (every status) purchased in the period.

        Used for order-level analyses such as the order status distribution,
        which must not be restricted to delivered orders.

        Returns:
            pd.DataFrame: Orders purchased in the period.
        """
        return filter_by_period(self.processed_data['orders'],
                                start_year, start_month, end_year, end_month)

    def create_sales_dataset(self,
                             start_year: Optional[int] = None,
                             start_month: Optional[int] = None,
                             end_year: Optional[int] = None,
                             end_month: Optional[int] = None,
                             status_filter: Union[str, Iterable[str], None] = 'delivered',
                             year_filter: Optional[int] = None,
                             month_filter: Optional[int] = None) -> pd.DataFrame:
        """
        Build the item-level sales dataset: one row per order item, enriched with
        order, product, customer and review attributes.

        The period can be given either as an explicit month range
        (``start_year``/``start_month``/``end_year``/``end_month``) or, for
        backward compatibility, as ``year_filter`` with an optional
        ``month_filter``. If neither is given, all dates are kept.

        Args:
            start_year, start_month: First month of the period (inclusive).
                ``start_month`` defaults to 1.
            end_year, end_month: Last month of the period (inclusive).
                ``end_year`` defaults to ``start_year``; ``end_month`` to 12.
            status_filter: Order status (or list of statuses) to keep; None keeps all.
            year_filter: Shortcut for a single calendar year.
            month_filter: With ``year_filter``, restrict to a single month.

        Returns:
            pd.DataFrame: Columns from order items plus ``customer_id``,
            ``order_status``, ``order_purchase_timestamp``,
            ``order_delivered_customer_date``, ``purchase_year``,
            ``purchase_month``, ``product_category_name``, ``customer_state``,
            ``customer_city``, ``review_score`` and ``delivery_days``.
        """
        if not self.processed_data:
            self.process_all_data()

        orders_cols = ['order_id', 'customer_id', 'order_status',
                       'order_purchase_timestamp', 'order_delivered_customer_date',
                       'purchase_year', 'purchase_month']
        sales_data = self.processed_data['order_items'].merge(
            self.processed_data['orders'][orders_cols], on='order_id', how='left')

        if status_filter is not None:
            statuses = [status_filter] if isinstance(status_filter, str) else list(status_filter)
            sales_data = sales_data[sales_data['order_status'].isin(statuses)]

        # Resolve the period from either the range arguments or the legacy filters
        if start_year is None and year_filter is not None:
            start_year = end_year = year_filter
            if month_filter is not None:
                start_month = end_month = month_filter
        if start_year is not None:
            sales_data = filter_by_period(
                sales_data,
                start_year, start_month or 1,
                end_year if end_year is not None else start_year, end_month or 12,
            )

        if 'products' in self.processed_data:
            sales_data = sales_data.merge(
                self.processed_data['products'][['product_id', 'product_category_name']],
                on='product_id', how='left')

        if 'customers' in self.processed_data:
            sales_data = sales_data.merge(
                self.processed_data['customers'][['customer_id', 'customer_state', 'customer_city']],
                on='customer_id', how='left')

        # Left join keeps orders without a review (review_score is then NaN);
        # the reviews file has at most one review per order, so no rows are duplicated.
        if 'reviews' in self.processed_data:
            sales_data = sales_data.merge(
                self.processed_data['reviews'][['order_id', 'review_score']],
                on='order_id', how='left')

        sales_data['delivery_days'] = (
            sales_data['order_delivered_customer_date'] - sales_data['order_purchase_timestamp']
        ).dt.days

        return sales_data.reset_index(drop=True)

    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Summarise the size of each processed dataset.

        Returns:
            Dict[str, Dict]: Dataset name -> rows, columns, memory (MB) and,
            for orders, the purchase date range.
        """
        summary = {}
        for name, df in self.processed_data.items():
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2,
                'date_range': None,
            }
            if name == 'orders':
                summary[name]['date_range'] = {
                    'start': df['order_purchase_timestamp'].min(),
                    'end': df['order_purchase_timestamp'].max(),
                }
        return summary


def load_and_process_data(data_path: str = 'ecommerce_data',
                          verbose: bool = True) -> Tuple[EcommerceDataLoader, Dict[str, pd.DataFrame]]:
    """
    Convenience wrapper: create a loader, read the CSVs and clean them.

    Args:
        data_path: Directory containing the CSV files.
        verbose: Print the number of records loaded per file.

    Returns:
        Tuple[EcommerceDataLoader, Dict[str, pd.DataFrame]]: The loader and its
        processed datasets.
    """
    loader = EcommerceDataLoader(data_path)
    loader.load_raw_data(verbose=verbose)
    processed_data = loader.process_all_data()
    return loader, processed_data
