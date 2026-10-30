# E-commerce Business Analytics Dashboard

A comprehensive business intelligence solution featuring both Jupyter notebook analysis and a professional Streamlit dashboard for e-commerce sales data with configurable time periods and reusable business metrics calculations.

## Overview

This project transforms a basic exploratory data analysis into a professional, maintainable business intelligence framework. The refactored solution provides:

- **Configurable Analysis**: Analyze any inclusive month range (including ranges that cross years) against any comparison period
- **Modular Architecture**: Separate modules for data loading, metric calculation and charts
- **Consistent Visualizations**: Titled charts with labelled axes, units, date ranges and one shared palette
- **Neutral Summary**: Observations generated from the calculated values, with no assumed business thresholds

## Project Structure

```
data_analysis/
├── EDA_Refactored.ipynb     # Main analysis notebook
├── dashboard.py             # Streamlit dashboard application
├── EDA.ipynb                # Original exploratory analysis (kept for reference)
├── data_loader.py           # Data loading, cleaning, joining and period filtering
├── business_metrics.py      # Business metric calculations only (no plotting)
├── visualizations.py        # Charts and text formatting helpers
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── ecommerce_data/         # Data directory
    ├── orders_dataset.csv
    ├── order_items_dataset.csv
    ├── products_dataset.csv
    ├── customers_dataset.csv
    ├── order_reviews_dataset.csv
    └── order_payments_dataset.csv
```

## Features

### 1. Configurable Analysis Framework
- Set the analysis period as a start and end month (inclusive), e.g. Jan-Dec 2023, Q4 2023 or Jul 2022 - Jun 2023
- Comparison period defaults to the same months one year earlier, or can be set explicitly or turned off
- Order status and delivery-time buckets are configuration values, not hardcoded
- A data coverage check shows how many months of data each period actually contains

### 2. Comprehensive Business Metrics
- **Revenue Analysis**: Total revenue, growth rates, average order value
- **Product Performance**: Category analysis, revenue share, top performers
- **Geographic Insights**: State-level revenue and order analysis
- **Customer Satisfaction**: Average review score and share of orders at each score (1-5)
- **Delivery Performance**: Average and median delivery time, review score by delivery time
- **Order Status**: Share of orders in each status

### 3. Visualizations
- Monthly revenue line chart, with the comparison period overlaid
- Revenue by product category (horizontal bar)
- Interactive US state choropleth of revenue
- Review score distribution and review score by delivery time
- Every chart title includes the date range; axes carry units; one blue primary color, grey for the comparison period, and a single-hue blue scale for the map

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (for notebook analysis)

### Installation Steps

1. **Clone or download the project files**

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data files are in place**:
   - Place CSV files in the `ecommerce_data/` directory
   - Verify all required files are present (see Project Structure above)

4. **Run the applications**:

   **For Streamlit Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

   **For Jupyter Notebook Analysis**:
   ```bash
   jupyter notebook EDA_Refactored.ipynb
   ```

## Usage Guide

### Streamlit Dashboard

1. **Launch the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

2. **Choose a date range** with the month-range slider in the top-right of the header. It defaults to Jan 2023 - Dec 2023 and applies to every card and chart on the page. Trends compare the selected months with the same months one year earlier; if those months have no data, cards show "No comparison data" and the dashed previous-period line is omitted.

3. **Read the page** from top to bottom: KPI cards, the 2x2 chart grid, then the customer experience cards (see [Dashboard Features](#dashboard-features)).

4. **Configuration**: the constants at the top of `dashboard.py` set the data path, order status, delivery-time buckets and default date range. They match the notebook defaults.

### Notebook Analysis

1. **Open the refactored notebook**: `EDA_Refactored.ipynb`

2. **Configure analysis parameters** in the configuration cell (Section 2.1):
   ```python
   DATA_PATH = 'ecommerce_data'

   # Analysis period (inclusive month range)
   START_YEAR, START_MONTH = 2023, 1
   END_YEAR, END_MONTH = 2023, 12

   # Comparison period: same months one year earlier, or set explicit bounds
   COMPARE_TO_PREVIOUS_YEAR = True
   COMPARISON_START_YEAR, COMPARISON_START_MONTH = None, None
   COMPARISON_END_YEAR, COMPARISON_END_MONTH = None, None

   # Order status included in sales metrics
   ORDER_STATUS = 'delivered'

   # Delivery-time buckets (right-inclusive edges in days, one label per bucket)
   DELIVERY_BIN_EDGES = [0, 3, 7, np.inf]
   DELIVERY_BIN_LABELS = ['1-3 days', '4-7 days', '8+ days']
   ```

3. **Run all cells** (Kernel > Restart & Run All) to generate the complete analysis. To run it headlessly:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace EDA_Refactored.ipynb
   ```

4. **Notebook sections**: Introduction & Business Objectives, Data Loading & Configuration, Data Dictionary, Data Preparation & Transformation, Business Metrics (revenue, product, geographic, customer experience), and Summary of Observations.

### Advanced Configuration

#### Analyzing Specific Time Periods
```python
# Q4 2023 compared with Q4 2022
START_YEAR, START_MONTH = 2023, 10
END_YEAR, END_MONTH = 2023, 12

# A single month
START_YEAR, START_MONTH = 2023, 3
END_YEAR, END_MONTH = 2023, 3

# A range crossing a year boundary, compared with an explicit period
START_YEAR, START_MONTH = 2023, 7
END_YEAR, END_MONTH = 2024, 6
COMPARE_TO_PREVIOUS_YEAR = False
COMPARISON_START_YEAR, COMPARISON_START_MONTH = 2022, 7
COMPARISON_END_YEAR, COMPARISON_END_MONTH = 2023, 6
```

Check the data coverage output in Section 4: if a period contains fewer months of data than requested, its comparison figures cover different amounts of time.

#### Custom Data Paths
```python
# Use different data location
DATA_PATH = '/path/to/your/data'
```

#### Filtering by Order Status
```python
# In the notebook configuration cell: a single status, a list, or None for all
ORDER_STATUS = 'delivered'
ORDER_STATUS = ['delivered', 'shipped']
```

### Module Usage

#### Data Loading Module (`data_loader.py`)
```python
import numpy as np
from data_loader import (load_and_process_data, describe_period,
                         get_comparison_period, add_delivery_speed_category)

loader, processed_data = load_and_process_data('ecommerce_data')

# Item-level sales for an inclusive month range
period = (2023, 1, 2023, 12)
current = loader.create_sales_dataset(*period, status_filter='delivered')
previous = loader.create_sales_dataset(*get_comparison_period(*period))

# Single-year shortcut (still supported, used by dashboard.py)
current_2023 = loader.create_sales_dataset(year_filter=2023)

# Optional delivery-time buckets; the boundaries are always passed in
current = add_delivery_speed_category(current, [0, 3, 7, np.inf],
                                      ['1-3 days', '4-7 days', '8+ days'])

# All orders in the period regardless of status
orders = loader.filter_orders_by_period(*period)

describe_period(*period)  # 'Jan 2023 - Dec 2023'
```

#### Business Metrics Module (`business_metrics.py`)
```python
from business_metrics import BusinessMetricsCalculator, calculate_order_status_distribution

calc = BusinessMetricsCalculator(current, comparison_data=previous)
report = calc.generate_comprehensive_report()

report['revenue_metrics']        # revenue, orders, AOV, period-over-period changes
report['monthly_trends']         # per-month revenue, orders, AOV and MoM change
report['average_mom_growth']     # mean of monthly revenue growth rates
calc.analyze_product_performance(top_n=5)
calc.review_score_by_delivery('delivery_speed_category')
calculate_order_status_distribution(orders)
```

The calculator never filters by date itself: pass it data that is already filtered to the period you want.

#### Visualization Module (`visualizations.py`)
```python
import visualizations as viz

label = describe_period(*period)
viz.plot_monthly_revenue(report['monthly_trends'], label,
                         report['comparison_monthly_trends'], 'Jan 2022 - Dec 2022')
viz.plot_category_revenue(report['product_performance'], label)
viz.plot_revenue_by_state(report['geographic_performance'], label).show()
viz.print_metrics_summary(report, label)
```

## Key Business Metrics

### Revenue Metrics
- **Total Revenue**: Sum of item prices for orders with the configured status (freight excluded)
- **Revenue Growth Rate**: Percent change from the comparison period; reported as n/a when the comparison value is zero
- **Average Order Value (AOV)**: Mean revenue per order
- **Monthly Growth Trends**: Month-over-month percent change in revenue, and its average over the period

### Product Performance
- **Category Revenue**: Revenue by product category
- **Revenue Share**: Percentage of total revenue by category
- **Items and Orders**: Items sold and distinct orders per category

### Geographic Analysis
- **State Performance**: Revenue and order count by state
- **Revenue Share**: Percentage of total revenue by state
- **Regional AOV**: Average order value by state

### Customer Experience
- **Review Scores**: Average rating (1-5 scale) over reviewed orders
- **Score Distribution**: Percentage of reviewed orders at each score
- **Delivery Performance**: Average and median days from purchase to delivery
- **Review Score by Delivery Time**: Average score per delivery day and per configured bucket
- **Order Status Distribution**: Percentage of all orders in each status

## Output Examples

### Console Output
```
============================================================
BUSINESS METRICS SUMMARY: Jan 2023 - Dec 2023
Compared with: Jan 2022 - Dec 2022
============================================================
Total revenue:          $3,360,294.74
Total orders:           4,635
Average order value:    $724.98
Revenue change:         -2.46%
Order count change:     -2.40%
AOV change:             -0.06%
Average MoM growth:     -0.39%
Average review score:   4.10 / 5
Average delivery time:  8.0 days
============================================================
```

### Generated Visualizations
- Monthly revenue line chart (analysis vs comparison period)
- Revenue by product category horizontal bar chart
- Interactive US state choropleth map
- Review score distribution and review score by delivery time bar charts

## Customization Options

### Adding New Metrics
1. Add a method to `BusinessMetricsCalculator` in `business_metrics.py` (calculations only, with a docstring)
2. Add a chart function to `visualizations.py` that takes a `period_label` for its title
3. Call both from a new subsection of the notebook, with a short markdown explanation

### Custom Visualizations
```python
# Example: custom chart in visualizations.py
def plot_custom_metric(data, period_label):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(data['label'], data['value'], color=COLORS['primary'])
    ax.set_title(f"Custom Metric, {period_label}", loc='left')
    ax.set_ylabel('Value (units)')
    _style_axes(ax)
    return fig
```

### Data Source Modifications
- Modify `data_loader.py` to handle different CSV structures
- Update column mappings in the `EcommerceDataLoader` class
- Add new data validation rules

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
   - Ensure all files are in the same directory
   - Check Python path configuration

2. **Missing Data Files**:
   - Verify CSV files are in the `ecommerce_data/` directory
   - Check file naming matches expected patterns

3. **Empty Results**:
   - Verify date filters match available data
   - Check order status filtering

4. **Visualization Issues**:
   - Ensure all required packages are installed
   - Check Plotly version compatibility for interactive maps

### Performance Optimization
- For large datasets, consider chunked processing
- Use data sampling for initial exploration
- Implement caching for repeated analysis

## Dashboard Features

### Layout Structure
- **Header**: Title on the left; month-range slider on the right (applies globally), with the comparison period shown underneath
- **KPI Row**: 4 cards of equal height
  - Total Revenue, Average Order Value and Total Orders, each with a trend indicator against the comparison period
  - Monthly Growth: the average month-over-month revenue change in the selected range (needs at least 2 months)
- **Charts Grid**: 2x2 layout, all Plotly
  - Monthly revenue trend: solid line for the selected period, dashed line for the same months a year earlier, grid lines, y-axis in $300K format
  - Top 10 product categories by revenue, sorted descending, blue gradient (lighter for lower values), values as $300K / $2M
  - Revenue by state: US choropleth on a blue gradient
  - Average review score by delivery-time bucket (1-3 days, 4-7 days, 8+ days)
- **Bottom Row**: 2 cards of equal height
  - Average delivery time with trend indicator
  - Average review score as a large number with partially filled stars

### Trend Indicators
- An up or down arrow shows the direction of the change, followed by the change with two decimals, e.g. "▼ 2.46% vs Jan 2022 - Dec 2022"
- Green means a favourable change and red an unfavourable one. For revenue, order value and orders, an increase is green. For delivery time, a decrease (faster delivery) is green

### Technical Notes
- All data preparation comes from `data_loader.py` and all metrics from `business_metrics.py`, so the dashboard shows the same numbers as the notebook for the same period
- Delivery time and review score are averaged per order, not per item
- Data is loaded once per server process and each date range is cached
- Partial months at the edges of the data (Dec 2021 and Jan 2024 contain only one day each) can produce extreme month-over-month growth values when included in the range

## Future Enhancements

### Planned Features
- Real-time data connections
- Predictive analytics and forecasting
- Customer segmentation analysis
- A/B testing framework
- Automated report scheduling
- Export functionality (PDF reports)

### Extension Ideas
- Integration with business intelligence tools
- API endpoints for metrics access
- Machine learning model integration
- Advanced statistical analysis
- Mobile-responsive improvements

## Contributing

To extend this analysis framework:

1. Follow the existing code structure and documentation patterns
2. Add comprehensive docstrings to new functions
3. Include unit tests for new business logic
4. Update this README with new features

## License

This project is provided as-is for educational and business analysis purposes.

---

**Note**: This framework is designed to be easily maintained and extended for ongoing business intelligence needs. The modular architecture ensures that updates to data sources or metric calculations can be made without affecting the overall analysis structure.