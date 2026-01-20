"""
reg_gender_styles_analysis.py

Analysis script to generate Women and Boys styles analysis from EBO sales data.

Usage (example):
python reg_gender_styles_analysis.py \
  --sales "D:\\DATA TILL DATE\\Desktop\\EBO FOLDER\\EBO SALES FOLDER\\EBO SALES DATA.xlsx" \
  --sku "D:\\DATA TILL DATE\\Desktop\\EBO FOLDER\\MASTERS\\SKU MASTER.xlsx" \
  --style "D:\\DATA TILL DATE\\Desktop\\EBO FOLDER\\MASTERS\\STYLE MASTER.xlsx" \
  --output "gender_style_analysis.xlsx"

This script:
- Loads sales, sku master, and style master files
- Filters sales from 2025-04-01 to today
- Merges datasets and computes metrics per STYLE and per gender (Women, Boys)
- Flags potential reasons for poor performance (high discount, high void/return rate, low velocity, size concentration)
- Writes results to an Excel workbook with multiple sheets

Note: The script relies on pandas/numpy/openpyxl which are listed in the repository requirements.
"""

from __future__ import annotations
import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default date range start (inclusive)
DEFAULT_START_DATE = pd.Timestamp("2025-04-01")


def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return np.nan


def load_excel_any(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load an Excel file (first sheet by default) into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # If no sheet_name provided, read the first sheet (sheet_name=0).
    sheet_to_use = sheet_name if sheet_name is not None else 0
    df = pd.read_excel(path, sheet_name=sheet_to_use)
    # pandas may return a dict when sheet_name=None or other cases; handle gracefully
    if isinstance(df, dict):
        # pick the first sheet
        first_key = list(df.keys())[0]
        df = df[first_key]
    # Log rows/shape
    try:
        rows = len(df)
    except Exception:
        rows = 0
    logger.info("Loaded %s rows from %s", rows, path)
    return df


def prepare_data(sales_df: pd.DataFrame, sku_df: pd.DataFrame, style_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and merge sales, sku and style masters into a single DataFrame with parsed dates and normalized columns."""
    df = sales_df.copy()

    # Early filter for Women/Boys gender if possible
    if 'GENDER' in style_df.columns:
        style_df = style_df[style_df['GENDER'].str.strip().str.lower().isin(['women', 'boys'])].copy()
        logger.info("Filtered style master to Women/Boys only: kept %d styles", len(style_df))

    # Normalize column names to uppercase trimmed
    df.columns = [c.strip() for c in df.columns]
    sku_df.columns = [c.strip() for c in sku_df.columns]
    style_df.columns = [c.strip() for c in style_df.columns]

    # Parse bill date (try common names)
    date_cols = [c for c in df.columns if "DATE" in c.upper()]
    if not date_cols:
        raise ValueError("No date column found in sales data")
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Filter timeframe
    mask = df[date_col].notna()
    df = df.loc[mask]

    # Ensure numeric columns
    for col in ["BILL_QUANTITY", "NET_AMOUNT", "RSP", "ITEM_DISCOUNT_AMOUNT", "PROMO_AMOUNT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Merge SKU master if possible on SKU or ICODE
    left_on = None
    if "SKU" in df.columns and "SKU" in sku_df.columns:
        left_on = ("SKU", "SKU")
    elif "ICODE" in df.columns and "SKU" in sku_df.columns:
        left_on = ("ICODE", "SKU")

    if left_on:
        df = df.merge(sku_df, left_on=left_on[0], right_on=left_on[1], how="left", suffixes=("", "_sku"))
        logger.info("Merged SKU master; missing SKU style count=%d", df['STYLE'].isna().sum() if 'STYLE' in df.columns else 0)
    else:
        logger.info("Could not auto-detect SKU join column; skipping SKU merge")

    # Merge style master on STYLE
    if 'STYLE' in df.columns and 'STYLE' in style_df.columns:
        df = df.merge(style_df, on='STYLE', how='left', suffixes=("", "_style"))
    else:
        logger.info("STYLE column not present in both datasets; style master merge skipped")

    return df


def filter_date_range(df: pd.DataFrame, start_date: pd.Timestamp, date_col_name: Optional[str] = None) -> pd.DataFrame:
    if date_col_name is None:
        candidates = [c for c in df.columns if 'DATE' in c.upper()]
        if not candidates:
            raise ValueError("No date column present to filter")
        date_col_name = candidates[0]
    df = df.loc[pd.to_datetime(df[date_col_name], errors='coerce') >= start_date].copy()
    logger.info("Filtered data to %s onwards: rows=%d", start_date.date(), len(df))
    return df


def compute_style_metrics(df: pd.DataFrame, group_cols=('STYLE',)) -> pd.DataFrame:
    """Aggregate important metrics per group (usually per STYLE)."""
    g = df.groupby(list(group_cols))

    agg = g.agg(
        units_sold=pd.NamedAgg(column='BILL_QUANTITY', aggfunc='sum'),
        revenue=pd.NamedAgg(column='NET_AMOUNT', aggfunc='sum'),
        gross_amount=pd.NamedAgg(column='GROSS_AMOUNT', aggfunc='sum') if 'GROSS_AMOUNT' in df.columns else pd.NamedAgg(column='NET_AMOUNT', aggfunc='sum'),
        avg_rsp=pd.NamedAgg(column='RSP', aggfunc=lambda x: np.nanmean([v for v in x if pd.notna(v) and v>0]) if len(x)>0 else np.nan),
        avg_item_discount=pd.NamedAgg(column='ITEM_DISCOUNT_AMOUNT', aggfunc='sum'),
        promo_amount=pd.NamedAgg(column='PROMO_AMOUNT', aggfunc='sum') if 'PROMO_AMOUNT' in df.columns else pd.NamedAgg(column='ITEM_PROMO_NAME' if 'ITEM_PROMO_NAME' in df.columns else 'BILL_NO', aggfunc=lambda x: 0),
        void_count=pd.NamedAgg(column='VOID STATUS', aggfunc=lambda x: x.eq('Y').sum() if 'VOID STATUS' in df.columns else 0),
        first_sale=pd.NamedAgg(column='BILL_DATE', aggfunc='min'),
        last_sale=pd.NamedAgg(column='BILL_DATE', aggfunc='max'),
    )

    # Derived metrics
    agg = agg.reset_index()
    agg['avg_price'] = agg.apply(lambda r: safe_div(r['revenue'], r['units_sold']), axis=1)
    agg['avg_item_discount_per_unit'] = agg.apply(lambda r: safe_div(r['avg_item_discount'], r['units_sold']), axis=1)
    # weeks on sale
    agg['weeks_on_sale'] = (pd.to_datetime(agg['last_sale']) - pd.to_datetime(agg['first_sale'])).dt.days.div(7).clip(lower=1)
    agg['velocity_units_per_week'] = agg.apply(lambda r: safe_div(r['units_sold'], r['weeks_on_sale']), axis=1)

    # Share and rank
    total_units = agg['units_sold'].sum()
    agg['share_of_units_pct'] = agg['units_sold'] / total_units * 100
    agg['revenue_rank'] = agg['revenue'].rank(method='dense', ascending=False).astype(int)

    return agg


def size_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute size concentration (Herfindahl index) per style; higher value -> concentrated few sizes."""
    if 'SIZE' not in df.columns:
        return pd.DataFrame(columns=['STYLE', 'size_hhi'])
    g = df.groupby(['STYLE', 'SIZE'])['BILL_QUANTITY'].sum().reset_index()
    out = []
    for style, grp in g.groupby('STYLE'):
        s = grp['BILL_QUANTITY']
        shares = s / s.sum() if s.sum()>0 else s*0
        hhi = (shares ** 2).sum()
        out.append({'STYLE': style, 'size_hhi': hhi, 'unique_sizes': grp['SIZE'].nunique()})
    return pd.DataFrame(out)


def detect_reason_flags(style_metrics: pd.DataFrame, size_conc: pd.DataFrame) -> pd.DataFrame:
    """Add heuristic flags for potential reasons why a style performed poorly or well."""
    df = style_metrics.copy()
    df = df.merge(size_conc, on='STYLE', how='left')

    # thresholds (data driven)
    median_velocity = df['velocity_units_per_week'].median() if 'velocity_units_per_week' in df.columns else 0
    std_velocity = df['velocity_units_per_week'].std(ddof=0) if 'velocity_units_per_week' in df.columns else 0
    median_discount = (df['avg_item_discount_per_unit'] / df['avg_price']).median() if 'avg_item_discount_per_unit' in df.columns and 'avg_price' in df.columns else 0
    median_hhi = df['size_hhi'].median() if 'size_hhi' in df.columns else 0

    df['flag_low_velocity'] = df['velocity_units_per_week'] < max(1, median_velocity - std_velocity)
    # flag high discounts relative to price
    df['flag_high_discount_pct'] = (df['avg_item_discount_per_unit'] / df['avg_price']).fillna(0) > max(0.05, median_discount * 1.5)
    df['flag_high_size_concentration'] = df['size_hhi'].fillna(0) > max(0.2, median_hhi * 1.2)
    # flag many voids relative to units
    df['void_rate'] = df.apply(lambda r: safe_div(r.get('void_count', 0), r['units_sold']), axis=1)
    df['flag_high_void_rate'] = df['void_rate'] > 0.02

    # composite reason column
    def reasons(row):
        rs = []
        if row['flag_low_velocity']:
            rs.append('low_velocity')
        if row['flag_high_discount_pct']:
            rs.append('high_discount')
        if row['flag_high_size_concentration']:
            rs.append('size_concentration')
        if row['flag_high_void_rate']:
            rs.append('high_voids')
        if not rs:
            rs.append('none')
        return ','.join(rs)

    df['reasons_flags'] = df.apply(reasons, axis=1)
    return df


def summarize_by_gender(df: pd.DataFrame, start_date: pd.Timestamp) -> dict:
    """Produce analysis for Women and Boys genders and return a dict of DataFrames to be exported."""
    out = {}
    # Ensure gender column exists (case-insensitive)
    gender_col = next((c for c in df.columns if c.strip().upper() == 'GENDER'), None)
    if gender_col is None:
        logger.warning("No GENDER column available in merged data; all analysis will be run on full set and gender will be empty.")
        df['GENDER'] = np.nan
        gender_col = 'GENDER'

    df_filtered = df.copy()
    # standardize common column names
    if 'BILL_DATE' in df_filtered.columns:
        df_filtered['BILL_DATE'] = pd.to_datetime(df_filtered['BILL_DATE'], errors='coerce')

    for gender in ['Women', 'Boys']:
        gender_df = df_filtered.loc[df_filtered[gender_col].str.strip().str.lower() == gender.lower()].copy()
        if gender_df.empty:
            logger.info("No rows found for gender=%s", gender)
            out[gender] = {
                'style_metrics': pd.DataFrame(),
                'top_by_revenue': pd.DataFrame(),
                'bottom_by_revenue': pd.DataFrame(),
                'size_concentration': pd.DataFrame(),
                'reason_flags': pd.DataFrame(),
                'raw': gender_df
            }
            continue
        # compute metrics
        style_metrics = compute_style_metrics(gender_df, group_cols=('STYLE',))
        size_conc = size_concentration(gender_df)
        flagged = detect_reason_flags(style_metrics, size_conc)

        top_by_revenue = flagged.sort_values('revenue', ascending=False).head(20)
        bottom_by_revenue = flagged.sort_values('revenue', ascending=True).head(20)

        # time series per style (monthly)
        gender_df['month'] = gender_df['BILL_DATE'].dt.to_period('M')
        ts = gender_df.groupby(['STYLE', 'month']).agg(units_sold=('BILL_QUANTITY', 'sum'), revenue=('NET_AMOUNT', 'sum')).reset_index()

        out[gender] = {
            'style_metrics': style_metrics,
            'top_by_revenue': top_by_revenue,
            'bottom_by_revenue': bottom_by_revenue,
            'size_concentration': size_conc,
            'reason_flags': flagged[['STYLE', 'reasons_flags', 'flag_low_velocity', 'flag_high_discount_pct', 'flag_high_size_concentration', 'flag_high_void_rate', 'void_rate']],
            'time_series': ts,
            'raw': gender_df
        }

    return out


### Demand planning helpers
def week_start(dt_series: pd.Series) -> pd.Series:
    return pd.to_datetime(dt_series).dt.to_period('W').apply(lambda r: r.start_time)


def z_for_service_level(p: float) -> float:
    """Return z-score for common service levels. If p not in table, linearly interpolate between nearest values."""
    table = {
        0.8: 0.841,
        0.85: 1.036,
        0.9: 1.282,
        0.95: 1.645,
        0.975: 1.96,
        0.98: 2.054,
        0.99: 2.326,
    }
    if p in table:
        return table[p]
    keys = sorted(table.keys())
    if p <= keys[0]:
        return table[keys[0]]
    if p >= keys[-1]:
        return table[keys[-1]]
    # linear interpolation
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if a < p < b:
            za, zb = table[a], table[b]
            t = (p - a) / (b - a)
            return za + t * (zb - za)
    return 1.65


def compute_weekly_demand_and_forecast(df: pd.DataFrame, horizon_weeks: int = 12, history_weeks: int = 26, alpha: float = 0.3) -> pd.DataFrame:
    """Compute weekly demand per SKU and a simple forecast using EWMA and SMA.

    Returns a DataFrame with per-SKU stats and forecast for next `horizon_weeks` weeks (single value = avg weekly forecast).
    """
    if 'BILL_DATE' not in df.columns:
        raise ValueError('BILL_DATE required for weekly demand computation')

    # Ensure date
    df = df.copy()
    df['BILL_DATE'] = pd.to_datetime(df['BILL_DATE'], errors='coerce')
    df = df.loc[df['BILL_DATE'].notna()]

    df['week_start'] = week_start(df['BILL_DATE'])

    # Choose grouping SKU if present else STYLE
    group_col = 'SKU' if 'SKU' in df.columns else 'STYLE'

    weekly = df.groupby([group_col, 'week_start']).agg(units_sold=('BILL_QUANTITY', 'sum')).reset_index()

    # limit to recent history_weeks per SKU
    last_week = weekly['week_start'].max()
    cutoff = last_week - pd.Timedelta(weeks=history_weeks)
    weekly_recent = weekly.loc[weekly['week_start'] > cutoff].copy()

    rows = []
    for sku, grp in weekly_recent.groupby(group_col):
        series = grp.set_index('week_start').reindex(pd.date_range(start=grp['week_start'].min(), end=last_week, freq='W-MON')).fillna(0)
        series_units = series['units_sold'].astype(float)
        mean_weekly = series_units.mean()
        std_weekly = series_units.std(ddof=0)
        cv = safe_div(std_weekly, mean_weekly)
        # EWMA forecast (next week) and horizon average = ewma value
        ewma = series_units.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        # SMA over last 4 weeks
        sma_4 = series_units.tail(4).mean()
        # Use blended forecast: weighted avg of ewma and sma
        forecast_week = 0.6 * ewma + 0.4 * sma_4

        rows.append({
            group_col: sku,
            'mean_weekly': mean_weekly,
            'std_weekly': std_weekly,
            'cv_weekly': cv,
            'ewma_weekly': ewma,
            'sma4_weekly': sma_4,
            'forecast_weekly': forecast_week,
            'history_weeks': len(series_units)
        })

    res = pd.DataFrame(rows)
    # Add horizon totals
    res['forecast_horizon_units'] = res['forecast_weekly'] * horizon_weeks
    return res


def compute_reorder_recommendations(forecast_df: pd.DataFrame, lead_time_weeks: int = 4, service_level: float = 0.95, target_cover_weeks: int = 12) -> pd.DataFrame:
    """Compute safety stock, reorder point and order-up-to recommendations per SKU.

    Note: On-hand inventory not available; recommended_order_up_to is provided (user must subtract on-hand to calculate order qty).
    """
    df = forecast_df.copy()
    z = z_for_service_level(service_level)
    # safety stock = z * std_weekly * sqrt(lead_time)
    df['safety_stock_units'] = df['std_weekly'] * (lead_time_weeks ** 0.5) * z
    df['reorder_point_units'] = df['forecast_weekly'] * lead_time_weeks + df['safety_stock_units']
    # order up to = forecast for target_cover_weeks + safety stock
    df['order_up_to_units'] = df['forecast_weekly'] * target_cover_weeks + df['safety_stock_units']
    # round up
    for c in ['safety_stock_units', 'reorder_point_units', 'order_up_to_units']:
        df[c] = np.ceil(df[c].fillna(0)).astype(int)
    # priority score for replenishment: forecast_horizon_units * cv (higher cv means more variable)
    df['priority_score'] = df['forecast_horizon_units'] * (1 + df['cv_weekly'].fillna(0))
    df = df.sort_values('priority_score', ascending=False)
    return df


def export_to_excel(out_dict: dict, out_path: Path):
    """Write the analysis dict to an Excel workbook with multiple sheets."""
    out_path = Path(out_path)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # summary sheet
        summary_rows = []
        for gender, data in out_dict.items():
            sm = data.get('style_metrics')
            if sm is None or sm.empty:
                summary_rows.append({'gender': gender, 'total_styles': 0, 'total_units': 0, 'total_revenue': 0})
            else:
                summary_rows.append({'gender': gender, 'total_styles': len(sm), 'total_units': int(sm['units_sold'].sum()), 'total_revenue': float(sm['revenue'].sum())})
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='summary', index=False)

        for gender, data in out_dict.items():
            # skip internal overall bucket
            if gender == '__overall__':
                continue
            if data is None or not isinstance(data, dict):
                continue
            prefix = gender.lower()
            raw_df = data.get('raw')
            if raw_df is not None and not raw_df.empty:
                raw_df.to_excel(writer, sheet_name=f'{prefix}_raw', index=False)
            if not data.get('style_metrics', pd.DataFrame()).empty:
                data['style_metrics'].to_excel(writer, sheet_name=f'{prefix}_style_metrics', index=False)
            if not data.get('top_by_revenue', pd.DataFrame()).empty:
                data['top_by_revenue'].to_excel(writer, sheet_name=f'{prefix}_top20', index=False)
            if not data.get('bottom_by_revenue', pd.DataFrame()).empty:
                data['bottom_by_revenue'].to_excel(writer, sheet_name=f'{prefix}_bottom20', index=False)
            if not data.get('reason_flags', pd.DataFrame()).empty:
                data['reason_flags'].to_excel(writer, sheet_name=f'{prefix}_reasons', index=False)
            if not data.get('time_series', pd.DataFrame()).empty:
                # write a time series sheet; pivot for easier viewing
                ts = data['time_series'].copy()
                ts['month'] = ts['month'].astype(str)
                ts_pivot = ts.pivot_table(index='month', columns='STYLE', values='units_sold', aggfunc='sum').fillna(0)
                ts_pivot.to_excel(writer, sheet_name=f'{prefix}_ts_monthly')

            # demand planning outputs if present
            if data.get('demand_forecast') is not None and not data['demand_forecast'].empty:
                data['demand_forecast'].to_excel(writer, sheet_name=f'{prefix}_demand_forecast', index=False)
            if data.get('reorder_reco') is not None and not data['reorder_reco'].empty:
                data['reorder_reco'].to_excel(writer, sheet_name=f'{prefix}_reorder_reco', index=False)

            # Write gender-specific analytics
            # departments and categories
            dept_cat = data.get('overall_dept_cat', {})
            dept = dept_cat.get('department')
            if dept is not None and not dept.empty:
                dept.to_excel(writer, sheet_name=f'{prefix}_department', index=False)
            cat = dept_cat.get('category')
            if cat is not None and not cat.empty:
                cat.to_excel(writer, sheet_name=f'{prefix}_category', index=False)

            # top lists
            top = data.get('overall_top_lists', {})
            for k, df in top.items():
                if df is not None and not df.empty:
                    # sheet names must be <=31 chars; truncate
                    sheet = (f"{prefix}_{k}")[:31]
                    df.to_excel(writer, sheet_name=sheet, index=False)

            # style daily metrics
            sdm = data.get('overall_style_daily')
            if sdm is not None and not sdm.empty:
                sdm.to_excel(writer, sheet_name=f'{prefix}_style_daily', index=False)

            # monthly key styles (by revenue and by units)
            mks = data.get('overall_monthly_key_styles', {})
            if isinstance(mks, dict):
                by_rev = mks.get('by_revenue')
                if by_rev is not None and not by_rev.empty:
                    by_rev.to_excel(writer, sheet_name=f'{prefix}_monthly_by_revenue', index=False)
                by_units = mks.get('by_units')
                if by_units is not None and not by_units.empty:
                    by_units.to_excel(writer, sheet_name=f'{prefix}_monthly_by_units', index=False)

        # Write glossary once (same for all genders)
        glossary = build_glossary()
        if glossary is not None and not glossary.empty:
            glossary.to_excel(writer, sheet_name='glossary', index=False)

        logger.info('Appended overall sheets to %s', out_path)


def get_style_first_sale(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Return first sale date per STYLE (using the full sales dataframe)."""
    df = sales_df.copy()
    # find date column
    date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)
    if date_col is None:
        raise ValueError('No date column found in sales to compute first sale')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    res = df.loc[df[date_col].notna()].groupby('STYLE')[date_col].min().reset_index().rename(columns={date_col: 'first_sale_date'})
    return res


def compute_department_and_category_aggregates(merged: pd.DataFrame) -> dict:
    """Aggregate metrics at department and category level."""
    out = {}
    if 'DEPARTMENT' in merged.columns:
        dept = merged.groupby('DEPARTMENT').agg(units_sold=('BILL_QUANTITY', 'sum'), revenue=('NET_AMOUNT', 'sum')).reset_index()
        dept['avg_price'] = dept.apply(lambda r: safe_div(r['revenue'], r['units_sold']), axis=1)
        out['department'] = dept.sort_values('revenue', ascending=False)
    else:
        out['department'] = pd.DataFrame()

    # Category may exist from style master
    cat_col = next((c for c in merged.columns if c.strip().upper() == 'CATEGORY'), None)
    if cat_col:
        cat = merged.groupby(cat_col).agg(units_sold=('BILL_QUANTITY', 'sum'), revenue=('NET_AMOUNT', 'sum')).reset_index()
        cat['avg_price'] = cat.apply(lambda r: safe_div(r['revenue'], r['units_sold']), axis=1)
        out['category'] = cat.sort_values('revenue', ascending=False)
    else:
        out['category'] = pd.DataFrame()

    return out


def top_lists_by_metrics(merged: pd.DataFrame, top_n: int = 50) -> dict:
    """Return top styles and SKUs by various metrics."""
    results = {}
    # style-level
    if 'STYLE' in merged.columns:
        style_agg = merged.groupby('STYLE').agg(units_sold=('BILL_QUANTITY', 'sum'), revenue=('NET_AMOUNT', 'sum'), avg_price=('RSP', 'mean')).reset_index()
        style_agg['asp'] = style_agg['revenue'] / style_agg['units_sold'].replace(0, np.nan)
        results['top_styles_by_units'] = style_agg.sort_values('units_sold', ascending=False).head(top_n)
        results['top_styles_by_revenue'] = style_agg.sort_values('revenue', ascending=False).head(top_n)
        results['top_styles_by_asp'] = style_agg.sort_values('asp', ascending=False).head(top_n)
    else:
        results['top_styles_by_units'] = pd.DataFrame()
        results['top_styles_by_revenue'] = pd.DataFrame()
        results['top_styles_by_asp'] = pd.DataFrame()

    # SKU-level
    sku_col = 'SKU' if 'SKU' in merged.columns else None
    if sku_col:
        sku_agg = merged.groupby(sku_col).agg(units_sold=('BILL_QUANTITY', 'sum'), revenue=('NET_AMOUNT', 'sum'), avg_price=('RSP', 'mean')).reset_index()
        sku_agg['asp'] = sku_agg['revenue'] / sku_agg['units_sold'].replace(0, np.nan)
        results['top_skus_by_units'] = sku_agg.sort_values('units_sold', ascending=False).head(top_n)
        results['top_skus_by_revenue'] = sku_agg.sort_values('revenue', ascending=False).head(top_n)
        results['top_skus_by_asp'] = sku_agg.sort_values('asp', ascending=False).head(top_n)
    else:
        results['top_skus_by_units'] = pd.DataFrame()
        results['top_skus_by_revenue'] = pd.DataFrame()
        results['top_skus_by_asp'] = pd.DataFrame()

    return results


def compute_style_daily_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute daily/inventory related metrics per style to infer stockouts and sales-days.

    Metrics:
    - first_sale, last_sale, days_between
    - days_with_sales, days_without_sales_between
    - median_daily_sales, days_above_median
    - longest_zero_gap (consecutive days with zero sales between first and last)
    - max_daily_run_rate (maximum units sold in a single day)
    - max_weekly_run_rate (maximum units sold in any 7-day period)
    - max_monthly_run_rate (maximum units sold in any calendar month)
    """
    if 'BILL_DATE' not in merged.columns:
        return pd.DataFrame()
    df = merged.copy()
    df['BILL_DATE'] = pd.to_datetime(df['BILL_DATE'], errors='coerce')
    daily = df.groupby(['STYLE', 'BILL_DATE']).agg(units=('BILL_QUANTITY', 'sum')).reset_index()

    rows = []
    for style, grp in daily.groupby('STYLE'):
        grp = grp.sort_values('BILL_DATE')
        first = grp['BILL_DATE'].min()
        last = grp['BILL_DATE'].max()
        days_between = (last - first).days + 1
        
        # Calculate maximum daily run rate
        max_drr = grp['units'].max()
        
        # Calculate maximum weekly run rate (rolling 7-day window)
        grp = grp.sort_values('BILL_DATE')
        grp['rolling_7d'] = grp.set_index('BILL_DATE')['units'].rolling('7D').sum()
        max_wrr = grp['rolling_7d'].max()
        
        # Calculate maximum monthly run rate
        grp['month'] = grp['BILL_DATE'].dt.to_period('M')
        monthly_sales = grp.groupby('month')['units'].sum()
        max_mrr = monthly_sales.max()
        
        # create full date range
        all_dates = pd.DataFrame({'BILL_DATE': pd.date_range(first, last)})
        merged_dates = all_dates.merge(grp[['BILL_DATE', 'units']], on='BILL_DATE', how='left').fillna(0)
        units = merged_dates['units'].values
        days_with_sales = (units > 0).sum()
        days_without_sales = (units == 0).sum()
        median_daily = np.median(units)
        days_above_median = (units > median_daily).sum()
        # compute longest consecutive zeros
        zero_runs = (units == 0).astype(int)
        # compute lengths of consecutive ones in zero_runs
        max_zero_run = 0
        cur = 0
        for v in zero_runs:
            if v == 1:
                cur += 1
            else:
                max_zero_run = max(max_zero_run, cur)
                cur = 0
        max_zero_run = max(max_zero_run, cur)

        rows.append({
            'STYLE': style, 
            'first_sale': first, 
            'last_sale': last, 
            'days_between': days_between, 
            'days_with_sales': int(days_with_sales), 
            'days_without_sales': int(days_without_sales), 
            'median_daily_sales': float(median_daily), 
            'days_above_median': int(days_above_median), 
            'longest_zero_gap_days': int(max_zero_run),
            'max_daily_run_rate': float(max_drr),
            'max_weekly_run_rate': float(max_wrr) if not pd.isna(max_wrr) else 0.0,
            'max_monthly_run_rate': float(max_mrr) if not pd.isna(max_mrr) else 0.0
        })

    return pd.DataFrame(rows)


def monthly_key_styles(merged: pd.DataFrame, top_n: int = 10) -> dict:
    """For each month, list top N styles by revenue and by quantity along with category/department.
    Returns a dict with two DataFrames: 'by_revenue' and 'by_units'
    """
    df = merged.copy()
    if 'BILL_DATE' not in df.columns:
        return {'by_revenue': pd.DataFrame(), 'by_units': pd.DataFrame()}

    df['month'] = pd.to_datetime(df['BILL_DATE']).dt.to_period('M').astype(str)
    grp = df.groupby(['month', 'STYLE']).agg(
        units_sold=('BILL_QUANTITY', 'sum'), 
        revenue=('NET_AMOUNT', 'sum')
    ).reset_index()

    # attach category/department from merged
    meta = merged[['STYLE'] + [c for c in merged.columns if c.strip().upper() in ('CATEGORY', 'DEPARTMENT')]].drop_duplicates('STYLE')
    grp = grp.merge(meta, on='STYLE', how='left')

    # Process by revenue
    revenue_rows = []
    for month, g in grp.groupby('month'):
        top = g.sort_values('revenue', ascending=False).head(top_n).reset_index(drop=True)
        for i, r in top.iterrows():
            revenue_rows.append({
                'month': month, 
                'rank': i+1, 
                'STYLE': r['STYLE'], 
                'revenue': r['revenue'], 
                'units_sold': r['units_sold'], 
                'CATEGORY': r.get('CATEGORY', None), 
                'DEPARTMENT': r.get('DEPARTMENT', None)
            })

    # Process by units
    units_rows = []
    for month, g in grp.groupby('month'):
        top = g.sort_values('units_sold', ascending=False).head(top_n).reset_index(drop=True)
        for i, r in top.iterrows():
            units_rows.append({
                'month': month, 
                'rank': i+1, 
                'STYLE': r['STYLE'], 
                'revenue': r['revenue'], 
                'units_sold': r['units_sold'], 
                'CATEGORY': r.get('CATEGORY', None), 
                'DEPARTMENT': r.get('DEPARTMENT', None)
            })

    return {
        'by_revenue': pd.DataFrame(revenue_rows),
        'by_units': pd.DataFrame(units_rows)
    }


def build_glossary() -> pd.DataFrame:
    """Return a DataFrame describing all computed keywords/columns used in outputs."""
    terms = [
        ('units_sold', 'Total units sold for the grouping (sum of BILL_QUANTITY).'),
        ('revenue', 'Total NET_AMOUNT for the grouping.'),
        ('avg_price / asp', 'Average selling price (RSP or revenue/units).'),
        ('velocity_units_per_week', 'Average weekly units sold (units_sold / weeks_on_sale).'),
        ('size_hhi', 'Size concentration using Herfindahl index (sum of squared size shares). Higher -> concentrated in few sizes.'),
        ('flag_low_velocity', 'Heuristic flag indicating lower-than-expected velocity.'),
        ('flag_high_discount_pct', 'Heuristic flag indicating high discount per unit relative to price.'),
        ('flag_high_size_concentration', 'Heuristic flag indicating high size concentration.'),
        ('flag_high_void_rate', 'Heuristic flag indicating high voids/returns relative to units.'),
        ('forecast_weekly', 'Blended forecast of weekly demand using EWMA + SMA.'),
        ('forecast_horizon_units', 'Forecasted units over the selected horizon (forecast_weekly * horizon_weeks).'),
        ('safety_stock_units', 'Safety stock computed as z * std_weekly * sqrt(lead_time_weeks).'),
        ('reorder_point_units', 'Reorder point = forecast_weekly * lead_time_weeks + safety_stock_units.'),
        ('order_up_to_units', 'Recommended order-up-to level covering target_cover_weeks + safety stock.'),
        ('priority_score', 'Simple priority score for replenishment = forecast_horizon_units * (1 + cv).'),
        ('first_sale_date / first_sale', 'Date of first recorded sale for the STYLE.'),
        ('days_with_sales', 'Number of days with >0 sales between first and last sale for the style.'),
        ('longest_zero_gap_days', 'Longest consecutive run of zero-sales days between first and last sale (possible stockout).'),
        ('launched_after_april', 'Boolean indicating the STYLE had its first sale on or after the analysis start date (launched after April 2025).'),
        ('max_daily_run_rate', 'Maximum number of units sold in a single day for the style'),
        ('max_weekly_run_rate', 'Maximum number of units sold in any rolling 7-day period for the style'),
        ('max_monthly_run_rate', 'Maximum number of units sold in any calendar month for the style'),
    ]
    return pd.DataFrame(terms, columns=['term', 'description'])


def main(args=None):
    parser = argparse.ArgumentParser(description='Run Women/Boys style analysis from EBO sales data')
    parser.add_argument('--sales', required=False, help='Path to sales Excel file', default=None)
    parser.add_argument('--sku', required=False, help='Path to SKU master Excel file', default=None)
    parser.add_argument('--style', required=False, help='Path to STYLE master Excel file', default=None)
    parser.add_argument('--start-date', required=False, help='Start date (YYYY-MM-DD) for analysis', default=str(DEFAULT_START_DATE.date()))
    parser.add_argument('--output', required=False, help='Output Excel path', default='gender_style_analysis.xlsx')
    parser.add_argument('--lead-time-weeks', required=False, type=int, default=4, help='Assumed lead time in weeks for reorder point calculations')
    parser.add_argument('--service-level', required=False, type=float, default=0.95, help='Service level for safety stock (e.g., 0.95)')
    parser.add_argument('--forecast-horizon-weeks', required=False, type=int, default=12, help='Forecast horizon (weeks) used for horizon forecast and order-up-to')

    parsed = parser.parse_args(args=args)

    # sensible defaults if not provided (use the paths user gave earlier)
    sales_path = Path(parsed.sales) if parsed.sales else Path(r"D:\DATA TILL DATE\Desktop\EBO FOLDER\EBO SALES FOLDER\EBO SALES DATA.xlsx")
    sku_path = Path(parsed.sku) if parsed.sku else Path(r"D:\DATA TILL DATE\Desktop\EBO FOLDER\MASTERS\SKU MASTER.xlsx")
    style_path = Path(parsed.style) if parsed.style else Path(r"D:\DATA TILL DATE\Desktop\EBO FOLDER\MASTERS\STYLE MASTER.xlsx")

    start_date = pd.to_datetime(parsed.start_date)

    logger.info("Loading files")
    sales_df = load_excel_any(sales_path)
    sku_df = load_excel_any(sku_path)
    style_df = load_excel_any(style_path)

    # compute first sale per style from full sales (before we filter by start_date)
    try:
        style_first_sales = get_style_first_sale(sales_df)
    except Exception as e:
        logger.warning("Could not compute first sale per style: %s", e)
        style_first_sales = pd.DataFrame(columns=['STYLE', 'first_sale_date'])

    logger.info("Preparing and merging data")
    merged = prepare_data(sales_df, sku_df, style_df)

    logger.info("Filtering date range from %s", start_date.date())
    merged = filter_date_range(merged, start_date, date_col_name='BILL_DATE' if 'BILL_DATE' in merged.columns else None)

    # attach first sale date and launched flag
    if not style_first_sales.empty and 'STYLE' in merged.columns:
        merged = merged.merge(style_first_sales, on='STYLE', how='left')
        merged['first_sale_date'] = pd.to_datetime(merged['first_sale_date'], errors='coerce')
        merged['launched_after_april'] = merged['first_sale_date'] >= start_date
    else:
        merged['first_sale_date'] = pd.NaT
        merged['launched_after_april'] = False

    logger.info("Running gender-based summarization")
    out = summarize_by_gender(merged, start_date)

    # Demand planning: compute weekly demand and forecasts per gender
    logger.info("Computing demand-planning metrics (lead_time=%swks, svc_level=%s, horizon=%swks)", parsed.lead_time_weeks, parsed.service_level, parsed.forecast_horizon_weeks)
    for gender, data in out.items():
        raw = data.get('raw')
        if raw is None or raw.empty:
            data['demand_forecast'] = pd.DataFrame()
            data['reorder_reco'] = pd.DataFrame()
            continue
        weekly_forecast = compute_weekly_demand_and_forecast(raw, horizon_weeks=parsed.forecast_horizon_weeks)
        reorder = compute_reorder_recommendations(weekly_forecast, lead_time_weeks=parsed.lead_time_weeks, service_level=parsed.service_level, target_cover_weeks=parsed.forecast_horizon_weeks)
        data['demand_forecast'] = weekly_forecast
        data['reorder_reco'] = reorder
        # merchandising/demand insights scoped to this gender (so outputs focus on Women and Boys only)
        data['overall_dept_cat'] = compute_department_and_category_aggregates(raw)
        data['overall_top_lists'] = top_lists_by_metrics(raw, top_n=100)
        data['overall_style_daily'] = compute_style_daily_metrics(raw)
        data['overall_monthly_key_styles'] = monthly_key_styles(raw, top_n=20)

    # build a single glossary (same for both genders)
    glossary = build_glossary()

    out_path = Path(parsed.output)
    try:
        export_to_excel(out, out_path)
        logger.info("Done. Output saved to %s", out_path)
    except PermissionError as e:
        # likely the file is open in Excel; retry with timestamped filename
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        alt = out_path.with_name(out_path.stem + f"_{ts}" + out_path.suffix)
        logger.warning("Could not write to %s (PermissionError). Retrying with %s", out_path, alt)
        export_to_excel(out, alt)
        logger.info("Done. Output saved to %s", alt)


if __name__ == '__main__':
    main()
