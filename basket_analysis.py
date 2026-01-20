import argparse
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_excel_any(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load an Excel file (first sheet by default) into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    sheet_to_use = sheet_name if sheet_name is not None else 0
    df = pd.read_excel(path, sheet_name=sheet_to_use)
    if isinstance(df, dict):
        first_key = list(df.keys())[0]
        df = df[first_key]
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def prepare_october_data(sales_df: pd.DataFrame, sku_df: pd.DataFrame, style_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and merge sales data, filter to October, and attach SKU/style information."""
    df = sales_df.copy()
    
    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]
    sku_df.columns = [c.strip().upper() for c in sku_df.columns]
    style_df.columns = [c.strip().upper() for c in style_df.columns]
    
    # Remove styles marked as freebies in style master
    if 'GENDER' in style_df.columns:
        # Keep only non-freebie styles
        non_freebie_styles = style_df[~style_df['GENDER'].str.strip().str.upper().eq('FREEBIES')]['STYLE']
        style_df = style_df[style_df['STYLE'].isin(non_freebie_styles)]
        logger.info("Filtered out freebie styles from style master: kept %d styles", len(style_df))
    
    # List of promotional/gift items to exclude
    promo_items = {
        'POWERBANK 300 LITE',
        'UNISEX GYM BAG',
        'AIRDOPES JOY',
        'TROLLEY CASE',
        'BAMBREW CARRY BAG',
        'Boat Airdopes 192'
    }
    
    # Remove freebies (items with zero or negative net amount)
    if 'NET_AMOUNT' in df.columns:
        df = df[df['NET_AMOUNT'] > 0]
        logger.info("Removed freebies (zero/negative amount items): %d rows remaining", len(df))
    
    # Remove items marked as free in promo
    promo_cols = [col for col in df.columns if 'PROMO' in col.upper()]
    for col in promo_cols:
        if df[col].dtype == object:  # If text column
            free_mask = df[col].str.contains('free|gift', case=False, na=False)
            df = df[~free_mask]
            logger.info("Removed items marked as free in %s: %d rows remaining", col, len(df))
    
    # Remove known promotional/gift items by description
    item_desc_col = next((col for col in df.columns if any(x in col.upper() for x in ['DESCRIPTION', 'ITEM_DESC', 'DESC'])), None)
    if item_desc_col:
        df['TEMP_DESC'] = df[item_desc_col].str.upper() if df[item_desc_col].dtype == object else df[item_desc_col]
        mask = ~df['TEMP_DESC'].str.contains('|'.join(x.upper() for x in promo_items), na=False)
        df = df[mask]
        df = df.drop('TEMP_DESC', axis=1)
        logger.info("Removed known promotional items: %d rows remaining", len(df))
    
    # Also try to remove by SKU description if available
    if 'SKU' in df.columns and 'DESCRIPTION' in sku_df.columns:
        sku_mask = ~sku_df['DESCRIPTION'].str.contains('|'.join(x.upper() for x in promo_items), na=False, case=False)
        valid_skus = set(sku_df[sku_mask]['SKU'].unique())
        df = df[df['SKU'].isin(valid_skus)]
        logger.info("Removed promotional SKUs: %d rows remaining", len(df))
    
    # Parse and filter dates
    date_col = next((c for c in df.columns if 'DATE' in c), None)
    if date_col is None:
        raise ValueError("No date column found in sales data")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    oct_mask = (df[date_col].dt.year == 2025) & (df[date_col].dt.month == 10)
    df = df.loc[oct_mask]
    logger.info("Filtered to October 2025: %d rows", len(df))
    
    # Merge SKU data
    if 'SKU' in df.columns and 'SKU' in sku_df.columns:
        df = df.merge(sku_df, on='SKU', how='left', suffixes=('', '_SKU'))
    elif 'ICODE' in df.columns and 'SKU' in sku_df.columns:
        df = df.merge(sku_df, left_on='ICODE', right_on='SKU', how='left', suffixes=('', '_SKU'))
    
    # Merge style data and filter out freebies
    if 'STYLE' in df.columns and 'STYLE' in style_df.columns:
        # Use inner merge to keep only styles that exist in the filtered style master
        df = df.merge(style_df, on='STYLE', how='inner', suffixes=('', '_STYLE'))
        logger.info("Merged with style master and filtered to non-freebie styles: %d rows", len(df))
    
    return df


def create_basket_matrix(df: pd.DataFrame, group_col: str, basket_col: str = 'BILL_NO') -> pd.DataFrame:
    """Create a binary basket matrix showing which items appear in each transaction."""
    # Create item presence matrix (1 if item in basket, 0 if not)
    basket_matrix = pd.crosstab(df[basket_col], df[group_col])
    # Convert to binary (some items might appear multiple times in same transaction)
    basket_matrix = (basket_matrix > 0).astype(int)
    return basket_matrix


def compute_association_rules(basket_matrix: pd.DataFrame, 
                            min_support: float = 0.01,
                            min_confidence: float = 0.1,
                            min_lift: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute frequent itemsets and association rules from basket matrix."""
    # Find frequent itemsets
    frequent_itemsets = apriori(basket_matrix, 
                               min_support=min_support,
                               use_colnames=True)
    
    # Generate rules
    rules = association_rules(frequent_itemsets, 
                            metric="confidence",
                            min_threshold=min_confidence)
    
    # Filter rules by lift
    rules = rules.loc[rules['lift'] >= min_lift]
    
    # Sort by lift descending
    rules = rules.sort_values('lift', ascending=False)
    
    return frequent_itemsets, rules


def add_metadata_to_rules(rules: pd.DataFrame, 
                         df: pd.DataFrame, 
                         group_col: str,
                         meta_cols: List[str]) -> pd.DataFrame:
    """Add metadata (category, department, etc.) to association rules for better interpretation."""
    # Create a mapping of item to its metadata
    meta_map = df.groupby(group_col)[meta_cols].first().to_dict()
    
    # Function to get metadata for a frozen set of items
    def get_meta_str(items: frozenset, col: str) -> str:
        return ' + '.join(str(meta_map[col].get(item, 'Unknown') or 'Unknown') for item in items)
    
    # Add metadata columns for antecedents and consequents
    for col in meta_cols:
        rules[f'antecedent_{col}'] = rules['antecedents'].apply(lambda x: get_meta_str(x, col))
        rules[f'consequent_{col}'] = rules['consequents'].apply(lambda x: get_meta_str(x, col))
    
    return rules


def analyze_common_pairs(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Find most common item pairs in transactions regardless of order."""
    pairs = []
    for bill_no, group in df.groupby('BILL_NO'):
        # Filter out null values and convert to strings for consistent sorting
        items = sorted(str(x) for x in group[group_col].unique() if pd.notna(x))
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                pairs.append((items[i], items[j]))
    
    pair_df = pd.DataFrame(pairs, columns=['item1', 'item2'])
    pair_counts = pair_df.groupby(['item1', 'item2']).size().reset_index(name='frequency')
    pair_counts = pair_counts.sort_values('frequency', ascending=False)
    
    return pair_counts


def analyze_basket_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze distribution of items per transaction."""
    basket_sizes = df.groupby('BILL_NO').size()
    
    stats = pd.DataFrame([{
        'metric': 'Average items per basket',
        'value': basket_sizes.mean()
    }, {
        'metric': 'Median items per basket',
        'value': basket_sizes.median()
    }, {
        'metric': 'Most common basket size',
        'value': basket_sizes.mode().iloc[0]
    }, {
        'metric': 'Max items in a basket',
        'value': basket_sizes.max()
    }, {
        'metric': '% single-item baskets',
        'value': (basket_sizes == 1).mean() * 100
    }])
    
    return stats


def export_results(results: Dict[str, pd.DataFrame], output_path: Path):
    """Export all analysis results to an Excel workbook."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write each analysis to its own sheet
        for sheet_name, df in results.items():
            if df is not None and not df.empty:
                # Truncate sheet name if needed (Excel limit)
                safe_name = str(sheet_name)[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
    
    logger.info(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze October 2025 basket data for product bundling insights')
    parser.add_argument('--sales', required=False, help='Path to sales Excel file')
    parser.add_argument('--sku', required=False, help='Path to SKU master Excel file')
    parser.add_argument('--style', required=False, help='Path to STYLE master Excel file')
    parser.add_argument('--output', required=False, default='basket_analysis_oct2025.xlsx',
                       help='Output Excel path')
    parser.add_argument('--min-support', type=float, default=0.01,
                       help='Minimum support threshold for frequent itemsets')
    parser.add_argument('--min-confidence', type=float, default=0.1,
                       help='Minimum confidence threshold for rules')
    parser.add_argument('--min-lift', type=float, default=1.0,
                       help='Minimum lift threshold for rules')
    
    args = parser.parse_args()
    
    # Default paths if not provided
    sales_path = Path(args.sales) if args.sales else Path(r"D:\DATA TILL DATE\Desktop\EBO FOLDER\EBO SALES FOLDER\EBO SALES DATA.xlsx")
    sku_path = Path(args.sku) if args.sku else Path(r"D:\DATA TILL DATE\Desktop\EBO FOLDER\MASTERS\SKU MASTER.xlsx")
    style_path = Path(args.style) if args.style else Path(r"D:\DATA TILL DATE\Desktop\EBO FOLDER\MASTERS\STYLE MASTER.xlsx")
    
    logger.info("Loading data files...")
    sales_df = load_excel_any(sales_path)
    sku_df = load_excel_any(sku_path)
    style_df = load_excel_any(style_path)
    
    logger.info("Preparing October data...")
    oct_data = prepare_october_data(sales_df, sku_df, style_df)
    
    # Store results for export
    results = {}
    
    # Basic basket statistics
    logger.info("Computing basket statistics...")
    results['basket_stats'] = analyze_basket_sizes(oct_data)
    
    # Analyze at SKU level
    if 'SKU' in oct_data.columns:
        logger.info("Computing SKU-level associations...")
        sku_matrix = create_basket_matrix(oct_data, 'SKU')
        sku_itemsets, sku_rules = compute_association_rules(
            sku_matrix, 
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_lift=args.min_lift
        )
        
        # Add metadata to SKU rules
        sku_rules_meta = add_metadata_to_rules(
            sku_rules, 
            oct_data, 
            'SKU',
            ['STYLE', 'CATEGORY', 'DEPARTMENT']
        )
        
        results['sku_frequent_itemsets'] = sku_itemsets
        results['sku_association_rules'] = sku_rules_meta
        results['sku_common_pairs'] = analyze_common_pairs(oct_data, 'SKU')
    
    # Analyze at STYLE level
    if 'STYLE' in oct_data.columns:
        logger.info("Computing STYLE-level associations...")
        style_matrix = create_basket_matrix(oct_data, 'STYLE')
        style_itemsets, style_rules = compute_association_rules(
            style_matrix,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_lift=args.min_lift
        )
        
        # Add metadata to STYLE rules
        style_rules_meta = add_metadata_to_rules(
            style_rules,
            oct_data,
            'STYLE',
            ['CATEGORY', 'DEPARTMENT']
        )
        
        results['style_frequent_itemsets'] = style_itemsets
        results['style_association_rules'] = style_rules_meta
        results['style_common_pairs'] = analyze_common_pairs(oct_data, 'STYLE')
    
    # Export all results
    output_path = Path(args.output)
    logger.info("Exporting results...")
    export_results(results, output_path)
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()