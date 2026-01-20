"""
Optimal Style Count Analysis for EBO Stores
Determines the ideal number of styles per store based on sales, stock, and warehouse data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def determine_optimal_styles_per_store():
    """
    Main function to determine optimal number of styles per store
    Based on sales velocity, stock efficiency, and warehouse availability
    """
    
    print("🎯 OPTIMAL STYLE COUNT ANALYSIS FOR EBO STORES")
    print("=" * 60)
    print("Goal: Determine ideal number of styles per store based on:")
    print("  • Sales performance and velocity")
    print("  • Current stock levels and turnover")
    print("  • Warehouse stock availability")
    print("=" * 60)
    
    # File paths
    file_paths = {
        'sales': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\EBO SALES FOLDER\EBO SALES DATA.xlsx',
        'stock': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\STOCK\EBo Stock Data.xlsx',
        'warehouse': r'D:\DATA TILL DATE\Downloads\Inventory Available for Sales - OMS-2025-10-31T12_38_35.558+05_30.csv',
    }
    
    print("📂 Loading data...")
    
    # Load Sales Data
    sales_raw = pd.read_excel(file_paths['sales'])
    sales_df = pd.DataFrame({
        'DATE': pd.to_datetime(sales_raw['BILL_DATE'], errors='coerce'),
        'STORE': sales_raw['store_code'].astype(str).str.strip(),
        'STORE_NAME': sales_raw['EBO NAME'].astype(str).str.strip(),
        'SKU': sales_raw['SKU'].astype(str).str.strip(),
        'QUANTITY': pd.to_numeric(sales_raw['BILL_QUANTITY'], errors='coerce').fillna(0),
        'STYLE': sales_raw['STYLE'].astype(str).str.strip(),
        'NET_AMOUNT': pd.to_numeric(sales_raw['NET_AMOUNT'], errors='coerce').fillna(0),
        'DEPARTMENT': sales_raw['DEPARTMENT'].astype(str).str.strip()
    })
    
    # Clean sales data
    sales_df = sales_df.dropna(subset=['DATE'])
    sales_df = sales_df[sales_df['QUANTITY'] > 0]
    sales_df = sales_df[sales_df['SKU'] != '']
    sales_df = sales_df[sales_df['STYLE'] != '']
    sales_df = sales_df[sales_df['STYLE'] != 'nan']
    
    # Load Stock Data
    stock_raw = pd.read_excel(file_paths['stock'])
    stock_df = pd.DataFrame({
        'STORE': stock_raw['store_code'].astype(str).str.strip(),
        'SKU': stock_raw['SKU'].astype(str).str.strip(),
        'STOCK_QTY': pd.to_numeric(stock_raw['quantity'], errors='coerce').fillna(0),
        'STYLE': stock_raw['Style'].astype(str).str.strip(),
        'DEPARTMENT': stock_raw['Department'].astype(str).str.strip()
    })
    
    # Load Warehouse Data
    warehouse_raw = pd.read_csv(file_paths['warehouse'])
    warehouse_df = pd.DataFrame({
        'SKU': warehouse_raw['Each Client SKU ID'].astype(str).str.strip(),
        'WAREHOUSE_STOCK': pd.to_numeric(warehouse_raw['Total Available Quantity'], errors='coerce').fillna(0),
        'STYLE': warehouse_raw['Style'].astype(str).str.strip()
    })
    
    print(f"✅ Data loaded successfully")
    print(f"   Sales: {len(sales_df):,} records")
    print(f"   Stock: {len(stock_df):,} records") 
    print(f"   Warehouse: {len(warehouse_df):,} records")
    
    # Analysis period - last 90 days
    end_date = sales_df['DATE'].max()
    start_date = end_date - timedelta(days=90)
    recent_sales = sales_df[sales_df['DATE'] >= start_date].copy()
    
    print(f"\n📅 Analysis Period: {start_date.date()} to {end_date.date()} (90 days)")
    print(f"   Recent sales records: {len(recent_sales):,}")
    
    # Step 1: Analyze Style Performance by Store
    print(f"\n📊 STEP 1: Analyzing Style Performance by Store")
    
    # Calculate style-level metrics for each store
    style_performance = recent_sales.groupby(['STORE', 'STORE_NAME', 'STYLE']).agg({
        'QUANTITY': ['sum', 'count'],  # Total qty and transaction count
        'NET_AMOUNT': 'sum',
        'DATE': 'nunique'  # Number of days with sales
    }).reset_index()
    
    # Flatten column names
    style_performance.columns = ['STORE', 'STORE_NAME', 'STYLE', 'TOTAL_QTY', 'TRANSACTION_COUNT', 'TOTAL_REVENUE', 'SALES_DAYS']
    
    # Calculate performance metrics
    style_performance['DAILY_AVG_QTY'] = style_performance['TOTAL_QTY'] / 90
    style_performance['SALES_FREQUENCY'] = style_performance['SALES_DAYS'] / 90
    style_performance['AVG_TRANSACTION_SIZE'] = style_performance['TOTAL_QTY'] / style_performance['TRANSACTION_COUNT']
    style_performance['REVENUE_PER_UNIT'] = style_performance['TOTAL_REVENUE'] / style_performance['TOTAL_QTY']
    
    print(f"   ✅ Style performance calculated for {len(style_performance)} store-style combinations")
    
    # Step 2: Add Stock Information
    print(f"\n📦 STEP 2: Adding Current Stock Analysis")
    
    # Aggregate stock by store and style
    stock_summary = stock_df.groupby(['STORE', 'STYLE']).agg({
        'STOCK_QTY': 'sum'
    }).reset_index()
    
    # Merge stock with performance
    style_analysis = style_performance.merge(
        stock_summary,
        on=['STORE', 'STYLE'],
        how='left'
    )
    
    style_analysis['STOCK_QTY'] = style_analysis['STOCK_QTY'].fillna(0)
    
    # Calculate stock metrics
    style_analysis['DAYS_OF_STOCK'] = style_analysis['STOCK_QTY'] / (style_analysis['DAILY_AVG_QTY'] + 0.01)
    style_analysis['STOCK_TURNOVER'] = style_analysis['TOTAL_QTY'] / (style_analysis['STOCK_QTY'] + 1)
    
    print(f"   ✅ Stock analysis added")
    
    # Step 3: Add Warehouse Stock
    print(f"\n🏭 STEP 3: Adding Warehouse Stock Analysis")
    
    # Aggregate warehouse stock by style
    warehouse_summary = warehouse_df.groupby('STYLE').agg({
        'WAREHOUSE_STOCK': 'sum'
    }).reset_index()
    
    # Merge warehouse data
    style_analysis = style_analysis.merge(
        warehouse_summary,
        on='STYLE',
        how='left'
    )
    
    style_analysis['WAREHOUSE_STOCK'] = style_analysis['WAREHOUSE_STOCK'].fillna(0)
    
    # Calculate availability metrics
    style_analysis['TOTAL_AVAILABLE'] = style_analysis['STOCK_QTY'] + style_analysis['WAREHOUSE_STOCK']
    style_analysis['WAREHOUSE_COVERAGE_DAYS'] = style_analysis['WAREHOUSE_STOCK'] / (style_analysis['DAILY_AVG_QTY'] + 0.01)
    
    print(f"   ✅ Warehouse analysis added")
    
    # Step 4: Calculate Style Efficiency Scores
    print(f"\n🎯 STEP 4: Calculating Style Efficiency Scores")
    
    def calculate_percentile_score(series, higher_is_better=True):
        """Convert values to 0-100 percentile scores"""
        if len(series.unique()) == 1:
            return pd.Series([50] * len(series), index=series.index)
        
        if higher_is_better:
            return series.rank(pct=True) * 100
        else:
            return (1 - series.rank(pct=True)) * 100
    
    # Calculate individual metric scores
    style_analysis['SALES_VELOCITY_SCORE'] = calculate_percentile_score(style_analysis['DAILY_AVG_QTY'])
    style_analysis['FREQUENCY_SCORE'] = calculate_percentile_score(style_analysis['SALES_FREQUENCY'])
    style_analysis['REVENUE_SCORE'] = calculate_percentile_score(style_analysis['TOTAL_REVENUE'])
    style_analysis['TURNOVER_SCORE'] = calculate_percentile_score(style_analysis['STOCK_TURNOVER'])
    style_analysis['AVAILABILITY_SCORE'] = calculate_percentile_score(style_analysis['TOTAL_AVAILABLE'])
    
    # Overall efficiency score (weighted)
    style_analysis['EFFICIENCY_SCORE'] = (
        style_analysis['SALES_VELOCITY_SCORE'] * 0.30 +     # Sales velocity - most important
        style_analysis['FREQUENCY_SCORE'] * 0.25 +          # Consistency of sales
        style_analysis['REVENUE_SCORE'] * 0.20 +            # Revenue contribution
        style_analysis['TURNOVER_SCORE'] * 0.15 +           # Stock efficiency
        style_analysis['AVAILABILITY_SCORE'] * 0.10         # Stock availability
    )
    
    print(f"   ✅ Efficiency scores calculated")
    
    # Step 5: Determine Optimal Style Count per Store
    print(f"\n🏪 STEP 5: Determining Optimal Style Count per Store")
    
    # Classify styles by performance tiers
    def classify_style_performance(score):
        if score >= 80:
            return 'Star Performer'      # Top 20% - Must keep
        elif score >= 60:
            return 'Strong Performer'    # Top 20-40% - Keep
        elif score >= 40:
            return 'Average Performer'   # Middle 40-60% - Selective keep
        elif score >= 20:
            return 'Weak Performer'      # Bottom 20-40% - Consider removing
        else:
            return 'Poor Performer'      # Bottom 20% - Remove
    
    style_analysis['PERFORMANCE_TIER'] = style_analysis['EFFICIENCY_SCORE'].apply(classify_style_performance)
    
    # Store-level analysis
    store_analysis = []
    
    for store_code in style_analysis['STORE'].unique():
        store_data = style_analysis[style_analysis['STORE'] == store_code].copy()
        store_name = store_data['STORE_NAME'].iloc[0]
        
        # Current situation
        current_styles = len(store_data)
        total_revenue = store_data['TOTAL_REVENUE'].sum()
        total_quantity = store_data['TOTAL_QTY'].sum()
        avg_efficiency = store_data['EFFICIENCY_SCORE'].mean()
        
        # Count by performance tiers
        tier_counts = store_data['PERFORMANCE_TIER'].value_counts()
        
        star_performers = tier_counts.get('Star Performer', 0)
        strong_performers = tier_counts.get('Strong Performer', 0)
        average_performers = tier_counts.get('Average Performer', 0)
        weak_performers = tier_counts.get('Weak Performer', 0)
        poor_performers = tier_counts.get('Poor Performer', 0)
        
        # Determine optimal style count based on performance distribution
        
        # Strategy 1: Keep all star and strong performers + selective average
        base_styles = star_performers + strong_performers
        
        # Add average performers based on store performance
        if avg_efficiency >= 50:  # High performing store
            additional_average = int(average_performers * 0.7)  # Keep 70% of average
        elif avg_efficiency >= 30:  # Medium performing store  
            additional_average = int(average_performers * 0.5)  # Keep 50% of average
        else:  # Low performing store
            additional_average = int(average_performers * 0.3)  # Keep 30% of average
        
        optimal_styles = base_styles + additional_average
        
        # Ensure minimum viable range (20-150 styles)
        optimal_styles = max(20, min(150, optimal_styles))
        
        # Calculate impact metrics
        style_reduction = current_styles - optimal_styles
        reduction_percentage = (style_reduction / current_styles) * 100 if current_styles > 0 else 0
        
        # Estimate revenue retention (assuming we keep top performing styles)
        top_styles = store_data.nlargest(optimal_styles, 'EFFICIENCY_SCORE')
        retained_revenue = top_styles['TOTAL_REVENUE'].sum()
        revenue_retention = (retained_revenue / total_revenue) * 100 if total_revenue > 0 else 0
        
        # Determine strategy category
        if avg_efficiency >= 50:
            strategy_category = "Optimize & Expand"
            strategy_description = "High efficiency - optimize portfolio and consider expansion"
        elif avg_efficiency >= 30:
            strategy_category = "Focus & Improve"
            strategy_description = "Medium efficiency - focus on top performers"
        else:
            strategy_category = "Radical Optimization"
            strategy_description = "Low efficiency - significant portfolio reduction needed"
        
        store_analysis.append({
            'STORE': store_code,
            'STORE_NAME': store_name,
            'CURRENT_STYLES': current_styles,
            'OPTIMAL_STYLES': optimal_styles,
            'STYLE_REDUCTION': style_reduction,
            'REDUCTION_PERCENTAGE': reduction_percentage,
            'EFFICIENCY_SCORE': avg_efficiency,
            'STRATEGY_CATEGORY': strategy_category,
            'STRATEGY_DESCRIPTION': strategy_description,
            'TOTAL_REVENUE': total_revenue,
            'ESTIMATED_REVENUE_RETENTION': revenue_retention,
            'STAR_PERFORMERS': star_performers,
            'STRONG_PERFORMERS': strong_performers,
            'AVERAGE_PERFORMERS': average_performers,
            'WEAK_PERFORMERS': weak_performers,
            'POOR_PERFORMERS': poor_performers
        })
    
    store_recommendations = pd.DataFrame(store_analysis)
    store_recommendations = store_recommendations.sort_values('EFFICIENCY_SCORE', ascending=False)
    
    print(f"   ✅ Optimal style count determined for {len(store_recommendations)} stores")
    
    # Step 6: Generate Summary and Export
    print(f"\n💾 STEP 6: Generating Results and Export")
    
    # Calculate overall summary
    total_current_styles = store_recommendations['CURRENT_STYLES'].sum()
    total_optimal_styles = store_recommendations['OPTIMAL_STYLES'].sum()
    total_reduction = total_current_styles - total_optimal_styles
    avg_reduction_percentage = store_recommendations['REDUCTION_PERCENTAGE'].mean()
    avg_revenue_retention = store_recommendations['ESTIMATED_REVENUE_RETENTION'].mean()
    
    # Export to Excel
    output_file = f"Optimal_Style_Count_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main recommendations
        store_recommendations.to_excel(writer, sheet_name='Store_Recommendations', index=False)
        
        # Detailed style analysis
        style_export = style_analysis[[
            'STORE', 'STORE_NAME', 'STYLE', 'TOTAL_QTY', 'TOTAL_REVENUE',
            'DAILY_AVG_QTY', 'SALES_FREQUENCY', 'STOCK_QTY', 'WAREHOUSE_STOCK',
            'EFFICIENCY_SCORE', 'PERFORMANCE_TIER'
        ]].sort_values(['STORE', 'EFFICIENCY_SCORE'], ascending=[True, False])
        
        style_export.to_excel(writer, sheet_name='Style_Analysis', index=False)
        
        # Performance tier summary
        tier_summary = style_analysis.groupby(['STORE', 'STORE_NAME', 'PERFORMANCE_TIER']).size().unstack(fill_value=0)
        tier_summary.to_excel(writer, sheet_name='Performance_Tiers')
        
        # Top styles to keep (recommended portfolio)
        recommended_styles = []
        for store_code in store_recommendations['STORE']:
            store_data = style_analysis[style_analysis['STORE'] == store_code]
            optimal_count = store_recommendations[store_recommendations['STORE'] == store_code]['OPTIMAL_STYLES'].iloc[0]
            top_styles = store_data.nlargest(optimal_count, 'EFFICIENCY_SCORE')
            recommended_styles.append(top_styles)
        
        recommended_portfolio = pd.concat(recommended_styles)
        recommended_portfolio.to_excel(writer, sheet_name='Recommended_Portfolio', index=False)
        
        # Summary statistics
        summary_data = {
            'Metric': [
                'Total Stores Analyzed',
                'Current Total Styles',
                'Optimal Total Styles', 
                'Total Style Reduction',
                'Average Reduction %',
                'Average Revenue Retention %',
                'Stores Needing Reduction',
                'Stores Ready for Expansion',
                'High Efficiency Stores (50+)',
                'Medium Efficiency Stores (30-50)',
                'Low Efficiency Stores (<30)'
            ],
            'Value': [
                len(store_recommendations),
                total_current_styles,
                total_optimal_styles,
                total_reduction,
                f"{avg_reduction_percentage:.1f}%",
                f"{avg_revenue_retention:.1f}%",
                len(store_recommendations[store_recommendations['STYLE_REDUCTION'] > 0]),
                len(store_recommendations[store_recommendations['STYLE_REDUCTION'] < 0]),
                len(store_recommendations[store_recommendations['EFFICIENCY_SCORE'] >= 50]),
                len(store_recommendations[(store_recommendations['EFFICIENCY_SCORE'] >= 30) & (store_recommendations['EFFICIENCY_SCORE'] < 50)]),
                len(store_recommendations[store_recommendations['EFFICIENCY_SCORE'] < 30])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Results exported to: {output_file}")
    
    # Display Results
    print(f"\n" + "="*70)
    print(f"🎯 OPTIMAL STYLE COUNT ANALYSIS RESULTS")
    print(f"="*70)
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"   • Total Stores Analyzed: {len(store_recommendations)}")
    print(f"   • Current Average Styles per Store: {store_recommendations['CURRENT_STYLES'].mean():.1f}")
    print(f"   • Optimal Average Styles per Store: {store_recommendations['OPTIMAL_STYLES'].mean():.1f}")
    print(f"   • Total Style Reduction Needed: {total_reduction:,} styles")
    print(f"   • Average Reduction per Store: {avg_reduction_percentage:.1f}%")
    print(f"   • Expected Revenue Retention: {avg_revenue_retention:.1f}%")
    
    print(f"\n🏆 TOP 5 STORES (By Efficiency Score):")
    for _, store in store_recommendations.head(5).iterrows():
        print(f"   • {store['STORE']} - {store['STORE_NAME']}")
        print(f"     Current: {store['CURRENT_STYLES']} → Optimal: {store['OPTIMAL_STYLES']} styles")
        print(f"     Efficiency: {store['EFFICIENCY_SCORE']:.1f} | Strategy: {store['STRATEGY_CATEGORY']}")
        print()
    
    print(f"⚠️ STORES NEEDING MOST REDUCTION:")
    need_reduction = store_recommendations[store_recommendations['STYLE_REDUCTION'] > 20].nlargest(5, 'STYLE_REDUCTION')
    for _, store in need_reduction.iterrows():
        print(f"   • {store['STORE']} - {store['STORE_NAME']}")
        print(f"     Reduce: {store['STYLE_REDUCTION']} styles ({store['REDUCTION_PERCENTAGE']:.1f}%)")
        print(f"     Current: {store['CURRENT_STYLES']} → Optimal: {store['OPTIMAL_STYLES']}")
        print()
    
    print(f"📈 STRATEGY DISTRIBUTION:")
    strategy_counts = store_recommendations['STRATEGY_CATEGORY'].value_counts()
    for strategy, count in strategy_counts.items():
        print(f"   • {strategy}: {count} stores")
    
    print(f"\n✅ Analysis Complete!")
    print(f"📊 Check '{output_file}' for detailed store-by-store recommendations")
    print(f"🎯 This analysis provides the exact number of styles each store should carry")
    print(f"   based on sales performance, stock efficiency, and warehouse availability")
    
    return output_file

if __name__ == "__main__":
    determine_optimal_styles_per_store()