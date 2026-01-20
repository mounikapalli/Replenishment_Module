"""
Simple EBO analysis with step-by-step debugging
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def simple_ebo_analysis():
    """
    Simple version of EBO analysis with debugging
    """
    
    print("🎯 EBO Store Style Efficiency Analysis (Simple Version)")
    print("=" * 60)
    
    # File paths
    file_paths = {
        'sales': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\EBO SALES FOLDER\EBO SALES DATA.xlsx',
        'stock': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\STOCK\EBo Stock Data.xlsx',
        'warehouse': r'D:\DATA TILL DATE\Downloads\Inventory Available for Sales - OMS-2025-10-31T12_38_35.558+05_30.csv',
    }
    
    print("📂 Loading EBO data...")
    
    # Load and process sales data
    try:
        sales_raw = pd.read_excel(file_paths['sales'])
        sales_df = pd.DataFrame({
            'DATE': pd.to_datetime(sales_raw['BILL_DATE'], errors='coerce'),
            'STORE': sales_raw['store_code'].astype(str).str.strip(),
            'STORE_NAME': sales_raw['EBO NAME'].astype(str).str.strip(),
            'SKU': sales_raw['SKU'].astype(str).str.strip(),
            'QUANTITY': pd.to_numeric(sales_raw['BILL_QUANTITY'], errors='coerce').fillna(0),
            'STYLE': sales_raw['STYLE'].astype(str).str.strip(),
            'COLOR': sales_raw['COLOR'].astype(str).str.strip(),
            'SIZE': sales_raw['SIZE'].astype(str).str.strip(),
            'DEPARTMENT': sales_raw['DEPARTMENT'].astype(str).str.strip(),
            'NET_AMOUNT': pd.to_numeric(sales_raw['NET_AMOUNT'], errors='coerce').fillna(0)
        })
        
        # Clean data
        sales_df = sales_df.dropna(subset=['DATE'])
        sales_df = sales_df[sales_df['QUANTITY'] > 0]
        sales_df = sales_df[sales_df['SKU'] != '']
        sales_df = sales_df[sales_df['STYLE'] != '']
        sales_df = sales_df[sales_df['STYLE'] != 'nan']
        
        print(f"✅ Sales data: {len(sales_df):,} records")
        
    except Exception as e:
        print(f"❌ Error loading sales data: {str(e)}")
        return
    
    # Load stock data
    try:
        stock_raw = pd.read_excel(file_paths['stock'])
        stock_df = pd.DataFrame({
            'STORE': stock_raw['store_code'].astype(str).str.strip(),
            'SKU': stock_raw['SKU'].astype(str).str.strip(),
            'STOCK': pd.to_numeric(stock_raw['quantity'], errors='coerce').fillna(0),
            'STYLE': stock_raw['Style'].astype(str).str.strip()
        })
        
        stock_df = stock_df[stock_df['SKU'] != '']
        stock_df = stock_df[stock_df['STYLE'] != '']
        
        print(f"✅ Stock data: {len(stock_df):,} records")
        
    except Exception as e:
        print(f"❌ Error loading stock data: {str(e)}")
        return
    
    print(f"\n📊 Data Overview:")
    print(f"   • Sales Period: {sales_df['DATE'].min()} to {sales_df['DATE'].max()}")
    print(f"   • Total Stores: {sales_df['STORE'].nunique()}")
    print(f"   • Total Styles: {sales_df['STYLE'].nunique()}")
    print(f"   • Total Sales Quantity: {sales_df['QUANTITY'].sum():,.0f}")
    print(f"   • Total Sales Value: ₹{sales_df['NET_AMOUNT'].sum():,.0f}")
    
    # Recent data analysis (last 90 days)
    end_date = sales_df['DATE'].max()
    start_date = end_date - timedelta(days=90)
    recent_sales = sales_df[sales_df['DATE'] >= start_date].copy()
    
    print(f"\n🕒 Last 90 Days Analysis ({start_date.date()} to {end_date.date()}):")
    print(f"   • Records: {len(recent_sales):,}")
    print(f"   • Sales Quantity: {recent_sales['QUANTITY'].sum():,.0f}")
    print(f"   • Sales Value: ₹{recent_sales['NET_AMOUNT'].sum():,.0f}")
    
    # Calculate style performance by store
    print(f"\n📈 Calculating Style Performance by Store...")
    
    # Store-Style level analysis
    store_style_perf = recent_sales.groupby(['STORE', 'STORE_NAME', 'STYLE']).agg({
        'QUANTITY': 'sum',
        'NET_AMOUNT': 'sum',
        'DATE': 'nunique'  # Number of days with sales
    }).reset_index()
    
    store_style_perf.columns = ['STORE', 'STORE_NAME', 'STYLE', 'TOTAL_QTY', 'TOTAL_REVENUE', 'SALES_DAYS']
    
    # Calculate performance metrics
    store_style_perf['DAILY_AVG_QTY'] = store_style_perf['TOTAL_QTY'] / 90  # Daily average
    store_style_perf['REVENUE_PER_QTY'] = store_style_perf['TOTAL_REVENUE'] / store_style_perf['TOTAL_QTY']
    store_style_perf['SALES_FREQUENCY'] = store_style_perf['SALES_DAYS'] / 90  # Frequency of sales
    
    print(f"✅ Style performance calculated for {len(store_style_perf)} store-style combinations")
    
    # Add stock information
    print(f"📦 Adding current stock information...")
    
    store_stock = stock_df.groupby(['STORE', 'STYLE']).agg({
        'STOCK': 'sum'
    }).reset_index()
    
    # Merge with performance data
    store_style_analysis = store_style_perf.merge(
        store_stock, 
        on=['STORE', 'STYLE'], 
        how='left'
    )
    
    store_style_analysis['STOCK'] = store_style_analysis['STOCK'].fillna(0)
    
    # Calculate efficiency scores
    print(f"🎯 Calculating Efficiency Scores...")
    
    # Normalize metrics (0-100 scale)
    def normalize_metric(series, higher_is_better=True):
        if series.max() == series.min():
            return pd.Series([50] * len(series), index=series.index)
        
        if higher_is_better:
            return ((series - series.min()) / (series.max() - series.min())) * 100
        else:
            return ((series.max() - series) / (series.max() - series.min())) * 100
    
    # Calculate normalized scores for each metric
    store_style_analysis['SALES_VELOCITY_SCORE'] = normalize_metric(store_style_analysis['DAILY_AVG_QTY'])
    store_style_analysis['REVENUE_SCORE'] = normalize_metric(store_style_analysis['TOTAL_REVENUE'])
    store_style_analysis['FREQUENCY_SCORE'] = normalize_metric(store_style_analysis['SALES_FREQUENCY'])
    
    # Calculate stock efficiency (lower stock relative to sales is better)
    store_style_analysis['STOCK_DAYS'] = store_style_analysis['STOCK'] / (store_style_analysis['DAILY_AVG_QTY'] + 0.1)
    store_style_analysis['STOCK_EFFICIENCY_SCORE'] = normalize_metric(store_style_analysis['STOCK_DAYS'], higher_is_better=False)
    
    # Overall efficiency score (weighted)
    store_style_analysis['EFFICIENCY_SCORE'] = (
        store_style_analysis['SALES_VELOCITY_SCORE'] * 0.30 +
        store_style_analysis['REVENUE_SCORE'] * 0.25 +
        store_style_analysis['FREQUENCY_SCORE'] * 0.25 +
        store_style_analysis['STOCK_EFFICIENCY_SCORE'] * 0.20
    )
    
    print(f"✅ Efficiency scores calculated")
    
    # Store-level recommendations
    print(f"🏪 Generating Store-level Recommendations...")
    
    store_summary = store_style_analysis.groupby(['STORE', 'STORE_NAME']).agg({
        'STYLE': 'count',
        'EFFICIENCY_SCORE': 'mean',
        'TOTAL_QTY': 'sum',
        'TOTAL_REVENUE': 'sum'
    }).reset_index()
    
    store_summary.columns = ['STORE', 'STORE_NAME', 'CURRENT_STYLES', 'AVG_EFFICIENCY', 'TOTAL_QTY', 'TOTAL_REVENUE']
    
    # Classify styles by performance
    high_performers = store_style_analysis[store_style_analysis['EFFICIENCY_SCORE'] >= 70]
    medium_performers = store_style_analysis[(store_style_analysis['EFFICIENCY_SCORE'] >= 50) & (store_style_analysis['EFFICIENCY_SCORE'] < 70)]
    low_performers = store_style_analysis[store_style_analysis['EFFICIENCY_SCORE'] < 50]
    
    store_recommendations = []
    
    for _, store in store_summary.iterrows():
        store_code = store['STORE']
        store_name = store['STORE_NAME']
        current_styles = store['CURRENT_STYLES']
        
        store_high = len(high_performers[high_performers['STORE'] == store_code])
        store_medium = len(medium_performers[medium_performers['STORE'] == store_code])
        store_low = len(low_performers[low_performers['STORE'] == store_code])
        
        # Recommendation logic
        if store['AVG_EFFICIENCY'] >= 70:
            # High performing store - maintain or slightly expand
            recommended_styles = max(store_high + int(store_medium * 0.5), current_styles)
            category = "High Performer"
            strategy = "Maintain high performers, selective expansion"
        elif store['AVG_EFFICIENCY'] >= 50:
            # Medium performing store - focus on high performers
            recommended_styles = store_high + int(store_medium * 0.7)
            category = "Medium Performer"
            strategy = "Focus on proven styles, reduce underperformers"
        else:
            # Low performing store - significant reduction needed
            recommended_styles = max(store_high + int(store_medium * 0.3), int(current_styles * 0.5))
            category = "Needs Improvement"
            strategy = "Major optimization needed, focus on top performers"
        
        store_recommendations.append({
            'STORE': store_code,
            'STORE_NAME': store_name,
            'CURRENT_STYLES': current_styles,
            'RECOMMENDED_STYLES': recommended_styles,
            'CHANGE': recommended_styles - current_styles,
            'CHANGE_PCT': ((recommended_styles - current_styles) / current_styles) * 100,
            'EFFICIENCY_SCORE': store['AVG_EFFICIENCY'],
            'CATEGORY': category,
            'STRATEGY': strategy,
            'HIGH_PERFORMERS': store_high,
            'MEDIUM_PERFORMERS': store_medium,
            'LOW_PERFORMERS': store_low,
            'TOTAL_SALES_QTY': store['TOTAL_QTY'],
            'TOTAL_REVENUE': store['TOTAL_REVENUE']
        })
    
    recommendations_df = pd.DataFrame(store_recommendations)
    recommendations_df = recommendations_df.sort_values('EFFICIENCY_SCORE', ascending=False)
    
    print(f"✅ Recommendations generated for {len(recommendations_df)} stores")
    
    # Export to Excel
    print(f"\n💾 Exporting results to Excel...")
    
    output_file = f"EBO_Style_Efficiency_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main recommendations
            recommendations_df.to_excel(writer, sheet_name='Store_Recommendations', index=False)
            
            # Detailed style performance
            store_style_analysis_export = store_style_analysis[
                ['STORE', 'STORE_NAME', 'STYLE', 'TOTAL_QTY', 'TOTAL_REVENUE', 
                 'DAILY_AVG_QTY', 'SALES_FREQUENCY', 'STOCK', 'EFFICIENCY_SCORE']
            ].sort_values(['STORE', 'EFFICIENCY_SCORE'], ascending=[True, False])
            
            store_style_analysis_export.to_excel(writer, sheet_name='Style_Performance', index=False)
            
            # Top performing styles overall
            top_styles = store_style_analysis.nlargest(50, 'EFFICIENCY_SCORE')[
                ['STYLE', 'STORE', 'STORE_NAME', 'TOTAL_QTY', 'TOTAL_REVENUE', 'EFFICIENCY_SCORE']
            ]
            top_styles.to_excel(writer, sheet_name='Top_Performing_Styles', index=False)
            
            # Summary statistics
            summary_stats = pd.DataFrame({
                'Metric': [
                    'Total Stores Analyzed',
                    'Total Styles Analyzed',
                    'Average Styles per Store',
                    'Average Efficiency Score',
                    'High Performing Stores (70+)',
                    'Medium Performing Stores (50-70)',
                    'Low Performing Stores (<50)',
                    'Total Sales Quantity (90 days)',
                    'Total Sales Revenue (90 days)',
                    'Recommended Style Reduction',
                    'Stores Needing Reduction'
                ],
                'Value': [
                    len(recommendations_df),
                    store_style_analysis['STYLE'].nunique(),
                    recommendations_df['CURRENT_STYLES'].mean(),
                    recommendations_df['EFFICIENCY_SCORE'].mean(),
                    len(recommendations_df[recommendations_df['CATEGORY'] == 'High Performer']),
                    len(recommendations_df[recommendations_df['CATEGORY'] == 'Medium Performer']),
                    len(recommendations_df[recommendations_df['CATEGORY'] == 'Needs Improvement']),
                    recommendations_df['TOTAL_SALES_QTY'].sum(),
                    recommendations_df['TOTAL_REVENUE'].sum(),
                    recommendations_df['CHANGE'].sum(),
                    len(recommendations_df[recommendations_df['CHANGE'] < 0])
                ]
            })
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        print(f"✅ Excel file created: {output_file}")
        
        # Print summary results
        print(f"\n" + "="*60)
        print(f"📈 EBO STORE STYLE EFFICIENCY ANALYSIS RESULTS")
        print(f"="*60)
        
        print(f"\n📊 Overall Summary:")
        print(f"   • Total Stores Analyzed: {len(recommendations_df)}")
        print(f"   • Average Current Styles per Store: {recommendations_df['CURRENT_STYLES'].mean():.1f}")
        print(f"   • Average Recommended Styles per Store: {recommendations_df['RECOMMENDED_STYLES'].mean():.1f}")
        print(f"   • Overall Efficiency Score: {recommendations_df['EFFICIENCY_SCORE'].mean():.1f}/100")
        
        total_reduction = recommendations_df['CHANGE'].sum()
        if total_reduction < 0:
            print(f"   • Total Style Reduction Recommended: {abs(total_reduction)} styles")
        else:
            print(f"   • Total Style Increase Recommended: {total_reduction} styles")
        
        print(f"\n🏪 Store Performance Distribution:")
        for category in ['High Performer', 'Medium Performer', 'Needs Improvement']:
            count = len(recommendations_df[recommendations_df['CATEGORY'] == category])
            print(f"   • {category}: {count} stores")
        
        print(f"\n🎯 Top 5 Performing Stores:")
        for _, store in recommendations_df.head(5).iterrows():
            print(f"   • {store['STORE']} - {store['STORE_NAME']}: {store['EFFICIENCY_SCORE']:.1f} score")
        
        print(f"\n⚠️ Stores Needing Most Attention:")
        need_attention = recommendations_df[recommendations_df['CHANGE'] < -5].sort_values('CHANGE')
        for _, store in need_attention.head(5).iterrows():
            print(f"   • {store['STORE']} - {store['STORE_NAME']}: {store['CHANGE']:.0f} styles reduction")
        
        print(f"\n💰 Revenue Impact (Last 90 Days):")
        print(f"   • Total Sales Revenue: ₹{recommendations_df['TOTAL_REVENUE'].sum():,.0f}")
        print(f"   • Average Revenue per Store: ₹{recommendations_df['TOTAL_REVENUE'].mean():,.0f}")
        
        print(f"\n✨ Analysis complete! Check the Excel file for detailed results.")
        
    except Exception as e:
        print(f"❌ Error creating Excel file: {str(e)}")
        return None
    
    return output_file

if __name__ == "__main__":
    simple_ebo_analysis()