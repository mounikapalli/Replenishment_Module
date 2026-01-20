"""
Test the Store Style Efficiency Analysis with sample data
"""

import pandas as pd
import numpy as np
from store_style_efficiency import StoreStyleEfficiencyAnalyzer
import warnings
warnings.filterwarnings('ignore')

def load_sample_data():
    """Load the generated sample data"""
    try:
        sales_df = pd.read_csv('sample_sales_data.csv')
        stock_df = pd.read_csv('sample_stock_data.csv')
        warehouse_df = pd.read_csv('sample_warehouse_data.csv')
        sku_master_df = pd.read_csv('sample_sku_master.csv')
        style_master_df = pd.read_csv('sample_style_master.csv')
        
        # Convert date column to datetime
        sales_df['DATE'] = pd.to_datetime(sales_df['DATE'])
        
        print("✅ Sample data loaded successfully!")
        return sales_df, stock_df, warehouse_df, sku_master_df, style_master_df
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return None, None, None, None, None

def test_store_style_efficiency():
    """Test the Store Style Efficiency Analysis"""
    
    print("🚀 Testing Store Style Efficiency Analysis...")
    print("=" * 60)
    
    # Load sample data
    sales_df, stock_df, warehouse_df, sku_master_df, style_master_df = load_sample_data()
    
    if any(df is None for df in [sales_df, stock_df, warehouse_df, sku_master_df, style_master_df]):
        print("❌ Failed to load sample data. Please run generate_sample_data.py first.")
        return
    
    # Initialize the analyzer
    analyzer = StoreStyleEfficiencyAnalyzer()
    
    # Load data into analyzer
    analyzer.load_data(
        sales_df, stock_df, warehouse_df, sku_master_df, style_master_df
    )
    
    print("\n📊 STEP 1: Calculating Style Performance Metrics...")
    print("-" * 50)
    
    # Calculate style performance metrics
    style_metrics = analyzer.calculate_style_performance_metrics(time_period_days=90)
    
    if style_metrics.empty:
        print("❌ Failed to calculate style metrics")
        return
    
    print(f"✅ Calculated metrics for {len(style_metrics)} store-style combinations")
    print(f"📈 Average Efficiency Score: {style_metrics['STYLE_EFFICIENCY_SCORE'].mean():.1f}")
    print(f"🎯 Styles with Score ≥60: {len(style_metrics[style_metrics['STYLE_EFFICIENCY_SCORE'] >= 60])}")
    
    # Show top performing styles
    print(f"\n🌟 TOP 10 PERFORMING STYLES:")
    top_styles = style_metrics.nlargest(10, 'STYLE_EFFICIENCY_SCORE')[
        ['STORE', 'STYLE', 'STYLE_EFFICIENCY_SCORE', 'SALES_VELOCITY', 'STOCK_TURNOVER']
    ]
    print(top_styles.to_string(index=False))
    
    print(f"\n📊 STEP 2: Determining Optimal Styles per Store...")
    print("-" * 50)
    
    # Determine optimal styles per store
    store_recommendations = analyzer.determine_optimal_styles_per_store(
        style_metrics, efficiency_threshold=60
    )
    
    if store_recommendations.empty:
        print("❌ Failed to generate store recommendations")
        return
    
    print(f"✅ Generated recommendations for {len(store_recommendations)} stores")
    
    # Display store recommendations
    print(f"\n🏪 STORE RECOMMENDATIONS SUMMARY:")
    print("-" * 80)
    
    display_cols = [
        'STORE', 'STORE_CATEGORY', 'CURRENT_TOTAL_STYLES', 
        'CURRENT_EFFICIENT_STYLES', 'RECOMMENDED_STYLES', 
        'EFFICIENCY_IMPROVEMENT', 'AVG_EFFICIENCY_SCORE'
    ]
    
    recommendations_display = store_recommendations[display_cols].copy()
    recommendations_display['AVG_EFFICIENCY_SCORE'] = recommendations_display['AVG_EFFICIENCY_SCORE'].round(1)
    
    print(recommendations_display.to_string(index=False))
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print("-" * 30)
    print(f"Total Stores Analyzed: {len(store_recommendations)}")
    print(f"Average Current Styles per Store: {store_recommendations['CURRENT_TOTAL_STYLES'].mean():.1f}")
    print(f"Average Recommended Styles per Store: {store_recommendations['RECOMMENDED_STYLES'].mean():.1f}")
    print(f"Average Efficiency Score: {store_recommendations['AVG_EFFICIENCY_SCORE'].mean():.1f}")
    
    # Store category breakdown
    category_counts = store_recommendations['STORE_CATEGORY'].value_counts()
    print(f"\n🏪 STORE PERFORMANCE CATEGORIES:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} stores")
    
    # Impact analysis
    total_current_styles = store_recommendations['CURRENT_TOTAL_STYLES'].sum()
    total_recommended_styles = store_recommendations['RECOMMENDED_STYLES'].sum()
    style_reduction = total_current_styles - total_recommended_styles
    
    print(f"\n💡 POTENTIAL IMPACT:")
    print(f"  Current Total Styles Across All Stores: {total_current_styles}")
    print(f"  Recommended Total Styles: {total_recommended_styles}")
    
    if style_reduction > 0:
        print(f"  📉 Recommended Reduction: {style_reduction} styles ({style_reduction/total_current_styles*100:.1f}%)")
        print(f"  🎯 Focus Strategy: Concentrate on high-performing styles")
    elif style_reduction < 0:
        print(f"  📈 Recommended Increase: {abs(style_reduction)} styles ({abs(style_reduction)/total_current_styles*100:.1f}%)")
        print(f"  🚀 Growth Strategy: Expand successful style portfolios")
    else:
        print(f"  ⚖️ Current distribution is optimal")
    
    print(f"\n📊 STEP 3: Getting Specific Style Recommendations...")
    print("-" * 50)
    
    # Get specific recommendations for each store
    style_recommendations = analyzer.get_style_recommendations_by_store(
        style_metrics, store_recommendations
    )
    
    # Show detailed recommendations for top 3 stores
    top_stores = store_recommendations.nlargest(3, 'AVG_EFFICIENCY_SCORE')['STORE'].tolist()
    
    for store in top_stores[:3]:  # Show top 3 stores
        if store in style_recommendations:
            store_rec = style_recommendations[store]
            store_name = store_recommendations[store_recommendations['STORE'] == store]['STORE_NAME'].iloc[0] if 'STORE_NAME' in store_recommendations.columns else 'N/A'
            
            print(f"\n🏪 DETAILED RECOMMENDATIONS FOR {store} ({store_name}):")
            print(f"   Category: {store_rec['STORE_CATEGORY']}")
            print(f"   Recommended Style Count: {store_rec['RECOMMENDED_COUNT']}")
            
            if store_rec['STYLES']:
                styles_df = pd.DataFrame(store_rec['STYLES'])
                styles_display = styles_df[['STYLE', 'EFFICIENCY_SCORE', 'SALES_VELOCITY', 'TOTAL_SALES']].round(2)
                print(f"   Recommended Styles:")
                print(f"   {styles_display.to_string(index=False)}")
            else:
                print(f"   No specific style recommendations available")
    
    print(f"\n✅ Store Style Efficiency Analysis completed successfully!")
    print(f"📁 Sample data files are available for manual inspection:")
    print(f"   - sample_sales_data.csv")
    print(f"   - sample_stock_data.csv") 
    print(f"   - sample_warehouse_data.csv")
    print(f"   - sample_sku_master.csv")
    print(f"   - sample_style_master.csv")
    
    return style_metrics, store_recommendations, style_recommendations

if __name__ == "__main__":
    test_store_style_efficiency()