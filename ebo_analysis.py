"""
Custom data processor for specific EBO data format
Maps actual column names to standard format for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from standalone_store_analysis import StoreAnalysisRunner
import os

def process_ebo_data():
    """
    Process EBO data files with custom column mapping
    """
    
    print("🎯 EBO Store Style Efficiency Analysis")
    print("=" * 50)
    
    # File paths
    file_paths = {
        'sales': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\EBO SALES FOLDER\EBO SALES DATA.xlsx',
        'stock': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\STOCK\EBo Stock Data.xlsx',
        'warehouse': r'D:\DATA TILL DATE\Downloads\Inventory Available for Sales - OMS-2025-10-31T12_38_35.558+05_30.csv',
    }
    
    print("📂 Loading and processing EBO data...")
    
    # Load Sales Data
    print("   Loading sales data...")
    try:
        sales_raw = pd.read_excel(file_paths['sales'])
        print(f"   ✅ Sales: {len(sales_raw)} records loaded")
        
        # Map columns to standard format (using actual column names)
        sales_df = pd.DataFrame({
            'DATE': pd.to_datetime(sales_raw['BILL_DATE'], errors='coerce'),
            'STORE': sales_raw['store_code'].astype(str).str.strip(),
            'STORE_NAME': sales_raw['EBO NAME'].astype(str).str.strip(),
            'SKU': sales_raw['SKU'].astype(str).str.strip(),
            'QUANTITY': pd.to_numeric(sales_raw['BILL_QUANTITY'], errors='coerce').fillna(0),
            'STYLE': sales_raw['STYLE'].astype(str).str.strip(),
            'COLOR': sales_raw['COLOR'].astype(str).str.strip(),
            'SIZE': sales_raw['SIZE'].astype(str).str.strip(),
            'DEPARTMENT': sales_raw['DEPARTMENT'].astype(str).str.strip()
        })
        
        # Filter out invalid data
        sales_df = sales_df.dropna(subset=['DATE'])
        sales_df = sales_df[sales_df['QUANTITY'] > 0]
        sales_df = sales_df[sales_df['SKU'] != '']
        
        print(f"   ✅ Sales processed: {len(sales_df)} valid records")
        
    except Exception as e:
        print(f"   ❌ Error loading sales data: {str(e)}")
        return None
    
    # Load Stock Data
    print("   Loading stock data...")
    try:
        stock_raw = pd.read_excel(file_paths['stock'])
        print(f"   ✅ Stock: {len(stock_raw)} records loaded")
        
        # Map columns to standard format (using actual column names)
        stock_df = pd.DataFrame({
            'STORE': stock_raw['store_code'].astype(str).str.strip(),
            'STORE_NAME': stock_raw['Store Name'].astype(str).str.strip(),
            'SKU': stock_raw['SKU'].astype(str).str.strip(),
            'STOCK': pd.to_numeric(stock_raw['quantity'], errors='coerce').fillna(0),
            'STYLE': stock_raw['Style'].astype(str).str.strip(),
            'COLOR': stock_raw['Colour'].astype(str).str.strip(),
            'SIZE': stock_raw['Size'].astype(str).str.strip(),
            'DEPARTMENT': stock_raw['Department'].astype(str).str.strip()
        })
        
        # Filter out invalid data
        stock_df = stock_df[stock_df['SKU'] != '']
        
        print(f"   ✅ Stock processed: {len(stock_df)} valid records")
        
    except Exception as e:
        print(f"   ❌ Error loading stock data: {str(e)}")
        return None
    
    # Load Warehouse Data  
    print("   Loading warehouse data...")
    try:
        warehouse_raw = pd.read_csv(file_paths['warehouse'])
        print(f"   ✅ Warehouse: {len(warehouse_raw)} records loaded")
        
        # Map columns to standard format (using actual column names)
        warehouse_df = pd.DataFrame({
            'SKU': warehouse_raw['Each Client SKU ID'].astype(str).str.strip(),
            'WAREHOUSE_STOCK': pd.to_numeric(warehouse_raw['Total Available Quantity'], errors='coerce').fillna(0),
            'STYLE': warehouse_raw['Style'].astype(str).str.strip() if 'Style' in warehouse_raw.columns else '',
            'COLOR': warehouse_raw['Color'].astype(str).str.strip() if 'Color' in warehouse_raw.columns else '',
            'SIZE': warehouse_raw['Size'].astype(str).str.strip() if 'Size' in warehouse_raw.columns else ''
        })
        
        # Filter out invalid data
        warehouse_df = warehouse_df[warehouse_df['SKU'] != '']
        
        print(f"   ✅ Warehouse processed: {len(warehouse_df)} valid records")
        
    except Exception as e:
        print(f"   ❌ Error loading warehouse data: {str(e)}")
        return None
    
    # Create SKU Master from sales data (since it has the most complete SKU info)
    print("   Creating SKU master from sales data...")
    sku_master_df = sales_df[['SKU', 'STYLE', 'COLOR', 'SIZE']].drop_duplicates()
    sku_master_df = sku_master_df[sku_master_df['SKU'] != '']
    print(f"   ✅ SKU Master created: {len(sku_master_df)} unique SKUs")
    
    # Create Style Master from sales data
    print("   Creating style master from sales data...")
    style_master_df = sales_df[['STYLE', 'DEPARTMENT']].drop_duplicates()
    style_master_df = style_master_df.rename(columns={'DEPARTMENT': 'GENDER'})
    style_master_df = style_master_df[style_master_df['STYLE'] != '']
    print(f"   ✅ Style Master created: {len(style_master_df)} unique styles")
    
    # Data summary
    print(f"\n📊 Data Summary:")
    print(f"   Sales Records: {len(sales_df):,}")
    print(f"   Stock Records: {len(stock_df):,}")
    print(f"   Warehouse Records: {len(warehouse_df):,}")
    print(f"   Unique SKUs: {len(sku_master_df):,}")
    print(f"   Unique Styles: {len(style_master_df):,}")
    print(f"   Unique Stores: {sales_df['STORE'].nunique()}")
    print(f"   Date Range: {sales_df['DATE'].min()} to {sales_df['DATE'].max()}")
    
    # Show store information
    store_info = sales_df.groupby('STORE').agg({
        'STORE_NAME': 'first',
        'QUANTITY': 'sum',
        'SKU': 'nunique'
    }).reset_index()
    store_info.columns = ['STORE_CODE', 'STORE_NAME', 'TOTAL_SALES', 'UNIQUE_SKUS']
    
    print(f"\n🏪 Store Information:")
    for _, row in store_info.head(10).iterrows():
        print(f"   {row['STORE_CODE']} - {row['STORE_NAME']}: {row['TOTAL_SALES']:,} sales, {row['UNIQUE_SKUS']} SKUs")
    
    # Return processed data
    data = {
        'sales': sales_df,
        'stock': stock_df,
        'warehouse': warehouse_df,
        'sku_master': sku_master_df,
        'style_master': style_master_df
    }
    
    return data

def run_ebo_analysis():
    """Run the analysis with processed EBO data"""
    
    # Process the data
    data = process_ebo_data()
    
    if not data:
        print("❌ Failed to process data")
        return
    
    # Analysis parameters
    analysis_params = {
        'time_period_days': 90,        # Analyze last 90 days
        'efficiency_threshold': 50     # Lower threshold for first run
    }
    
    print(f"\n🚀 Running Store Style Efficiency Analysis...")
    print(f"   Analysis Period: {analysis_params['time_period_days']} days")
    print(f"   Efficiency Threshold: {analysis_params['efficiency_threshold']}")
    
    # Initialize the runner
    runner = StoreAnalysisRunner()
    
    # Load data into analyzer
    runner.analyzer.load_data(
        sales_data=data['sales'],
        stock_data=data['stock'],
        warehouse_data=data['warehouse'],
        sku_master=data['sku_master'],
        style_master=data['style_master']
    )
    
    # Run analysis
    results = runner.run_analysis(data, analysis_params)
    
    if results:
        # Export to Excel
        output_file = f"EBO_store_efficiency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        exported_file = runner.export_to_excel(output_file)
        
        if exported_file:
            # Print final summary
            summary = runner.generate_summary_stats()
            
            print("\n" + "="*50)
            print("📈 EBO ANALYSIS RESULTS")
            print("="*50)
            print(f"✅ Analysis completed successfully!")
            print(f"📊 Excel file: {exported_file}")
            print()
            print("📋 Summary:")
            print(f"   • Total Stores Analyzed: {summary['total_stores']}")
            print(f"   • Average Current Styles per Store: {summary['avg_current_styles']:.1f}")
            print(f"   • Average Recommended Styles per Store: {summary['avg_recommended_styles']:.1f}")
            print(f"   • Overall Efficiency Score: {summary['avg_efficiency_score']:.1f}")
            
            reduction_pct = (summary['style_reduction'] / summary['total_current_styles'] * 100)
            if summary['style_reduction'] > 0:
                print(f"   • Recommended Style Reduction: {summary['style_reduction']} styles ({reduction_pct:.1f}%)")
                print("   • Strategy: Focus on high-performing styles")
            else:
                print(f"   • Recommended Style Increase: {abs(summary['style_reduction'])} styles")
                print("   • Strategy: Expand successful portfolios")
            
            print()
            print("🏪 Store Performance Distribution:")
            for category, count in summary['category_breakdown'].items():
                print(f"   • {category}: {count} stores")
            
            print()
            print("📊 Excel file contains 6 sheets with detailed analysis:")
            print("   1. Store_Recommendations - Main recommendations per store")
            print("   2. Style_Performance - Detailed style metrics")
            print("   3. Top_Performing_Styles - Best performing styles")
            print("   4. Category_Analysis - Store category breakdown")
            print("   5. Detailed_Style_Recs - Specific style recommendations")
            print("   6. Summary_Statistics - Overall analysis summary")
            
            print()
            print("✨ EBO Analysis complete! Open the Excel file to view detailed results.")
            
        else:
            print("\n⚠️ Analysis completed but Excel export failed.")
    
    else:
        print("\n❌ Analysis failed. Please check your data and try again.")

if __name__ == "__main__":
    run_ebo_analysis()