"""
Easy Configuration Script for Store Style Efficiency Analysis
Update the file paths below with your actual data files and run the analysis
"""

from standalone_store_analysis import StoreAnalysisRunner
import os

def run_analysis_with_custom_paths():
    """
    Configure your data file paths here and run the analysis
    """
    
    # ===============================
    # UPDATE THESE PATHS WITH YOUR DATA FILES
    # ===============================
    
    file_paths = {
        # Sales data should have columns: DATE, STORE, SKU, QUANTITY
        'sales': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\EBO SALES FOLDER\EBO SALES DATA.xlsx',
        
        # Stock data should have columns: STORE, SKU, STOCK  
        'stock': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\STOCK\EBo Stock Data.xlsx',
        
        # Warehouse data should have columns: SKU, WAREHOUSE_STOCK
        'warehouse': r'D:\DATA TILL DATE\Downloads\Inventory Available for Sales - OMS-2025-10-31T12_38_35.558+05_30.csv',
        
        # SKU Master should have columns: SKU, STYLE, COLOR, SIZE
        'sku_master': r'sample_sku_master.csv',
        
        # Style Master should have columns: STYLE, GENDER (optional)
        'style_master': r'sample_style_master.csv'
    }
    
    # ===============================
    # ANALYSIS PARAMETERS (OPTIONAL TO MODIFY)
    # ===============================
    
    analysis_params = {
        'time_period_days': 90,        # Number of days to analyze (30, 60, 90, 120)
        'efficiency_threshold': 60     # Minimum score for efficient styles (30-80)
    }
    
    # ===============================
    # OUTPUT FILE NAME (OPTIONAL TO MODIFY)
    # ===============================
    
    from datetime import datetime
    output_file = f"store_efficiency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # ===============================
    # RUN THE ANALYSIS
    # ===============================
    
    print("🎯 Store Style Efficiency Analysis")
    print("=" * 50)
    print("📋 Configuration:")
    print(f"   Analysis Period: {analysis_params['time_period_days']} days")
    print(f"   Efficiency Threshold: {analysis_params['efficiency_threshold']}")
    print(f"   Output File: {output_file}")
    print()
    
    # Check if files exist
    print("📂 Checking data files:")
    all_files_exist = True
    for data_type, path in file_paths.items():
        if os.path.exists(path):
            print(f"   ✅ {data_type}: {path}")
        else:
            print(f"   ❌ {data_type}: {path} (FILE NOT FOUND)")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some data files are missing. Please check the file paths above.")
        print("💡 Update the file_paths dictionary in this script with your actual file locations.")
        return
    
    # Initialize and run analysis
    runner = StoreAnalysisRunner()
    
    # Load data
    data = runner.load_data_from_paths(file_paths)
    
    if not data:
        print("\n❌ Failed to load data. Please check file paths and file formats.")
        return
    
    # Validate data structure
    if not runner.validate_data_structure(data):
        print("\n❌ Data validation failed. Please check your data structure.")
        print("\n📋 Required columns by file type:")
        print("   Sales: DATE, STORE, SKU, QUANTITY")
        print("   Stock: STORE, SKU, STOCK")
        print("   Warehouse: SKU, WAREHOUSE_STOCK")
        print("   SKU Master: SKU, STYLE, COLOR, SIZE")
        print("   Style Master: STYLE, GENDER (optional)")
        return
    
    # Run analysis
    results = runner.run_analysis(data, analysis_params)
    
    if results:
        # Export to Excel
        exported_file = runner.export_to_excel(output_file)
        
        if exported_file:
            # Print final summary
            summary = runner.generate_summary_stats()
            
            print("\n" + "="*50)
            print("📈 FINAL ANALYSIS RESULTS")
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
            print("📊 Excel file contains 6 sheets:")
            print("   1. Store_Recommendations - Main recommendations per store")
            print("   2. Style_Performance - Detailed style metrics")
            print("   3. Top_Performing_Styles - Best 50 styles across all stores")
            print("   4. Category_Analysis - Store category breakdown")
            print("   5. Detailed_Style_Recs - Specific style recommendations")
            print("   6. Summary_Statistics - Overall analysis summary")
            
            print()
            print("✨ Analysis complete! Open the Excel file to view detailed results.")
            
        else:
            print("\n⚠️ Analysis completed but Excel export failed.")
    
    else:
        print("\n❌ Analysis failed. Please check your data and try again.")

if __name__ == "__main__":
    run_analysis_with_custom_paths()