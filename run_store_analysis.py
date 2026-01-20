"""
Standalone Store Style Efficiency Analysis
Generates Excel output without Streamlit interface

This script loads data from file paths, runs the analysis, and exports results to Excel.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
from store_style_efficiency import StoreStyleEfficiencyAnalyzer

warnings.filterwarnings('ignore')

class StoreAnalysisRunner:
    def __init__(self):
        self.analyzer = StoreStyleEfficiencyAnalyzer()
        self.results = {}
        
    def load_data_from_paths(self, file_paths):
        """
        Load data from specified file paths
        
        Parameters:
        -----------
        file_paths : dict
            Dictionary with keys: 'sales', 'stock', 'warehouse', 'sku_master', 'style_master'
            Values should be file paths (CSV or Excel)
        """
        
        print("📂 Loading data from file paths...")
        loaded_data = {}
        
        for data_type, file_path in file_paths.items():
            if not file_path or not os.path.exists(file_path):
                print(f"⚠️  {data_type} file not found: {file_path}")
                continue
                
            try:
                print(f"   Loading {data_type} from: {file_path}")
                
                # Determine file type and load accordingly
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    print(f"❌ Unsupported file format for {data_type}: {file_path}")
                    continue
                
                # Clean column names
                df.columns = df.columns.str.strip().str.upper()
                
                # Convert date columns if present
                if data_type == 'sales' and 'DATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
                
                # Handle missing values
                if data_type in ['sales', 'stock', 'warehouse']:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(0)
                
                string_cols = df.select_dtypes(include=['object']).columns
                df[string_cols] = df[string_cols].fillna('').astype(str).str.strip()
                
                loaded_data[data_type] = df
                print(f"   ✅ {data_type}: {len(df)} records loaded")
                
            except Exception as e:
                print(f"❌ Error loading {data_type}: {str(e)}")
                continue
        
        return loaded_data
    
    def validate_data_structure(self, data):
        """Validate that required columns exist in the data"""
        
        print("\n🔍 Validating data structure...")
        
        required_columns = {
            'sales': ['DATE', 'STORE', 'SKU', 'QUANTITY'],
            'stock': ['STORE', 'SKU', 'STOCK'],
            'warehouse': ['SKU', 'WAREHOUSE_STOCK'],
            'sku_master': ['SKU', 'STYLE', 'COLOR', 'SIZE']
        }
        
        validation_passed = True
        
        for data_type, required_cols in required_columns.items():
            if data_type not in data:
                print(f"❌ Missing {data_type} data")
                validation_passed = False
                continue
                
            df = data[data_type]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ {data_type} missing columns: {missing_cols}")
                print(f"   Available columns: {list(df.columns)}")
                validation_passed = False
            else:
                print(f"✅ {data_type} structure validated")
        
        # Check for style_master (optional)
        if 'style_master' in data:
            style_df = data['style_master']
            if 'STYLE' in style_df.columns and 'GENDER' in style_df.columns:
                print(f"✅ style_master structure validated")
            else:
                print(f"⚠️  style_master missing required columns (STYLE, GENDER)")
        else:
            print(f"ℹ️  style_master not provided (optional)")
        
        return validation_passed
    
    def run_analysis(self, data, analysis_params=None):
        """
        Run the Store Style Efficiency analysis
        
        Parameters:
        -----------
        data : dict
            Dictionary containing the loaded DataFrames
        analysis_params : dict
            Analysis parameters (time_period_days, efficiency_threshold)
        """
        
        if analysis_params is None:
            analysis_params = {
                'time_period_days': 90,
                'efficiency_threshold': 60
            }
        
        print(f"\n🚀 Running Store Style Efficiency Analysis...")
        print(f"   Analysis Period: {analysis_params['time_period_days']} days")
        print(f"   Efficiency Threshold: {analysis_params['efficiency_threshold']}")
        
        # Load data into analyzer
        self.analyzer.load_data(
            sales_data=data['sales'],
            stock_data=data['stock'],
            warehouse_data=data['warehouse'],
            sku_master=data['sku_master'],
            style_master=data.get('style_master')
        )
        
        # Calculate style performance metrics
        print("\n📊 Step 1: Calculating style performance metrics...")
        style_metrics = self.analyzer.calculate_style_performance_metrics(
            time_period_days=analysis_params['time_period_days']
        )
        
        if style_metrics.empty:
            print("❌ Failed to calculate style metrics")
            return None
        
        print(f"   ✅ Calculated metrics for {len(style_metrics)} store-style combinations")
        self.results['style_metrics'] = style_metrics
        
        # Determine optimal styles per store
        print("\n🎯 Step 2: Determining optimal styles per store...")
        store_recommendations = self.analyzer.determine_optimal_styles_per_store(
            style_metrics, 
            efficiency_threshold=analysis_params['efficiency_threshold']
        )
        
        if store_recommendations.empty:
            print("❌ Failed to generate store recommendations")
            return None
        
        print(f"   ✅ Generated recommendations for {len(store_recommendations)} stores")
        self.results['store_recommendations'] = store_recommendations
        
        # Get specific style recommendations
        print("\n📝 Step 3: Getting specific style recommendations...")
        style_recommendations = self.analyzer.get_style_recommendations_by_store(
            style_metrics, store_recommendations
        )
        
        print(f"   ✅ Generated specific recommendations for {len(style_recommendations)} stores")
        self.results['style_recommendations'] = style_recommendations
        
        return self.results
    
    def generate_summary_stats(self):
        """Generate summary statistics for the analysis"""
        
        if 'store_recommendations' not in self.results:
            return {}
        
        store_recs = self.results['store_recommendations']
        
        summary = {
            'total_stores': len(store_recs),
            'avg_current_styles': store_recs['CURRENT_TOTAL_STYLES'].mean(),
            'avg_recommended_styles': store_recs['RECOMMENDED_STYLES'].mean(),
            'avg_efficiency_score': store_recs['AVG_EFFICIENCY_SCORE'].mean(),
            'total_current_styles': store_recs['CURRENT_TOTAL_STYLES'].sum(),
            'total_recommended_styles': store_recs['RECOMMENDED_STYLES'].sum(),
            'style_reduction': store_recs['CURRENT_TOTAL_STYLES'].sum() - store_recs['RECOMMENDED_STYLES'].sum(),
            'category_breakdown': store_recs['STORE_CATEGORY'].value_counts().to_dict()
        }
        
        return summary
    
    def export_to_excel(self, output_file="store_style_analysis_results.xlsx"):
        """Export all results to Excel file with multiple sheets"""
        
        if not self.results:
            print("❌ No results to export. Run analysis first.")
            return
        
        print(f"\n📊 Exporting results to Excel: {output_file}")
        
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                
                # Sheet 1: Store Recommendations Summary
                if 'store_recommendations' in self.results:
                    store_recs = self.results['store_recommendations'].copy()
                    
                    # Add store names if available in the display format
                    if 'STORE_NAME' in store_recs.columns:
                        store_recs['STORE_DISPLAY'] = store_recs['STORE'] + ' - ' + store_recs['STORE_NAME'].fillna('')
                        # Reorder columns
                        cols = ['STORE_DISPLAY'] + [col for col in store_recs.columns if col not in ['STORE', 'STORE_DISPLAY']]
                        store_recs = store_recs[cols]
                        store_recs = store_recs.drop('STORE', axis=1, errors='ignore')
                    
                    store_recs.to_excel(writer, sheet_name='Store_Recommendations', index=False)
                    print("   ✅ Store Recommendations sheet created")
                
                # Sheet 2: Style Performance Metrics
                if 'style_metrics' in self.results:
                    style_metrics = self.results['style_metrics'].copy()
                    
                    # Round numeric columns for better readability
                    numeric_cols = style_metrics.select_dtypes(include=[np.number]).columns
                    style_metrics[numeric_cols] = style_metrics[numeric_cols].round(2)
                    
                    style_metrics.to_excel(writer, sheet_name='Style_Performance', index=False)
                    print("   ✅ Style Performance sheet created")
                
                # Sheet 3: Top Performing Styles
                if 'style_metrics' in self.results:
                    top_styles = self.results['style_metrics'].nlargest(50, 'STYLE_EFFICIENCY_SCORE')[
                        ['STORE', 'STYLE', 'STYLE_EFFICIENCY_SCORE', 'SALES_VELOCITY', 
                         'STOCK_TURNOVER', 'TOTAL_SALES', 'STOCK']
                    ].round(2)
                    
                    top_styles.to_excel(writer, sheet_name='Top_Performing_Styles', index=False)
                    print("   ✅ Top Performing Styles sheet created")
                
                # Sheet 4: Store Category Analysis
                if 'store_recommendations' in self.results:
                    store_recs = self.results['store_recommendations']
                    
                    category_analysis = store_recs.groupby('STORE_CATEGORY').agg({
                        'CURRENT_TOTAL_STYLES': ['count', 'mean', 'sum'],
                        'RECOMMENDED_STYLES': ['mean', 'sum'],
                        'AVG_EFFICIENCY_SCORE': 'mean',
                        'TOTAL_SALES': 'mean'
                    }).round(2)
                    
                    category_analysis.columns = [
                        'Store_Count', 'Avg_Current_Styles', 'Total_Current_Styles',
                        'Avg_Recommended_Styles', 'Total_Recommended_Styles',
                        'Avg_Efficiency_Score', 'Avg_Total_Sales'
                    ]
                    
                    category_analysis.to_excel(writer, sheet_name='Category_Analysis', index=True)
                    print("   ✅ Category Analysis sheet created")
                
                # Sheet 5: Detailed Style Recommendations (flattened)
                if 'style_recommendations' in self.results:
                    detailed_recs = []
                    
                    for store, rec_data in self.results['style_recommendations'].items():
                        for style_info in rec_data['STYLES']:
                            detailed_recs.append({
                                'STORE': store,
                                'STORE_CATEGORY': rec_data['STORE_CATEGORY'],
                                'RECOMMENDED_STYLE': style_info['STYLE'],
                                'EFFICIENCY_SCORE': round(style_info['EFFICIENCY_SCORE'], 2),
                                'SALES_VELOCITY': round(style_info['SALES_VELOCITY'], 2),
                                'STOCK_TURNOVER': round(style_info['STOCK_TURNOVER'], 2),
                                'TOTAL_SALES': style_info['TOTAL_SALES'],
                                'CURRENT_STOCK': style_info['CURRENT_STOCK']
                            })
                    
                    if detailed_recs:
                        detailed_df = pd.DataFrame(detailed_recs)
                        detailed_df.to_excel(writer, sheet_name='Detailed_Style_Recs', index=False)
                        print("   ✅ Detailed Style Recommendations sheet created")
                
                # Sheet 6: Summary Statistics
                summary_stats = self.generate_summary_stats()
                if summary_stats:
                    summary_df = pd.DataFrame([summary_stats]).T
                    summary_df.columns = ['Value']
                    summary_df.index.name = 'Metric'
                    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=True)
                    print("   ✅ Summary Statistics sheet created")
            
            print(f"\n✅ Excel file exported successfully: {output_file}")
            print(f"📁 File location: {os.path.abspath(output_file)}")
            
            return os.path.abspath(output_file)
            
        except Exception as e:
            print(f"❌ Error exporting to Excel: {str(e)}")
            return None

def main():
    """Main function to run the analysis"""
    
    print("🎯 Store Style Efficiency Analysis - Standalone Version")
    print("=" * 60)
    
    # Initialize the runner
    runner = StoreAnalysisRunner()
    
    # Example file paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    file_paths = {
        'sales': 'sample_sales_data.csv',
        'stock': 'sample_stock_data.csv', 
        'warehouse': 'sample_warehouse_data.csv',
        'sku_master': 'sample_sku_master.csv',
        'style_master': 'sample_style_master.csv'
    }
    
    print("📋 Current file paths configuration:")
    for data_type, path in file_paths.items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"   {data_type}: {path} {exists}")
    
    print(f"\n💡 To use your own data files, update the file_paths dictionary in the script")
    print(f"   or modify the paths below:")
    
    # Load data
    data = runner.load_data_from_paths(file_paths)
    
    if not data:
        print("\n❌ No data loaded. Please check file paths and try again.")
        return
    
    # Validate data structure
    if not runner.validate_data_structure(data):
        print("\n❌ Data validation failed. Please check your data structure.")
        return
    
    # Analysis parameters
    analysis_params = {
        'time_period_days': 90,        # Analyze last 90 days
        'efficiency_threshold': 60     # Minimum efficiency score for viable styles
    }
    
    # Run analysis
    results = runner.run_analysis(data, analysis_params)
    
    if results:
        # Print summary
        summary = runner.generate_summary_stats()
        
        print(f"\n📈 ANALYSIS SUMMARY:")
        print(f"   Total Stores: {summary['total_stores']}")
        print(f"   Avg Current Styles/Store: {summary['avg_current_styles']:.1f}")
        print(f"   Avg Recommended Styles/Store: {summary['avg_recommended_styles']:.1f}")
        print(f"   Avg Efficiency Score: {summary['avg_efficiency_score']:.1f}")
        print(f"   Total Style Reduction: {summary['style_reduction']} ({summary['style_reduction']/summary['total_current_styles']*100:.1f}%)")
        
        print(f"\n🏪 Store Categories:")
        for category, count in summary['category_breakdown'].items():
            print(f"   {category}: {count} stores")
        
        # Export to Excel
        output_file = f"store_style_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        exported_file = runner.export_to_excel(output_file)
        
        if exported_file:
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📊 Results exported to: {exported_file}")
        else:
            print(f"\n⚠️  Analysis completed but export failed.")
    
    else:
        print(f"\n❌ Analysis failed. Please check your data and try again.")

if __name__ == "__main__":
    main()