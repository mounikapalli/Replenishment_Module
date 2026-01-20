"""
Clean Store Style Efficiency Analysis without Streamlit dependencies
Generates Excel output for standalone use
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class CleanStoreStyleAnalyzer:
    """
    Clean version of Store Style Efficiency Analyzer without Streamlit dependencies
    """
    
    def __init__(self):
        self.sales_data = None
        self.stock_data = None
        self.warehouse_data = None
        self.sku_master = None
        self.style_master = None
        
    def load_data(self, sales_data, stock_data, warehouse_data, sku_master, style_master=None):
        """Load all required datasets for analysis"""
        self.sales_data = sales_data.copy()
        self.stock_data = stock_data.copy()
        self.warehouse_data = warehouse_data.copy()
        self.sku_master = sku_master.copy()
        self.style_master = style_master.copy() if style_master is not None else None
        
    def calculate_style_performance_metrics(self, time_period_days=90):
        """Calculate comprehensive style performance metrics for each store"""
        
        if any(df is None or df.empty for df in [self.sales_data, self.stock_data, self.sku_master]):
            print("❌ Required data not loaded. Please ensure sales, stock, and SKU master data are available.")
            return pd.DataFrame()
        
        try:
            # Get recent sales data
            latest_date = self.sales_data['DATE'].max()
            start_date = latest_date - pd.Timedelta(days=time_period_days)
            recent_sales = self.sales_data[self.sales_data['DATE'] > start_date].copy()
            
            # Merge sales with SKU master to get style information
            sales_with_style = pd.merge(
                recent_sales,
                self.sku_master[['SKU', 'STYLE', 'COLOR', 'SIZE']],
                on='SKU',
                how='left'
            )
            
            # Group by store and style to calculate metrics
            style_sales = sales_with_style.groupby(['STORE', 'STYLE']).agg({
                'QUANTITY': ['sum', 'count', 'mean'],
                'DATE': ['min', 'max']
            }).reset_index()
            
            # Flatten column names
            style_sales.columns = [
                'STORE', 'STYLE', 'TOTAL_SALES', 'SALES_TRANSACTIONS', 
                'AVG_SALES_PER_TXN', 'FIRST_SALE_DATE', 'LAST_SALE_DATE'
            ]
            
            # Calculate sales velocity (daily sales)
            style_sales['SALES_VELOCITY'] = style_sales['TOTAL_SALES'] / time_period_days
            
            # Calculate sales consistency (how regularly the style sells)
            style_sales['SALES_FREQUENCY'] = style_sales['SALES_TRANSACTIONS'] / time_period_days
            
            # Merge with stock data to get current stock levels by style
            stock_with_style = pd.merge(
                self.stock_data,
                self.sku_master[['SKU', 'STYLE']],
                on='SKU',
                how='left'
            )
            
            # Aggregate stock by store and style
            style_stock = stock_with_style.groupby(['STORE', 'STYLE']).agg({
                'STOCK': 'sum'
            }).reset_index()
            
            # Merge sales and stock data
            style_metrics = pd.merge(
                style_sales,
                style_stock,
                on=['STORE', 'STYLE'],
                how='outer'
            ).fillna(0)
            
            # Calculate key performance metrics
            
            # 1. Stock Turnover (higher is better)
            style_metrics['STOCK_TURNOVER'] = np.where(
                style_metrics['STOCK'] > 0,
                style_metrics['TOTAL_SALES'] / style_metrics['STOCK'],
                np.where(style_metrics['TOTAL_SALES'] > 0, 999, 0)
            )
            
            # 2. Days of Stock Cover (lower is better for efficiency)
            style_metrics['DAYS_STOCK_COVER'] = np.where(
                style_metrics['SALES_VELOCITY'] > 0,
                style_metrics['STOCK'] / style_metrics['SALES_VELOCITY'],
                np.where(style_metrics['STOCK'] > 0, 999, 0)
            )
            
            # 3. Revenue potential
            style_metrics['REVENUE_CONTRIBUTION'] = style_metrics['TOTAL_SALES']
            
            # 4. Style consistency score
            max_transactions = style_metrics['SALES_TRANSACTIONS'].max()
            style_metrics['CONSISTENCY_SCORE'] = np.where(
                max_transactions > 0,
                style_metrics['SALES_TRANSACTIONS'] / max_transactions * 100,
                0
            )
            
            # 5. Calculate overall efficiency score (0-100)
            
            # Sales velocity score (30% weight)
            max_velocity = style_metrics['SALES_VELOCITY'].max()
            velocity_score = np.where(
                max_velocity > 0,
                (style_metrics['SALES_VELOCITY'] / max_velocity) * 30,
                0
            )
            
            # Stock turnover score (25% weight)
            capped_turnover = style_metrics['STOCK_TURNOVER'].clip(0, 10)
            turnover_score = (capped_turnover / 10) * 25
            
            # Consistency score (20% weight)
            consistency_score = (style_metrics['CONSISTENCY_SCORE'] / 100) * 20
            
            # Revenue contribution score (15% weight)
            max_revenue = style_metrics['REVENUE_CONTRIBUTION'].max()
            revenue_score = np.where(
                max_revenue > 0,
                (style_metrics['REVENUE_CONTRIBUTION'] / max_revenue) * 15,
                0
            )
            
            # Stock efficiency score (10% weight)
            stock_efficiency_score = np.where(
                style_metrics['DAYS_STOCK_COVER'] > 0,
                np.maximum(0, 10 - (style_metrics['DAYS_STOCK_COVER'] / 30) * 10),
                10
            )
            
            # Calculate final efficiency score
            style_metrics['STYLE_EFFICIENCY_SCORE'] = (
                velocity_score + 
                turnover_score + 
                consistency_score + 
                revenue_score + 
                stock_efficiency_score
            ).round(1)
            
            # Add warehouse availability information
            if self.warehouse_data is not None:
                warehouse_with_style = pd.merge(
                    self.warehouse_data,
                    self.sku_master[['SKU', 'STYLE']],
                    on='SKU',
                    how='left'
                )
                
                warehouse_by_style = warehouse_with_style.groupby('STYLE').agg({
                    'WAREHOUSE_STOCK': 'sum'
                }).reset_index()
                
                style_metrics = pd.merge(
                    style_metrics,
                    warehouse_by_style,
                    on='STYLE',
                    how='left'
                )
                style_metrics['WAREHOUSE_STOCK'] = style_metrics['WAREHOUSE_STOCK'].fillna(0)
                
                # Adjust efficiency score based on warehouse availability
                warehouse_factor = np.where(
                    style_metrics['WAREHOUSE_STOCK'] > 0, 1.0, 0.7
                )
                style_metrics['STYLE_EFFICIENCY_SCORE'] *= warehouse_factor
            
            # Add style category information if available
            if self.style_master is not None and 'GENDER' in self.style_master.columns:
                style_metrics = pd.merge(
                    style_metrics,
                    self.style_master[['STYLE', 'GENDER']],
                    on='STYLE',
                    how='left'
                )
                style_metrics['GENDER'] = style_metrics['GENDER'].fillna('UNISEX')
            
            return style_metrics
            
        except Exception as e:
            print(f"❌ Error calculating style performance metrics: {str(e)}")
            return pd.DataFrame()
    
    def determine_optimal_styles_per_store(self, style_metrics, efficiency_threshold=60):
        """Determine the optimal number of styles for each store"""
        
        if style_metrics.empty:
            return pd.DataFrame()
        
        try:
            # Filter for efficient styles only
            efficient_styles = style_metrics[
                style_metrics['STYLE_EFFICIENCY_SCORE'] >= efficiency_threshold
            ].copy()
            
            # Calculate store-level metrics
            store_analysis = []
            
            for store in style_metrics['STORE'].unique():
                store_data = style_metrics[style_metrics['STORE'] == store].copy()
                efficient_store_data = efficient_styles[efficient_styles['STORE'] == store].copy()
                
                # Basic metrics
                total_styles = len(store_data)
                efficient_styles_count = len(efficient_store_data)
                
                # Sales performance
                total_sales = store_data['TOTAL_SALES'].sum()
                efficient_sales = efficient_store_data['TOTAL_SALES'].sum()
                efficient_sales_pct = (efficient_sales / total_sales * 100) if total_sales > 0 else 0
                
                # Stock efficiency
                total_stock = store_data['STOCK'].sum()
                efficient_stock = efficient_store_data['STOCK'].sum()
                
                # Average metrics for efficient styles
                avg_efficiency_score = efficient_store_data['STYLE_EFFICIENCY_SCORE'].mean() if not efficient_store_data.empty else 0
                avg_sales_velocity = efficient_store_data['SALES_VELOCITY'].mean() if not efficient_store_data.empty else 0
                avg_stock_turnover = efficient_store_data['STOCK_TURNOVER'].mean() if not efficient_store_data.empty else 0
                
                # Store performance categories
                store_total_sales = store_data['TOTAL_SALES'].sum()
                store_avg_efficiency = store_data['STYLE_EFFICIENCY_SCORE'].mean()
                
                if store_avg_efficiency >= 70 and store_total_sales >= store_data['TOTAL_SALES'].quantile(0.75):
                    store_category = "High Performer"
                    recommended_styles = min(efficient_styles_count + 2, total_styles)
                    max_recommended = min(20, total_styles)
                elif store_avg_efficiency >= 50 and store_total_sales >= store_data['TOTAL_SALES'].quantile(0.5):
                    store_category = "Good Performer"
                    recommended_styles = efficient_styles_count
                    max_recommended = min(15, total_styles)
                elif store_avg_efficiency >= 30:
                    store_category = "Average Performer"
                    recommended_styles = max(efficient_styles_count - 1, 1)
                    max_recommended = min(10, total_styles)
                else:
                    store_category = "Needs Improvement"
                    recommended_styles = max(efficient_styles_count // 2, 1)
                    max_recommended = min(8, total_styles)
                
                recommended_styles = max(1, min(recommended_styles, max_recommended))
                
                # Calculate potential impact
                if efficient_styles_count > 0:
                    top_styles = efficient_store_data.nlargest(recommended_styles, 'STYLE_EFFICIENCY_SCORE')
                    projected_sales = top_styles['TOTAL_SALES'].sum()
                    projected_sales_improvement = ((projected_sales / total_sales) * 100) if total_sales > 0 else 0
                else:
                    projected_sales_improvement = 0
                
                store_analysis.append({
                    'STORE': store,
                    'STORE_CATEGORY': store_category,
                    'CURRENT_TOTAL_STYLES': total_styles,
                    'CURRENT_EFFICIENT_STYLES': efficient_styles_count,
                    'RECOMMENDED_STYLES': recommended_styles,
                    'EFFICIENCY_IMPROVEMENT': recommended_styles - efficient_styles_count,
                    'TOTAL_SALES': total_sales,
                    'EFFICIENT_SALES_PCT': round(efficient_sales_pct, 1),
                    'AVG_EFFICIENCY_SCORE': round(avg_efficiency_score, 1),
                    'AVG_SALES_VELOCITY': round(avg_sales_velocity, 2),
                    'AVG_STOCK_TURNOVER': round(avg_stock_turnover, 2),
                    'PROJECTED_SALES_FOCUS_PCT': round(projected_sales_improvement, 1),
                    'TOTAL_STOCK': total_stock,
                    'EFFICIENT_STOCK': efficient_stock
                })
            
            store_recommendations = pd.DataFrame(store_analysis)
            
            # Add store names if available
            if 'STORE_NAME' in self.sales_data.columns:
                store_names = self.sales_data[['STORE', 'STORE_NAME']].drop_duplicates()
                store_recommendations = pd.merge(
                    store_recommendations,
                    store_names,
                    on='STORE',
                    how='left'
                )
            elif 'STORE_NAME' in self.stock_data.columns:
                store_names = self.stock_data[['STORE', 'STORE_NAME']].drop_duplicates()
                store_recommendations = pd.merge(
                    store_recommendations,
                    store_names,
                    on='STORE',
                    how='left'
                )
            
            return store_recommendations.sort_values('AVG_EFFICIENCY_SCORE', ascending=False)
            
        except Exception as e:
            print(f"❌ Error determining optimal styles per store: {str(e)}")
            return pd.DataFrame()
    
    def get_style_recommendations_by_store(self, style_metrics, store_recommendations):
        """Get specific style recommendations for each store"""
        
        if style_metrics.empty or store_recommendations.empty:
            return {}
        
        recommendations = {}
        
        for _, store_row in store_recommendations.iterrows():
            store = store_row['STORE']
            recommended_count = store_row['RECOMMENDED_STYLES']
            
            # Get styles for this store sorted by efficiency score
            store_styles = style_metrics[
                style_metrics['STORE'] == store
            ].sort_values('STYLE_EFFICIENCY_SCORE', ascending=False)
            
            # Get top N styles
            top_styles = store_styles.head(recommended_count)
            
            # Create recommendation details
            style_details = []
            for _, style_row in top_styles.iterrows():
                style_details.append({
                    'STYLE': style_row['STYLE'],
                    'EFFICIENCY_SCORE': style_row['STYLE_EFFICIENCY_SCORE'],
                    'SALES_VELOCITY': style_row['SALES_VELOCITY'],
                    'STOCK_TURNOVER': style_row['STOCK_TURNOVER'],
                    'TOTAL_SALES': style_row['TOTAL_SALES'],
                    'CURRENT_STOCK': style_row['STOCK']
                })
            
            recommendations[store] = {
                'RECOMMENDED_COUNT': recommended_count,
                'STYLES': style_details,
                'STORE_CATEGORY': store_row['STORE_CATEGORY']
            }
        
        return recommendations


class StoreAnalysisRunner:
    def __init__(self):
        self.analyzer = CleanStoreStyleAnalyzer()
        self.results = {}
        
    def load_data_from_paths(self, file_paths):
        """Load data from specified file paths"""
        
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
                df.columns = [str(col).strip().upper() for col in df.columns]
                
                # Convert date columns if present
                if data_type == 'sales' and 'DATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
                
                # Handle missing values
                if data_type in ['sales', 'stock', 'warehouse']:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(0)
                
                # Handle string columns properly
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna('').astype(str)
                        df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
                
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
        """Run the Store Style Efficiency analysis"""
        
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