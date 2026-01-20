"""
Store Style Efficiency Analysis Module

This module analyzes sales, stock, and warehouse data to determine the optimal number of styles
that will be efficient for each store based on:
- Sales velocity and performance per style
- Stock turnover rates
- Warehouse availability
- Store capacity and performance metrics
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class StoreStyleEfficiencyAnalyzer:
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
        """
        Calculate comprehensive style performance metrics for each store
        
        Returns:
        --------
        DataFrame with columns:
        - STORE: Store identifier
        - STYLE: Style code
        - SALES_VELOCITY: Average daily sales
        - STOCK_TURNOVER: How fast the style moves (sales/stock ratio)
        - DAYS_STOCK_COVER: Days of stock remaining
        - PROFIT_CONTRIBUTION: Revenue contribution
        - STYLE_EFFICIENCY_SCORE: Overall efficiency score (0-100)
        """
        if any(df is None or df.empty for df in [self.sales_data, self.stock_data, self.sku_master]):
            st.error("Required data not loaded. Please ensure sales, stock, and SKU master data are available.")
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
                np.where(style_metrics['TOTAL_SALES'] > 0, 999, 0)  # High value for sold-out items
            )
            
            # 2. Days of Stock Cover (lower is better for efficiency)
            style_metrics['DAYS_STOCK_COVER'] = np.where(
                style_metrics['SALES_VELOCITY'] > 0,
                style_metrics['STOCK'] / style_metrics['SALES_VELOCITY'],
                np.where(style_metrics['STOCK'] > 0, 999, 0)  # High value for slow movers
            )
            
            # 3. Revenue potential (assuming average selling price)
            # For now, we'll use quantity as proxy for revenue
            style_metrics['REVENUE_CONTRIBUTION'] = style_metrics['TOTAL_SALES']
            
            # 4. Style consistency score (based on regular sales pattern)
            max_transactions = style_metrics['SALES_TRANSACTIONS'].max()
            style_metrics['CONSISTENCY_SCORE'] = np.where(
                max_transactions > 0,
                style_metrics['SALES_TRANSACTIONS'] / max_transactions * 100,
                0
            )
            
            # 5. Calculate overall efficiency score (0-100)
            # Normalize each metric to 0-100 scale
            
            # Sales velocity score (30% weight)
            max_velocity = style_metrics['SALES_VELOCITY'].max()
            velocity_score = np.where(
                max_velocity > 0,
                (style_metrics['SALES_VELOCITY'] / max_velocity) * 30,
                0
            )
            
            # Stock turnover score (25% weight)
            # Cap turnover at 10 for scoring to avoid extreme values
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
            
            # Stock efficiency score (10% weight) - lower stock cover is better
            # Invert the days stock cover for scoring (fewer days = higher score)
            stock_efficiency_score = np.where(
                style_metrics['DAYS_STOCK_COVER'] > 0,
                np.maximum(0, 10 - (style_metrics['DAYS_STOCK_COVER'] / 30) * 10),
                10  # Maximum score for items with no stock but sales
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
                # Get warehouse stock by style
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
                    style_metrics['WAREHOUSE_STOCK'] > 0, 1.0, 0.7  # Reduce score for unavailable styles
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
            st.error(f"Error calculating style performance metrics: {str(e)}")
            return pd.DataFrame()
    
    def determine_optimal_styles_per_store(self, style_metrics, efficiency_threshold=60):
        """
        Determine the optimal number of styles for each store based on efficiency metrics
        
        Parameters:
        -----------
        style_metrics : DataFrame
            Output from calculate_style_performance_metrics
        efficiency_threshold : float
            Minimum efficiency score to consider a style viable (default: 60)
            
        Returns:
        --------
        DataFrame with store-level recommendations
        """
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
                
                # Determine recommended style count based on store performance
                # Categorize stores by their overall performance
                store_total_sales = store_data['TOTAL_SALES'].sum()
                store_avg_efficiency = store_data['STYLE_EFFICIENCY_SCORE'].mean()
                
                # Store performance categories
                if store_avg_efficiency >= 70 and store_total_sales >= store_data['TOTAL_SALES'].quantile(0.75):
                    store_category = "High Performer"
                    recommended_styles = min(efficient_styles_count + 2, total_styles)  # Can handle more styles
                    max_recommended = min(20, total_styles)
                elif store_avg_efficiency >= 50 and store_total_sales >= store_data['TOTAL_SALES'].quantile(0.5):
                    store_category = "Good Performer"
                    recommended_styles = efficient_styles_count
                    max_recommended = min(15, total_styles)
                elif store_avg_efficiency >= 30:
                    store_category = "Average Performer"
                    recommended_styles = max(efficient_styles_count - 1, 1)  # Focus on fewer, better styles
                    max_recommended = min(10, total_styles)
                else:
                    store_category = "Needs Improvement"
                    recommended_styles = max(efficient_styles_count // 2, 1)  # Significant reduction needed
                    max_recommended = min(8, total_styles)
                
                # Ensure recommended styles is within reasonable bounds
                recommended_styles = max(1, min(recommended_styles, max_recommended))
                
                # Calculate potential impact
                if efficient_styles_count > 0:
                    # Top performing styles for this store
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
            st.error(f"Error determining optimal styles per store: {str(e)}")
            return pd.DataFrame()
    
    def get_style_recommendations_by_store(self, style_metrics, store_recommendations):
        """
        Get specific style recommendations for each store
        
        Returns:
        --------
        Dictionary with store as key and recommended styles as values
        """
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
    
    def create_efficiency_visualizations(self, style_metrics, store_recommendations):
        """Create comprehensive visualizations for style efficiency analysis"""
        
        if style_metrics.empty or store_recommendations.empty:
            st.warning("No data available for visualization")
            return
        
        # 1. Store Performance Overview
        st.markdown("#### 🏪 Store Performance Overview")
        
        # Store category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = store_recommendations['STORE_CATEGORY'].value_counts()
            fig_category = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Store Performance Categories",
                color_discrete_map={
                    'High Performer': '#00CC44',
                    'Good Performer': '#66CC00',
                    'Average Performer': '#FFAA00',
                    'Needs Improvement': '#FF4444'
                }
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
            # Efficiency score distribution
            fig_efficiency = px.histogram(
                store_recommendations,
                x='AVG_EFFICIENCY_SCORE',
                nbins=20,
                title="Distribution of Average Efficiency Scores",
                labels={'AVG_EFFICIENCY_SCORE': 'Average Efficiency Score', 'count': 'Number of Stores'}
            )
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # 2. Style Count Recommendations
        st.markdown("#### 📊 Style Count Recommendations by Store")
        
        # Prepare data for display
        display_data = store_recommendations.copy()
        if 'STORE_NAME' in display_data.columns:
            display_data['STORE_DISPLAY'] = display_data['STORE'] + ' - ' + display_data['STORE_NAME'].fillna('')
        else:
            display_data['STORE_DISPLAY'] = display_data['STORE']
        
        # Create comparison chart
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Current Total Styles',
            x=display_data['STORE_DISPLAY'],
            y=display_data['CURRENT_TOTAL_STYLES'],
            marker_color='lightgray'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Current Efficient Styles',
            x=display_data['STORE_DISPLAY'],
            y=display_data['CURRENT_EFFICIENT_STYLES'],
            marker_color='orange'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Recommended Styles',
            x=display_data['STORE_DISPLAY'],
            y=display_data['RECOMMENDED_STYLES'],
            marker_color='green'
        ))
        
        fig_comparison.update_layout(
            title='Style Count Analysis by Store',
            xaxis_title='Store',
            yaxis_title='Number of Styles',
            barmode='group',
            xaxis_tickangle=45
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # 3. Efficiency vs Sales Relationship
        st.markdown("#### 📈 Efficiency vs Sales Performance")
        
        fig_scatter = px.scatter(
            store_recommendations,
            x='AVG_EFFICIENCY_SCORE',
            y='TOTAL_SALES',
            size='RECOMMENDED_STYLES',
            color='STORE_CATEGORY',
            hover_data=['STORE', 'CURRENT_TOTAL_STYLES'],
            title='Store Efficiency vs Sales Performance',
            labels={
                'AVG_EFFICIENCY_SCORE': 'Average Efficiency Score',
                'TOTAL_SALES': 'Total Sales',
                'RECOMMENDED_STYLES': 'Recommended Styles'
            },
            color_discrete_map={
                'High Performer': '#00CC44',
                'Good Performer': '#66CC00',
                'Average Performer': '#FFAA00',
                'Needs Improvement': '#FF4444'
            }
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 4. Top Performing Styles Overall
        st.markdown("#### 🌟 Top Performing Styles Across All Stores")
        
        # Aggregate style performance across all stores
        overall_style_performance = style_metrics.groupby('STYLE').agg({
            'STYLE_EFFICIENCY_SCORE': 'mean',
            'TOTAL_SALES': 'sum',
            'SALES_VELOCITY': 'mean',
            'STOCK_TURNOVER': 'mean'
        }).reset_index()
        
        overall_style_performance = overall_style_performance.sort_values(
            'STYLE_EFFICIENCY_SCORE', ascending=False
        ).head(20)
        
        fig_top_styles = px.bar(
            overall_style_performance,
            x='STYLE',
            y='STYLE_EFFICIENCY_SCORE',
            title='Top 20 Performing Styles (Average Efficiency Score)',
            labels={'STYLE_EFFICIENCY_SCORE': 'Average Efficiency Score'}
        )
        
        fig_top_styles.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_top_styles, use_container_width=True)

def render_store_style_efficiency_analysis():
    """Main function to render the store style efficiency analysis interface"""
    
    st.markdown("## 🎯 Store Style Efficiency Analysis")
    st.markdown("""
    This analysis determines the optimal number of styles for each store based on:
    - **Sales Velocity**: How fast each style sells
    - **Stock Turnover**: Efficiency of inventory movement  
    - **Consistency**: Regular sales pattern
    - **Revenue Contribution**: Sales volume impact
    - **Warehouse Availability**: Stock availability for replenishment
    """)
    
    # Check if required data is available in session state
    required_data = ['sales', 'stock', 'warehouse', 'sku_master']
    missing_data = [data_type for data_type in required_data if data_type not in st.session_state or st.session_state[data_type].empty]
    
    if missing_data:
        st.warning(f"⚠️ Please upload the following data first: {', '.join(missing_data)}")
        st.info("💡 Upload your data files in the sidebar to proceed with the analysis.")
        return
    
    # Settings for analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_period = st.selectbox(
            "Analysis Period",
            [30, 60, 90, 120],
            index=2,
            format_func=lambda x: f"Last {x} Days"
        )
    
    with col2:
        efficiency_threshold = st.slider(
            "Efficiency Threshold",
            min_value=30,
            max_value=80,
            value=60,
            step=5,
            help="Minimum efficiency score to consider a style viable"
        )
    
    with col3:
        run_analysis = st.button("🚀 Run Style Efficiency Analysis", type="primary")
    
    if run_analysis:
        with st.spinner("Analyzing style efficiency across all stores..."):
            # Initialize analyzer
            analyzer = StoreStyleEfficiencyAnalyzer()
            
            # Load data
            analyzer.load_data(
                st.session_state['sales'],
                st.session_state['stock'], 
                st.session_state['warehouse'],
                st.session_state['sku_master'],
                st.session_state.get('style_master')
            )
            
            # Calculate style performance metrics
            style_metrics = analyzer.calculate_style_performance_metrics(analysis_period)
            
            if style_metrics.empty:
                st.error("❌ Could not calculate style metrics. Please check your data.")
                return
            
            # Determine optimal styles per store
            store_recommendations = analyzer.determine_optimal_styles_per_store(
                style_metrics, efficiency_threshold
            )
            
            if store_recommendations.empty:
                st.error("❌ Could not generate store recommendations. Please check your data.")
                return
            
            # Display key insights
            st.markdown("## 📊 Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_stores = len(store_recommendations)
                st.metric("Total Stores Analyzed", total_stores)
            
            with col2:
                avg_current_styles = store_recommendations['CURRENT_TOTAL_STYLES'].mean()
                st.metric("Avg Current Styles", f"{avg_current_styles:.1f}")
            
            with col3:
                avg_recommended_styles = store_recommendations['RECOMMENDED_STYLES'].mean()
                st.metric("Avg Recommended Styles", f"{avg_recommended_styles:.1f}")
            
            with col4:
                avg_efficiency = store_recommendations['AVG_EFFICIENCY_SCORE'].mean()
                st.metric("Avg Efficiency Score", f"{avg_efficiency:.1f}")
            
            # Create visualizations
            analyzer.create_efficiency_visualizations(style_metrics, store_recommendations)
            
            # Detailed recommendations table
            st.markdown("## 📋 Detailed Store Recommendations")
            
            # Prepare display data
            display_recommendations = store_recommendations.copy()
            
            # Add store names if available
            if 'STORE_NAME' in display_recommendations.columns:
                display_recommendations['Store'] = (
                    display_recommendations['STORE'] + ' - ' + 
                    display_recommendations['STORE_NAME'].fillna('')
                )
            else:
                display_recommendations['Store'] = display_recommendations['STORE']
            
            # Select and rename columns for display
            display_cols = {
                'Store': 'Store',
                'STORE_CATEGORY': 'Performance Category',
                'CURRENT_TOTAL_STYLES': 'Current Total Styles',
                'CURRENT_EFFICIENT_STYLES': 'Current Efficient Styles',
                'RECOMMENDED_STYLES': 'Recommended Styles',
                'EFFICIENCY_IMPROVEMENT': 'Change Required',
                'AVG_EFFICIENCY_SCORE': 'Avg Efficiency Score',
                'TOTAL_SALES': 'Total Sales',
                'EFFICIENT_SALES_PCT': 'Efficient Sales %',
                'PROJECTED_SALES_FOCUS_PCT': 'Focused Sales %'
            }
            
            final_display = display_recommendations[[col for col in display_cols.keys() if col in display_recommendations.columns]]
            final_display = final_display.rename(columns=display_cols)
            
            # Color coding for the table
            def color_efficiency_change(val):
                if val > 0:
                    return 'background-color: #d4edda'  # Light green for increase
                elif val < 0:
                    return 'background-color: #f8d7da'  # Light red for decrease
                else:
                    return 'background-color: #fff3cd'  # Light yellow for no change
            
            # Apply styling if Change Required column exists
            if 'Change Required' in final_display.columns:
                styled_df = final_display.style.applymap(
                    color_efficiency_change, 
                    subset=['Change Required']
                ).format({
                    'Avg Efficiency Score': '{:.1f}',
                    'Total Sales': '{:,.0f}',
                    'Efficient Sales %': '{:.1f}%',
                    'Focused Sales %': '{:.1f}%'
                })
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.dataframe(final_display, use_container_width=True)
            
            # Download recommendations
            csv_data = store_recommendations.to_csv(index=False)
            st.download_button(
                "📥 Download Store Recommendations",
                csv_data,
                "store_style_efficiency_recommendations.csv",
                "text/csv",
                key="download_store_recommendations"
            )
            
            # Store-specific style recommendations
            st.markdown("## 🎯 Specific Style Recommendations by Store")
            
            style_recommendations = analyzer.get_style_recommendations_by_store(
                style_metrics, store_recommendations
            )
            
            # Select store for detailed view
            selected_store = st.selectbox(
                "Select Store for Detailed Style Recommendations",
                list(style_recommendations.keys()),
                format_func=lambda x: f"{x} - {style_recommendations[x]['STORE_CATEGORY']}"
            )
            
            if selected_store and selected_store in style_recommendations:
                store_rec = style_recommendations[selected_store]
                
                st.markdown(f"### Store: {selected_store}")
                st.markdown(f"**Category**: {store_rec['STORE_CATEGORY']}")
                st.markdown(f"**Recommended Style Count**: {store_rec['RECOMMENDED_COUNT']}")
                
                # Display recommended styles
                styles_df = pd.DataFrame(store_rec['STYLES'])
                if not styles_df.empty:
                    styles_df = styles_df.round(2)
                    st.dataframe(
                        styles_df.rename(columns={
                            'STYLE': 'Style Code',
                            'EFFICIENCY_SCORE': 'Efficiency Score',
                            'SALES_VELOCITY': 'Daily Sales',
                            'STOCK_TURNOVER': 'Stock Turnover',
                            'TOTAL_SALES': 'Total Sales',
                            'CURRENT_STOCK': 'Current Stock'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No style recommendations available for this store.")
            
            # Store comparison insights
            st.markdown("## 🔍 Key Insights")
            
            # Performance insights
            high_performers = store_recommendations[
                store_recommendations['STORE_CATEGORY'] == 'High Performer'
            ]
            needs_improvement = store_recommendations[
                store_recommendations['STORE_CATEGORY'] == 'Needs Improvement'
            ]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not high_performers.empty:
                    st.success(f"🌟 **High Performers ({len(high_performers)} stores)**")
                    st.write(f"- Average efficiency score: {high_performers['AVG_EFFICIENCY_SCORE'].mean():.1f}")
                    st.write(f"- Average recommended styles: {high_performers['RECOMMENDED_STYLES'].mean():.1f}")
                    st.write(f"- Can handle {high_performers['RECOMMENDED_STYLES'].mean():.1f} styles efficiently")
            
            with col2:
                if not needs_improvement.empty:
                    st.warning(f"⚠️ **Needs Improvement ({len(needs_improvement)} stores)**")
                    st.write(f"- Average efficiency score: {needs_improvement['AVG_EFFICIENCY_SCORE'].mean():.1f}")
                    st.write(f"- Average recommended styles: {needs_improvement['RECOMMENDED_STYLES'].mean():.1f}")
                    st.write(f"- Should focus on {needs_improvement['RECOMMENDED_STYLES'].mean():.1f} top-performing styles")
            
            # Overall insights
            total_reduction = (store_recommendations['CURRENT_TOTAL_STYLES'].sum() - 
                             store_recommendations['RECOMMENDED_STYLES'].sum())
            
            if total_reduction > 0:
                st.info(f"💡 **Overall Recommendation**: Focus on {store_recommendations['RECOMMENDED_STYLES'].sum():.0f} styles across all stores (reduction of {total_reduction:.0f} styles) for improved efficiency.")
            else:
                st.info(f"💡 **Overall Recommendation**: Current style distribution is mostly optimal. Minor adjustments recommended for {abs(total_reduction):.0f} additional efficient styles.")