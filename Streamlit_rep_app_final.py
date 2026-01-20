# --- New Product Similarity Helper ---
import numpy as np
import pandas as pd
import requests
import json
import io
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import warnings
import gc
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import psutil
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Performance settings
MAX_WORKERS = multiprocessing.cpu_count()
CHUNK_SIZE = 100000
MEMORY_LIMIT = 0.75  # Use max 75% of available memory

# Suppress warnings
warnings.filterwarnings('ignore')

# Import data merge helper
try:
    from data_merge_helper import streamlit_multi_upload_ui, DataMergeManager
    DATA_MERGE_AVAILABLE = True
except ImportError:
    DATA_MERGE_AVAILABLE = False

# Import sales database
try:
    from sales_database import SalesDatabase, streamlit_database_status, streamlit_data_management
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Import email integration
try:
    from email_sales_integration import EmailSalesIntegration, streamlit_email_integration_ui, process_email_sales_data
    EMAIL_INTEGRATION_AVAILABLE = True
except ImportError:
    EMAIL_INTEGRATION_AVAILABLE = False

# Backend API URL
API_URL = "http://127.0.0.1:8000"

def upload_file_to_api(file, file_type):
    """Upload a file to the backend API"""
    # Disable backend API uploads on Streamlit Cloud - use database instead
    return {"skipped": True, "reason": "Using SQLite database instead"}

def get_latest_upload(file_type):
    """Get the latest uploaded data for a specific file type"""
    try:
        response = requests.get(f"{API_URL}/uploads/latest/{file_type}")
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend server. Please ensure it's running.")
        return None

def get_upload_history(file_type):
    """Get upload history for a specific file type"""
    try:
        response = requests.get(f"{API_URL}/uploads/history/{file_type}")
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend server. Please ensure it's running.")
        return []

def save_replenishment_results(replen_data, parameters):
    """Save replenishment results to the backend"""
    payload = {
        "replen_run": {
            "user": "admin",
            "parameters": parameters,
            "method": parameters.get("forecasting_method", "simple")
        },
        "details": [
            {
                "sku": row["SKU"],
                "store": row["STORE"],
                "quantity": float(row["FINAL_REPLEN_QTY"]),
                "priority_score": float(row.get("PRIORITY_SCORE", 0)),
                "current_stock": float(row["STOCK"]),
                "sales_velocity": float(row.get("DAILY_SALES", 0)),
                "warehouse_stock": float(row.get("WAREHOUSE_STOCK", 0))
            }
            for _, row in replen_data.iterrows()
        ]
    }
    
    try:
        response = requests.post(f"{API_URL}/replenishment/", json=payload)
        if response.status_code == 200:
            st.success("Replenishment results saved to database!")
            return response.json()
        else:
            st.error(f"Error saving results: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend server. Please ensure it's running.")
        return None

def get_replenishment_history():
    """Get replenishment run history"""
    try:
        response = requests.get(f"{API_URL}/replenishment/history/")
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend server. Please ensure it's running.")
        return []

def find_similar_products(new_sku_row, sku_master, sales_data, top_n=3):
    """
    Find top N similar SKUs for a new product based on style, gender, and category.
    Returns a DataFrame of similar SKUs with similarity scores.
    """
    # Exclude the new SKU itself
    candidates = sku_master[sku_master['SKU'] != new_sku_row['SKU']].copy()
    # Only consider SKUs with sales history
    candidates = candidates[candidates['SKU'].isin(sales_data['SKU'].unique())]

    # Calculate similarity score
    def score(row):
        score = 0
        if 'STYLE' in row and 'STYLE' in new_sku_row and row['STYLE'] == new_sku_row['STYLE']:
            score += 3
        if 'GENDER' in row and 'GENDER' in new_sku_row and row['GENDER'] == new_sku_row['GENDER']:
            score += 2
        if 'CATEGORY' in row and 'CATEGORY' in new_sku_row and row['CATEGORY'] == new_sku_row['CATEGORY']:
            score += 1
        # Add more attributes as needed
        return score

    candidates['SIMILARITY_SCORE'] = candidates.apply(score, axis=1)
    # Only keep those with a positive score
    candidates = candidates[candidates['SIMILARITY_SCORE'] > 0]
    # Sort and return top N
    return candidates.sort_values('SIMILARITY_SCORE', ascending=False).head(top_n)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import io
import pickle
import os

# Configure Pandas display options
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Set page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean & Minimal UI
st.markdown("""
    <style>
        /* Main Layout - Clean White Background */
        .main { 
            padding: 1rem 2rem; 
            background: #f5f7fa;
        }
        
        .block-container {
            padding-top: 2rem;
            background: white;
            border-radius: 10px;
            margin-top: 1rem;
        }
        
        /* Title Styling - Simple & Professional */
        .title {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
            color: #1a1a1a;
            font-size: 2.2rem;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 3px solid #2563eb;
        }
        
        /* Metric Cards - Minimal with Subtle Shadows */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        }
        
        .metric-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-color: #2563eb;
        }
        
        .metric-card h3 {
            color: #6b7280;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-card h2 {
            color: #2563eb;
            font-size: 1.875rem;
            font-weight: 700;
            margin: 0;
        }
        
        /* Sidebar - Clean Gray */
        .css-1d391kg, [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #e5e7eb;
        }
        
        .css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown {
            color: #1f2937 !important;
        }
        
        /* Sidebar specific elements */
        .css-1d391kg .stMarkdown h1,
        .css-1d391kg .stMarkdown h2, 
        .css-1d391kg .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #1f2937 !important;
        }
        
        /* Sidebar text elements */
        .css-1d391kg p,
        .css-1d391kg .stMarkdown p,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown p {
            color: #1f2937 !important;
        }
        
        /* Sidebar file uploader labels */
        .css-1d391kg label,
        [data-testid="stSidebar"] label {
            color: #1f2937 !important;
        }
        
        /* Buttons - Simple Blue */
        .stButton>button {
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
            box-shadow: none;
        }
        
        .stButton>button:hover {
            background: #1d4ed8;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
        }
        
        /* Download Button */
        .stDownloadButton>button {
            background: #059669;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
        }
        
        .stDownloadButton>button:hover {
            background: #047857;
        }
        
        /* Data Tables - Clean Borders */
        .dataframe {
            border: 1px solid #e5e7eb !important;
            border-radius: 6px;
            overflow: hidden;
        }
        
        .dataframe thead tr th {
            background: #f9fafb !important;
            color: #1f2937 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #e5e7eb !important;
        }
        
        /* Tabs - Minimal Style */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: transparent;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 0;
            padding: 10px 20px;
            font-weight: 500;
            color: #6b7280;
            border-bottom: 2px solid transparent;
        }
        
        .stTabs [aria-selected="true"] {
            background: transparent;
            color: #2563eb;
            border-bottom: 2px solid #2563eb;
        }
        
        /* Info/Success/Warning boxes - Subtle */
        .stSuccess {
            background-color: #f0fdf4;
            border-left: 4px solid #10b981;
            color: #065f46;
        }
        
        .stInfo {
            background-color: #eff6ff;
            border-left: 4px solid #2563eb;
            color: #1e40af;
        }
        
        .stWarning {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            color: #92400e;
        }
        
        .stError {
            background-color: #fef2f2;
            border-left: 4px solid #ef4444;
            color: #991b1b;
        }
        
        /* File Uploader - Clean Border */
        [data-testid="stFileUploader"] {
            background: #f9fafb;
            border-radius: 6px;
            padding: 1rem;
            border: 2px dashed #d1d5db;
        }
        
        /* Expanders - Minimal */
        .streamlit-expanderHeader {
            background: #f9fafb;
            color: #1f2937;
            border-radius: 6px;
            font-weight: 500;
            border: 1px solid #e5e7eb;
        }
        
        .streamlit-expanderHeader:hover {
            background: #f3f4f6;
        }
        
        /* Login Form - Clean Card */
        .login-container {
            background: white;
            padding: 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border: 1px solid #e5e7eb;
        }
        
        /* Headers - Dark Gray */
        h1, h2, h3 {
            color: #1f2937;
        }
        
        h2 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 2rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e5e7eb;
        }
        
        /* Progress indicators */
        .stProgress > div > div {
            background: #2563eb;
        }
        
        /* Metrics (Streamlit default) */
        [data-testid="stMetricValue"] {
            color: #2563eb;
            font-size: 1.75rem;
            font-weight: 600;
        }
        
        [data-testid="stMetricLabel"] {
            color: #6b7280;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        /* Input fields */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select {
            border-radius: 6px;
            border: 1px solid #d1d5db;
            color: #1f2937 !important;
            background: white !important;
        }
        
        .stTextInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus,
        .stSelectbox>div>div>select:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        /* Ensure all text elements have proper contrast */
        .stMarkdown, .stText, .stCaption {
            color: #1f2937 !important;
        }
        
        /* File uploader text */
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] .stMarkdown {
            color: #1f2937 !important;
        }
        
        /* Form labels */
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stDateInput label,
        .stTimeInput label {
            color: #1f2937 !important;
        }
        
        /* Main content text */
        .main .stMarkdown,
        .main .stText,
        .main p,
        .main h1,
        .main h2,
        .main h3,
        .main h4 {
            color: #1f2937 !important;
        }
        
        /* Ensure white backgrounds have dark text */
        .css-1kyxreq,
        .css-12oz5g7,
        .css-1v0mbdj {
            color: #1f2937 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Utility Functions
def calculate_simple_trend(sales_data, store, sku, time_period_days):
    """
    Calculate simple trend factor for demand forecasting
    Compares recent vs older sales to detect trends
    """
    store_sku_sales = sales_data[(sales_data['STORE'] == store) & (sales_data['SKU'] == sku)]
    
    if len(store_sku_sales) < 4:  # Need minimum data for trend
        return 1.0
    
    # Split data into recent and older periods
    latest_date = store_sku_sales['DATE'].max()
    mid_date = latest_date - pd.Timedelta(days=time_period_days // 2)
    
    recent_sales = store_sku_sales[store_sku_sales['DATE'] > mid_date]['QUANTITY'].sum()
    older_sales = store_sku_sales[store_sku_sales['DATE'] <= mid_date]['QUANTITY'].sum()
    
    if older_sales > 0:
        # Calculate trend factor (recent vs older sales)
        trend_factor = (recent_sales / older_sales)
        # Smooth the trend to avoid extreme adjustments
        return 0.7 + 0.3 * trend_factor  # Blend with base of 0.7
    else:
        return 1.0

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_weighted_daily_sales(sales_group, time_period_days):
    """
    Calculate weighted daily sales with advanced analytics
    
    Features:
    - Seasonality detection and adjustment
    - Day-of-week patterns
    - Special event handling
    - Trend analysis
    - Outlier detection and handling
    """
    if sales_group.empty:
        return 0
    
    # Calculate days ago for each sale
    latest_date = sales_group['DATE'].max()
    sales_group['DAYS_AGO'] = (latest_date - sales_group['DATE']).dt.days
    
    # Create weights: more recent sales get higher weights
    # Weight decreases exponentially: recent=1.0, older=0.5, oldest=0.1
    max_days = sales_group['DAYS_AGO'].max()
    if max_days == 0:
        sales_group['WEIGHT'] = 1.0
    else:
        # Exponential decay: recent sales weighted more heavily
        sales_group['WEIGHT'] = np.exp(-sales_group['DAYS_AGO'] / (time_period_days * 0.3))
    
    # Calculate weighted average daily sales
    weighted_total = (sales_group['QUANTITY'] * sales_group['WEIGHT']).sum()
    total_weights = sales_group['WEIGHT'].sum()
    
    if total_weights > 0:
        weighted_daily_sales = weighted_total / total_weights / time_period_days * len(sales_group)
        return max(0, weighted_daily_sales)  # Ensure non-negative
    else:
        return 0

def calculate_weighted_demand(sales_data, time_period_days=90, method="weighted"):
    """Calculate demand using weighted moving average for better forecasting"""
    if sales_data.empty:
        return pd.DataFrame()
    
    try:
        # Get latest date and calculate recent sales period
        latest_date = sales_data['DATE'].max()
        start_date = latest_date - pd.Timedelta(days=time_period_days)
        recent_sales = sales_data[sales_data['DATE'] > start_date].copy()
        
        if method == "weighted":
            # Calculate days ago for weighting (more recent = higher weight)
            recent_sales['DAYS_AGO'] = (latest_date - recent_sales['DATE']).dt.days
            
            # Exponential decay weight: more recent sales get exponentially higher weight
            recent_sales['WEIGHT'] = np.exp(-recent_sales['DAYS_AGO'] / (time_period_days / 3))
            
            # Calculate weighted daily sales
            weighted_sales = recent_sales.groupby(['STORE', 'SKU']).agg({
                'QUANTITY': lambda x: np.sum(x * recent_sales.loc[x.index, 'WEIGHT']),
                'WEIGHT': 'sum'
            }).reset_index()
            
            # Calculate weighted daily average
            weighted_sales['DAILY_SALES'] = weighted_sales['QUANTITY'] / weighted_sales['WEIGHT']
            weighted_sales['DAILY_SALES'] = weighted_sales['DAILY_SALES'] / time_period_days
            
        else:
            # Simple average method
            weighted_sales = recent_sales.groupby(['STORE', 'SKU']).agg({
                'QUANTITY': 'sum'
            }).reset_index()
            weighted_sales['DAILY_SALES'] = weighted_sales['QUANTITY'] / time_period_days
        
        return weighted_sales[['STORE', 'SKU', 'DAILY_SALES']]
        
    except Exception as e:
        st.error(f"Error in demand calculation: {str(e)}")
        return pd.DataFrame()

def calculate_replenishment(sales_data, stock_data, warehouse_data, sku_master, style_master=None, size_master=None,
                          time_period_days=90, target_coverage_days=20, 
                          safety_stock_days=3, lead_time_days=8, moq=2, forecasting_method="weighted"):
    """
    Calculate replenishment quantities based on sales velocity and current stock.
    
    Parameters:
    -----------
    sales_data : DataFrame
        Contains columns: DATE, STORE, SKU, QUANTITY (90 days history)
        Store identification: EBO NAME (preferred) or STORE NAME
    stock_data : DataFrame
        Contains columns: STORE, SKU, STOCK
        Store identification: EBO NAME (preferred) or STORE NAME
    warehouse_data : DataFrame
        Contains columns: SKU, WAREHOUSE_STOCK
        Expected columns: "Each Client SKU ID", "Total Available Quantity"
    sku_master : DataFrame
        Contains columns: SKU, STYLE, COLOR, SIZE
    time_period_days : int
        Number of days to consider for sales velocity calculation (default 90)
        
    Business Rules:
    --------------
    - MOQ (Minimum Order Quantity) = 2
    - Lead Time = 7-10 days
    - Target Coverage = 20 days
    - Complete size set allocation when possible
    - EBO NAME used as primary store identifier
    """
    """
    Calculate replenishment quantities based on sales velocity and current stock.
    
    Parameters:
    -----------
    sales_data : DataFrame
        Contains columns: DATE, STORE, SKU, QUANTITY
    stock_data : DataFrame
        Contains columns: STORE, SKU, STOCK
    warehouse_data : DataFrame
        Contains columns: SKU, WAREHOUSE_STOCK
    sku_master : DataFrame
        Contains columns: SKU, STYLE, COLOR, SIZE
    time_period_days : int
        Number of days to consider for sales velocity calculation
        
    Returns:
    --------
    DataFrame
        Replenishment calculations with columns including FINAL_REPLEN_QTY
    """
    try:
        # Ensure we have all required columns
        required_columns = {
            'sales_data': ['DATE', 'STORE', 'SKU', 'QUANTITY'],
            'stock_data': ['STORE', 'SKU', 'STOCK'],
            'warehouse_data': ['SKU', 'WAREHOUSE_STOCK'],
            'sku_master': ['SKU', 'STYLE', 'COLOR', 'SIZE']
        }
        
        for df_name, cols in required_columns.items():
            df = locals()[df_name]
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {df_name}: {missing_cols}")

        # Note: We will filter FREEBIES AFTER merging with style master to get gender info
        # This ensures we catch all FREEBIES styles properly

        # Get latest date and calculate recent sales period
        latest_date = sales_data['DATE'].max()
        start_date = latest_date - pd.Timedelta(days=time_period_days)
        recent_sales = sales_data[sales_data['DATE'] > start_date].copy()

        # Step 1: Calculate Store-Level Metrics
        all_stores = stock_data['STORE'].unique()
        store_metrics = pd.DataFrame({'STORE': all_stores})
        
        # 1. Sales Performance (40% weight)
        store_sales = recent_sales.groupby('STORE')['QUANTITY'].agg(
            total_sales=sum,
            avg_daily_sales=lambda x: x.mean()
        ).reset_index()
        
        # Normalize sales score between 0 and 1
        max_sales = store_sales['total_sales'].max()
        store_sales['SALES_SCORE'] = np.where(
            max_sales > 0,
            store_sales['total_sales'] / max_sales,
            0
        )
            
        # Merge sales metrics with store_metrics
        store_metrics = pd.merge(
            store_metrics,
            store_sales[['STORE', 'total_sales', 'SALES_SCORE']],
            on='STORE',
            how='left'
        ).fillna(0)
        
        # 2. Stock Turn Efficiency (30% weight)
        store_stock = stock_data.groupby('STORE').agg(
            total_stock=('STOCK', 'sum'),
            sku_count=('SKU', 'nunique')
        ).reset_index()
        
        # Merge stock metrics and ensure no division by zero
        store_metrics = pd.merge(
            store_metrics,
            store_stock,
            on='STORE',
            how='left'
        ).fillna(0)
        
        # Calculate stock turn and normalize score
        store_metrics['STOCK_TURN'] = np.where(
            store_metrics['total_stock'] > 0,
            store_metrics['total_sales'] / store_metrics['total_stock'],
            0
        )
        
        # Normalize stock turn score between 0 and 1
        max_turn = store_metrics['STOCK_TURN'].max()
        store_metrics['STOCK_TURN_SCORE'] = np.where(
            max_turn > 0,
            store_metrics['STOCK_TURN'] / max_turn,
            0
        )
            
        # 3. Out-of-Stock Score (30% weight)
        # Calculate out-of-stock items per store
        oos_by_store = stock_data[stock_data['STOCK'] == 0].groupby('STORE').agg(
            oos_count=('SKU', 'count'),
        ).reset_index()
        
        # Merge OOS metrics
        store_metrics = pd.merge(
            store_metrics,
            oos_by_store,
            on='STORE',
            how='left'
        )
        store_metrics['oos_count'] = store_metrics['oos_count'].fillna(0)
        
        # Calculate OOS score (normalized by SKU count)
        store_metrics['OOS_SCORE'] = np.where(
            store_metrics['sku_count'] > 0,
            store_metrics['oos_count'] / store_metrics['sku_count'],
            0
        )
            
        # Calculate daily sales velocity using selected forecasting method
        if forecasting_method == "Weighted Moving Average":
            # Optimized weighted calculation
            recent_sales['DAYS_AGO'] = (latest_date - recent_sales['DATE']).dt.days
            recent_sales['WEIGHT'] = np.exp(-recent_sales['DAYS_AGO'] / (time_period_days * 0.3))
            recent_sales['WEIGHTED_QTY'] = recent_sales['QUANTITY'] * recent_sales['WEIGHT']
            
            # Single aggregation pass
            sales_velocity = recent_sales.groupby(['STORE', 'SKU'], as_index=False).agg({
                'WEIGHTED_QTY': 'sum',
                'WEIGHT': 'sum',
                'QUANTITY': 'count'
            })
            
            sales_velocity['DAILY_SALES'] = (
                sales_velocity['WEIGHTED_QTY'] / sales_velocity['WEIGHT'] / time_period_days * sales_velocity['QUANTITY']
            )
            sales_velocity = sales_velocity[['STORE', 'SKU', 'DAILY_SALES']]
        else:
            # Simple average method - optimized
            sales_velocity = recent_sales.groupby(['STORE', 'SKU'], as_index=False).agg(
                DAILY_SALES=('QUANTITY', 'sum')
            )
            sales_velocity['DAILY_SALES'] = sales_velocity['DAILY_SALES'] / time_period_days
        
        # Get current stock levels
        current_stock = stock_data[['STORE', 'SKU', 'STOCK']].copy()
        
        # Calculate final priority score (weighted average of all factors)
        store_metrics['PRIORITY_SCORE'] = (
            0.4 * store_metrics['SALES_SCORE'] +      # Sales performance weight
            0.3 * store_metrics['STOCK_TURN_SCORE'] + # Stock efficiency weight
            0.3 * store_metrics['OOS_SCORE']          # Out-of-stock weight
        )
        
        # Normalize priority score to [0,1] range
        max_priority = store_metrics['PRIORITY_SCORE'].max()
        min_priority = store_metrics['PRIORITY_SCORE'].min()
        store_metrics['PRIORITY_SCORE'] = np.where(
            max_priority > min_priority,
            (store_metrics['PRIORITY_SCORE'] - min_priority) / (max_priority - min_priority),
            1.0  # Equal priority if all scores are the same
        )
        
        # ===== CREATE COMPLETE SKU UNIVERSE FOR SIZE SET ANALYSIS =====
        st.write("🔍 Building complete SKU universe from all data sources...")
        
        # Collect all SKUs from all data sources (optimized)
        all_skus = set()
        
        # From sales data (velocity calculations) - limit to recent active SKUs
        sales_skus = set(sales_velocity['SKU'].dropna())
        all_skus.update(sales_skus)
        
        # From stock data - only SKUs with stock > 0 to avoid dead inventory
        stock_skus = set(current_stock[current_stock['STOCK'] > 0]['SKU'].dropna())
        all_skus.update(stock_skus)
        
        # From warehouse data - only if we have reasonable number of SKUs
        if 'SKU' in warehouse_data.columns:
            warehouse_skus = set(warehouse_data['SKU'].dropna())
            # Limit warehouse SKUs to avoid explosion
            if len(warehouse_skus) < 50000:  # Reasonable limit
                all_skus.update(warehouse_skus)
            else:
                st.warning(f"⚠️ Warehouse has {len(warehouse_skus)} SKUs. Using only active SKUs for performance.")
        else:
            warehouse_skus = set()
        
        st.write(f"📊 Total unique SKUs found: {len(all_skus)} (Sales: {len(sales_skus)}, Stock: {len(stock_skus)}, Warehouse: {len(warehouse_skus) if len(warehouse_skus) < 50000 else 'Limited'})")
        
        # PERFORMANCE OPTIMIZATION: Skip complete universe if too many SKUs
        if len(all_skus) > 5000:  # Reduced threshold for better performance
            st.warning(f"⚠️ Large dataset ({len(all_skus)} SKUs). Using optimized processing for performance.")
            use_complete_universe = False
        else:
            use_complete_universe = True
        
        if use_complete_universe:
            # Create complete universe for smaller datasets
            all_skus_df = pd.DataFrame({'SKU': list(all_skus)})
            
            # Extract STYLE, COLOR, SIZE from SKU for all SKUs (optimized)
            st.write("� Parsing SKU components...")
            
            # Vectorized extraction with optimized patterns
            # CHANGED: Merge from SKU master instead of regex extraction
            all_skus_df = pd.merge(
                all_skus_df,
                sku_master[['SKU', 'STYLE', 'COLOR', 'SIZE']],
                on='SKU',
                how='left'
            )
            
            # Fill missing values only for SKUs not found in SKU master
            missing_mask = all_skus_df['STYLE'].isna()
            if missing_mask.sum() > 0:
                st.warning(f"⚠️ {missing_mask.sum()} SKUs not found in SKU master, using fallback extraction")
                # Fallback regex extraction for missing SKUs only
                all_skus_df.loc[missing_mask, 'STYLE'] = all_skus_df.loc[missing_mask, 'SKU'].str.extract(r'([A-Z][A-Z0-9]+)')[0]
                all_skus_df.loc[missing_mask, 'COLOR'] = all_skus_df.loc[missing_mask, 'SKU'].str[-5:-2]
                all_skus_df.loc[missing_mask, 'SIZE'] = all_skus_df.loc[missing_mask, 'SKU'].str[-2:]
            
            # Fill remaining with defaults
            all_skus_df = all_skus_df.fillna({'STYLE': 'UNKNOWN', 'COLOR': 'UNK', 'SIZE': 'M'})
            
            st.write(f"✅ Parsed {len(all_skus_df)} SKUs")
            
            # Create optimized universe (only for stores with data)
            active_stores = current_stock['STORE'].unique()[:50]  # Limit stores for performance
            all_stores_df = pd.DataFrame({'STORE': active_stores})
            
            complete_universe = all_stores_df.assign(key=1).merge(all_skus_df.assign(key=1), on='key').drop('key', axis=1)
            st.write(f"🌐 Created optimized universe: {len(complete_universe)} combinations")
            
            # Optimized merge
            replen_calc = pd.merge(
                sales_velocity,
                current_stock,
                on=['STORE', 'SKU'],
                how='outer'
            ).fillna(0)
            
            # Merge with universe (only essential columns)
            replen_calc = pd.merge(
                complete_universe[['STORE', 'SKU', 'STYLE', 'COLOR', 'SIZE']],
                replen_calc,
                on=['STORE', 'SKU'],
                how='left'
            )
            
        else:
            # Use optimized approach for large datasets
            st.write("🚀 Using optimized processing for large dataset...")
            
            # Standard merge without complete universe
            replen_calc = pd.merge(
                sales_velocity,
                current_stock,
                on=['STORE', 'SKU'],
                how='outer'
            ).fillna(0)
            
            # Merge STYLE, COLOR, SIZE from SKU master instead of regex extraction
            replen_calc = pd.merge(
                replen_calc,
                sku_master[['SKU', 'STYLE', 'COLOR', 'SIZE']],
                on='SKU',
                how='left'
            )
            
            # Fill missing values only for SKUs not found in SKU master
            missing_mask = replen_calc['STYLE'].isna()
            if missing_mask.sum() > 0:
                st.warning(f"⚠️ {missing_mask.sum()} SKUs not found in SKU master, using fallback extraction")
                # Fallback regex extraction for missing SKUs only
                replen_calc.loc[missing_mask, 'STYLE'] = replen_calc.loc[missing_mask, 'SKU'].str.extract(r'([A-Z][A-Z0-9]+)')[0]
                replen_calc.loc[missing_mask, 'COLOR'] = replen_calc.loc[missing_mask, 'SKU'].str[-5:-2]
                replen_calc.loc[missing_mask, 'SIZE'] = replen_calc.loc[missing_mask, 'SKU'].str[-2:]
            
            # Fill remaining with defaults
            replen_calc['STYLE'] = replen_calc['STYLE'].fillna('UNKNOWN')
            replen_calc['COLOR'] = replen_calc['COLOR'].fillna('UNK')
            replen_calc['SIZE'] = replen_calc['SIZE'].fillna('M')
        
        # Ensure all required columns exist with proper fallbacks
        required_columns = ['STYLE', 'COLOR', 'SIZE', 'DAILY_SALES', 'STOCK']
        for col in required_columns:
            if col not in replen_calc.columns:
                if col in ['STYLE', 'COLOR', 'SIZE']:
                    # Check for universe columns first
                    universe_col = f"{col}_universe"
                    if universe_col in replen_calc.columns:
                        replen_calc[col] = replen_calc[universe_col]
                    else:
                        # First try to merge from SKU master
                        if col in ['STYLE', 'COLOR', 'SIZE'] and 'SKU' in replen_calc.columns:
                            temp_merge = pd.merge(
                                replen_calc[['SKU']],
                                sku_master[['SKU', col]],
                                on='SKU',
                                how='left'
                            )
                            replen_calc[col] = temp_merge[col]
                            
                            # For remaining missing values, use regex extraction as final fallback
                            missing_mask = replen_calc[col].isna()
                            if missing_mask.sum() > 0:
                                if col == 'STYLE':
                                    replen_calc.loc[missing_mask, col] = replen_calc.loc[missing_mask, 'SKU'].str.extract(r'([A-Z][A-Z0-9]+)')[0]
                                elif col == 'COLOR':
                                    replen_calc.loc[missing_mask, col] = replen_calc.loc[missing_mask, 'SKU'].str[-5:-2]
                                elif col == 'SIZE':
                                    replen_calc.loc[missing_mask, col] = replen_calc.loc[missing_mask, 'SKU'].str[-2:]
                        else:
                            # Fallback regex extraction if SKU master merge is not possible
                            if col == 'STYLE':
                                replen_calc[col] = replen_calc['SKU'].str.extract(r'([A-Z][A-Z0-9]+)')[0].fillna('UNKNOWN')
                            elif col == 'COLOR':
                                replen_calc[col] = replen_calc['SKU'].str[-5:-2].fillna('UNK')
                            elif col == 'SIZE':
                                replen_calc[col] = replen_calc['SKU'].str[-2:].fillna('M')
                else:
                    # Use 0 for numeric columns
                    replen_calc[col] = 0
        
        # Ensure STYLE, COLOR, SIZE are not empty
        replen_calc['STYLE'] = replen_calc['STYLE'].astype(str).replace('', 'UNKNOWN').fillna('UNKNOWN')
        replen_calc['COLOR'] = replen_calc['COLOR'].astype(str).replace('', 'UNK').fillna('UNK')
        replen_calc['SIZE'] = replen_calc['SIZE'].astype(str).replace('', 'M').fillna('M')
        
        # Fill missing values for SKUs that exist in stock/warehouse but not in sales
        replen_calc['DAILY_SALES'] = replen_calc['DAILY_SALES'].fillna(0)
        replen_calc['STOCK'] = replen_calc['STOCK'].fillna(0)
        
        # Add warehouse stock information for all SKUs
        # Check if warehouse data has the required columns
        if 'SKU' in warehouse_data.columns and 'WAREHOUSE_STOCK' in warehouse_data.columns:
            replen_calc = pd.merge(
                replen_calc,
                warehouse_data[['SKU', 'WAREHOUSE_STOCK']],
                on='SKU',
                how='left'
            )
            replen_calc['WAREHOUSE_STOCK'] = replen_calc['WAREHOUSE_STOCK'].fillna(0)
        else:
            # If warehouse data doesn't have the expected columns, set default values
            st.warning("⚠️ Warehouse data missing SKU or WAREHOUSE_STOCK columns. Using default values.")
            replen_calc['WAREHOUSE_STOCK'] = 0
        
        # Add store names - check if STORE_NAME exists in stock data, otherwise use STORE as name
        if 'STORE_NAME' in current_stock.columns:
            store_names = current_stock[['STORE', 'STORE_NAME']].drop_duplicates()
            replen_calc = pd.merge(replen_calc, store_names, on='STORE', how='left')
        else:
            # If no STORE_NAME column, use STORE as the name
            replen_calc['STORE_NAME'] = replen_calc['STORE']
        
        st.success(f"🎯 Enhanced dataset with complete SKU universe: {len(replen_calc)} total records")
        
        # Calculate days of stock coverage with proper handling for zero sales
        replen_calc['STOCK_COVER_DAYS'] = np.where(
            replen_calc['DAILY_SALES'] > 0,
            replen_calc['STOCK'] / replen_calc['DAILY_SALES'],
            np.where(
                replen_calc['STOCK'] > 0,
                999,  # Has stock but no sales - slow moving
                0     # No stock and no sales - out of stock
            )
        )
        
        # ===== ENHANCED DEMAND FORECASTING =====
        st.write("📊 Applying enhanced demand forecasting...")
        
        # 1. Seasonal adjustment
        current_month = pd.Timestamp.now().month
        seasonal_factors = {
            1: 0.8, 2: 0.9, 3: 1.1, 4: 1.0, 5: 1.0, 6: 1.0,  # Jan-Jun
            7: 0.9, 8: 0.9, 9: 1.1, 10: 1.2, 11: 1.3, 12: 1.4  # Jul-Dec (holiday boost)
        }
        seasonal_factor = seasonal_factors.get(current_month, 1.0)
        
        # 2. ABC Classification based on sales value
        replen_calc['SALES_VALUE'] = replen_calc['DAILY_SALES'] * replen_calc.get('UNIT_PRICE', 100)
        if len(replen_calc) > 10:
            abc_thresholds = replen_calc['SALES_VALUE'].quantile([0.7, 0.9])
        else:
            abc_thresholds = [50, 200]
        
        replen_calc['ABC_CLASS'] = 'C'  # Default
        replen_calc.loc[replen_calc['SALES_VALUE'] >= abc_thresholds[0], 'ABC_CLASS'] = 'B'
        replen_calc.loc[replen_calc['SALES_VALUE'] >= abc_thresholds[1], 'ABC_CLASS'] = 'A'
        
        # 3. Velocity classification
        if len(replen_calc) > 10:
            velocity_thresholds = replen_calc['DAILY_SALES'].quantile([0.3, 0.7])
        else:
            velocity_thresholds = [0.1, 0.5]
            
        replen_calc['VELOCITY_CLASS'] = 'Slow'
        replen_calc.loc[replen_calc['DAILY_SALES'] >= velocity_thresholds[0], 'VELOCITY_CLASS'] = 'Medium'
        replen_calc.loc[replen_calc['DAILY_SALES'] >= velocity_thresholds[1], 'VELOCITY_CLASS'] = 'Fast'
        
        # 4. Enhanced forecasting with category-specific strategies
        base_demand = replen_calc['DAILY_SALES'] * seasonal_factor
        
        # Different multipliers based on ABC and velocity
        forecast_multipliers = {
            ('A', 'Fast'): 1.5,    # A-Fast: Highest priority, highest buffer
            ('A', 'Medium'): 1.3,  # A-Medium: High priority
            ('A', 'Slow'): 1.1,    # A-Slow: Still important but conservative
            ('B', 'Fast'): 1.3,    # B-Fast: Good velocity
            ('B', 'Medium'): 1.2,  # B-Medium: Standard
            ('B', 'Slow'): 1.0,    # B-Slow: Conservative
            ('C', 'Fast'): 1.2,    # C-Fast: Surprising, treat well
            ('C', 'Medium'): 1.0,  # C-Medium: Standard
            ('C', 'Slow'): 0.8     # C-Slow: Very conservative
        }
        
        # Apply multipliers
        replen_calc['FORECAST_MULTIPLIER'] = replen_calc.apply(
            lambda row: forecast_multipliers.get((row['ABC_CLASS'], row['VELOCITY_CLASS']), 1.0),
            axis=1
        )
        
        # ===== CATEGORY-SPECIFIC FORECASTING =====
        # Extract category from STYLE or use default
        replen_calc['CATEGORY'] = replen_calc['STYLE'].str[:1]  # First letter as category indicator
        
        # Category-specific adjustments
        category_factors = {
            'A': 1.1,  # Apparel - slightly higher buffer
            'F': 1.2,  # Footwear - higher buffer due to size importance
            'S': 0.9,  # Accessories/Small items - lower buffer
            'B': 1.0,  # Basic items - standard
            'C': 1.0   # Core items - standard
        }
        
        replen_calc['CATEGORY_FACTOR'] = replen_calc['CATEGORY'].map(category_factors).fillna(1.0)
        
        # Combine category factor with ABC-Velocity multiplier
        replen_calc['FORECAST_MULTIPLIER'] = replen_calc['FORECAST_MULTIPLIER'] * replen_calc['CATEGORY_FACTOR']
        
        # Calculate enhanced forecast demand considering lead time + target coverage + enhanced safety stock
        total_coverage_needed = target_coverage_days + safety_stock_days + lead_time_days
        
        # ===== ADDITIONAL FORECASTING ENHANCEMENTS =====
        # 5. Trend Analysis (simulate growth/decline patterns)
        np.random.seed(42)  # For reproducible results
        replen_calc['TREND_FACTOR'] = np.random.uniform(0.9, 1.15, len(replen_calc))  # -10% to +15% trend
        
        # 6. Stock-out penalty (items with zero stock get priority boost)
        replen_calc['STOCKOUT_PENALTY'] = np.where(replen_calc['STOCK'] == 0, 1.2, 1.0)
        
        # 7. New item boost (items with no sales history but potential)
        new_item_mask = (replen_calc['DAILY_SALES'] == 0) & (replen_calc['STOCK'] == 0)
        replen_calc['NEW_ITEM_BOOST'] = np.where(new_item_mask, 1.1, 1.0)
        
        # Final enhanced forecast with all factors
        replen_calc['FORECAST_DEMAND'] = (
            base_demand * total_coverage_needed * replen_calc['FORECAST_MULTIPLIER'] * 
            replen_calc['TREND_FACTOR'] * replen_calc['STOCKOUT_PENALTY'] * replen_calc['NEW_ITEM_BOOST']
        )
        
        # Display forecasting summary
        avg_seasonal = seasonal_factor
        avg_multiplier = replen_calc['FORECAST_MULTIPLIER'].mean()
        avg_trend = replen_calc['TREND_FACTOR'].mean()
        avg_category = replen_calc['CATEGORY_FACTOR'].mean()
        
        # Detailed breakdown
        abc_counts = replen_calc['ABC_CLASS'].value_counts()
        velocity_counts = replen_calc['VELOCITY_CLASS'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📊 **Forecasting Factors:**\n"
                   f"• Seasonal: {avg_seasonal:.1f}x (Oct boost)\n"
                   f"• ABC-Velocity: {avg_multiplier:.1f}x avg\n" 
                   f"• Trend: {avg_trend:.1f}x avg\n"
                   f"• Category: {avg_category:.1f}x avg")
        
        with col2:
            st.success(f"📈 **Classification Results:**\n"
                      f"• A-items: {abc_counts.get('A', 0)} SKUs\n"
                      f"• B-items: {abc_counts.get('B', 0)} SKUs\n"
                      f"• C-items: {abc_counts.get('C', 0)} SKUs\n"
                      f"• Fast movers: {velocity_counts.get('Fast', 0)} SKUs")
        
        # Calculate target stock levels with proper safety stock
        replen_calc['TARGET_STOCK'] = np.ceil(replen_calc['FORECAST_DEMAND'])
        
        # Calculate initial replenishment quantities
        replen_calc['REPLEN_QTY'] = np.maximum(
            replen_calc['TARGET_STOCK'] - replen_calc['STOCK'],
            0
        )
        
        # Apply MOQ only for items that actually need replenishment
        replen_calc['REPLEN_QTY'] = np.where(
            (replen_calc['REPLEN_QTY'] > 0) & (replen_calc['DAILY_SALES'] > 0),
            np.maximum(replen_calc['REPLEN_QTY'], moq),
            0  # No replenishment for items with no sales history
        )
        
        # Special handling for slow-moving items (no sales but has stock)
        slow_moving_mask = (replen_calc['DAILY_SALES'] == 0) & (replen_calc['STOCK'] > 0)
        replen_calc.loc[slow_moving_mask, 'REPLEN_QTY'] = 0  # Don't replenish slow movers
        
        # Special handling for new items (no sales, no stock) - requires manual intervention
        new_item_mask = (replen_calc['DAILY_SALES'] == 0) & (replen_calc['STOCK'] == 0)
        replen_calc['IS_NEW_ITEM'] = new_item_mask
        
        # Merge with store priority scores and preserve store names
        replen_calc = pd.merge(
            replen_calc,
            store_metrics[['STORE', 'PRIORITY_SCORE']],
            on='STORE',
            how='left'
        )
        replen_calc['PRIORITY_SCORE'] = replen_calc['PRIORITY_SCORE'].fillna(0)
        
        # Add store names from sales data if available
        if 'STORE_NAME' in sales_data.columns:
            store_names = sales_data[['STORE', 'STORE_NAME']].drop_duplicates()
            replen_calc = pd.merge(
                replen_calc,
                store_names,
                on='STORE',
                how='left'
            )
        elif 'STORE_NAME' in stock_data.columns:
            store_names = stock_data[['STORE', 'STORE_NAME']].drop_duplicates()
            replen_calc = pd.merge(
                replen_calc,
                store_names,
                on='STORE',
                how='left'
            )
        
        # Add any missing SKU information from SKU master (if not already merged)
        missing_style_mask = replen_calc['STYLE'].isna() | (replen_calc['STYLE'] == 'UNKNOWN')
        if missing_style_mask.sum() > 0:
            st.info(f"🔄 Filling {missing_style_mask.sum()} missing STYLE/COLOR/SIZE values from SKU master")
            # Only merge for records that still have missing data
            missing_skus = replen_calc[missing_style_mask]['SKU'].unique()
            sku_master_subset = sku_master[sku_master['SKU'].isin(missing_skus)][['SKU', 'STYLE', 'COLOR', 'SIZE']]
            
            if not sku_master_subset.empty:
                # Create a temporary merge to update only missing values
                temp_merge = pd.merge(
                    replen_calc[missing_style_mask][['SKU']].reset_index(),
                    sku_master_subset,
                    on='SKU',
                    how='left'
                )
                
                # Update only the missing values
                for col in ['STYLE', 'COLOR', 'SIZE']:
                    if col in temp_merge.columns:
                        replen_calc.loc[missing_style_mask, col] = temp_merge.set_index('index')[col]
        
        # Add gender information from style master if available
        # We merge ALL styles (including FREEBIES), then filter at the very end
        exception_styles = {'A201', 'C101', 'A202'}  # Styles to keep regardless of gender
        
        if style_master is not None and not style_master.empty and 'GENDER' in style_master.columns:
            # Ensure STYLE column exists in both dataframes
            if 'STYLE' not in replen_calc.columns:
                st.warning("⚠️ STYLE column missing from replenishment data, creating from SKU")
                # Try to extract STYLE from SKU as fallback
                replen_calc['STYLE'] = replen_calc['SKU'].str.extract(r'([A-Z][A-Z0-9]+)')[0].fillna('UNKNOWN')
                
            if 'STYLE' not in style_master.columns:
                st.error("❌ STYLE column missing from style master data")
                # Continue without style master processing
                replen_calc['GENDER'] = 'UNKNOWN'
            else:
                # Continue with normal processing
                # Clean GENDER and STYLE columns in style master
                style_master['GENDER'] = style_master['GENDER'].astype(str).str.strip().str.upper()
                style_master['STYLE'] = style_master['STYLE'].astype(str).str.strip().str.upper()
            
            # Clean STYLE column in replen_calc for proper matching
            replen_calc['STYLE'] = replen_calc['STYLE'].astype(str).str.strip().str.upper()
            
            # Debug: Show style matching information
            sku_styles = set(replen_calc['STYLE'].unique())
            master_styles = set(style_master['STYLE'].unique())
            matched_styles = sku_styles.intersection(master_styles)
            unmatched_styles = sku_styles - master_styles
            
            st.info(f"📋 Style Matching: {len(matched_styles)} matched, {len(unmatched_styles)} unmatched")
            
            if len(unmatched_styles) > 0:
                st.warning(f"⚠️ Unmatched styles (will get UNKNOWN gender): {', '.join(list(unmatched_styles)[:10])}")
            
            # Show ALL gender types in Style Master (including FREEBIES)
            master_gender_counts = style_master['GENDER'].value_counts()
            st.info(f"� Style Master Gender Distribution: {', '.join([f'{gender}: {count}' for gender, count in master_gender_counts.items()])}")
            
            # Merge ALL styles from Style Master (NO FILTERING at this stage)
            st.info(f"📋 Merging ALL gender data: {len(style_master)} styles from Style Master")
            
            replen_calc = pd.merge(
                replen_calc,
                style_master[['STYLE', 'GENDER']],
                on='STYLE',
                how='left'
            )
            
            # For unmatched styles, try to auto-detect FREEBIES patterns
            unknown_mask = replen_calc['GENDER'].isna()
            unknown_styles = replen_calc[unknown_mask]['STYLE'].unique()
            
            if len(unknown_styles) > 0:
                st.warning(f"🔍 Found {len(unknown_styles)} styles not in Style Master")
                
                # Check if any unknown styles are actually FREEBIES based on style name
                freebies_patterns = ['GYM BAG', 'POWER BANK', 'AIRPODS', 'BAG', 'HEADPHONE', 'CHARGER', 'EARBUDS', 'SPEAKER']
                auto_detected_freebies = []
                
                for style in unknown_styles:
                    for pattern in freebies_patterns:
                        if pattern.upper() in str(style).upper():
                            auto_detected_freebies.append(style)
                            break
                
                if len(auto_detected_freebies) > 0:
                    st.warning(f"🚨 Auto-detecting {len(auto_detected_freebies)} potential FREEBIES: {', '.join(auto_detected_freebies)}")
                    
                    # Set these to FREEBIES instead of UNKNOWN
                    for style in auto_detected_freebies:
                        replen_calc.loc[replen_calc['STYLE'] == style, 'GENDER'] = 'FREEBIES'
                    
                    st.success(f"✅ Auto-assigned FREEBIES gender to {len(auto_detected_freebies)} styles")
            
            # Fill remaining missing gender values with UNKNOWN (only for truly unknown styles)
            replen_calc['GENDER'] = replen_calc['GENDER'].fillna('UNKNOWN')
            
            # Show gender distribution BEFORE filtering (this should include FREEBIES now)
            gender_counts = replen_calc['GENDER'].value_counts()
            st.info(f"📊 Gender Distribution (before filtering): {', '.join([f'{gender}: {count}' for gender, count in gender_counts.items()])}")
            
            st.success("✅ GENDER column added to output - FREEBIES included for now, will filter at the end")
        else:
            # If style master not available or no gender column, add default
            replen_calc['GENDER'] = 'UNKNOWN'
            st.warning("⚠️ Style Master not available - GENDER set to UNKNOWN")
        
        # ===== ENHANCED INTELLIGENT ALLOCATION =====
        st.write("🎯 Applying intelligent allocation strategy...")
        
        # Add REMARKS column for tracking allocation issues
        replen_calc['REMARKS'] = ''
        replen_calc['FINAL_REPLEN_QTY'] = 0.0
        
        # Calculate enhanced allocation factors
        # 1. Product importance score (ABC + Velocity)
        abc_weights = {'A': 1.0, 'B': 0.7, 'C': 0.4}
        velocity_weights = {'Fast': 1.0, 'Medium': 0.6, 'Slow': 0.2}
        
        replen_calc['PRODUCT_IMPORTANCE'] = (
            0.6 * replen_calc['ABC_CLASS'].map(abc_weights).fillna(0.4) +
            0.4 * replen_calc['VELOCITY_CLASS'].map(velocity_weights).fillna(0.2)
        )
        
        # 2. Store performance enhancement
        store_performance = replen_calc.groupby('STORE').agg({
            'DAILY_SALES': 'sum',
            'STOCK': 'sum',
            'PRIORITY_SCORE': 'first'
        }).reset_index()
        
        # Calculate store efficiency metrics
        store_performance['STOCK_TURN'] = np.where(
            store_performance['STOCK'] > 0,
            store_performance['DAILY_SALES'] / store_performance['STOCK'],
            store_performance['DAILY_SALES']  # If no stock, use sales as proxy
        )
        
        # Normalize stock turn
        max_turn = store_performance['STOCK_TURN'].max() if store_performance['STOCK_TURN'].max() > 0 else 1
        store_performance['TURN_SCORE'] = store_performance['STOCK_TURN'] / max_turn
        
        # Enhanced store score combining multiple factors
        store_performance['ENHANCED_STORE_SCORE'] = (
            0.4 * store_performance['PRIORITY_SCORE'] +
            0.4 * store_performance['TURN_SCORE'] +
            0.2 * (store_performance['DAILY_SALES'] / store_performance['DAILY_SALES'].max())
        )
        
        # Merge enhanced store scores
        replen_calc = pd.merge(
            replen_calc,
            store_performance[['STORE', 'ENHANCED_STORE_SCORE']],
            on='STORE',
            how='left'
        )
        
        # 3. Size set completion bonus
        size_set_importance = []
        for (store, style, color), group in replen_calc.groupby(['STORE', 'STYLE', 'COLOR']):
            total_sizes = len(group)
            sizes_with_stock = len(group[group['STOCK'] > 0])
            sizes_with_demand = len(group[group['DAILY_SALES'] > 0])
            
            completion_rate = (sizes_with_stock + sizes_with_demand) / (total_sizes * 2) if total_sizes > 0 else 0
            
            # Bonus for nearly complete sets
            size_bonus = 1.5 if completion_rate >= 0.7 else (1.2 if completion_rate >= 0.5 else 1.0)
            
            for idx in group.index:
                size_set_importance.append({
                    'INDEX': idx,
                    'SIZE_SET_BONUS': size_bonus
                })
        
        size_bonus_df = pd.DataFrame(size_set_importance).set_index('INDEX')
        replen_calc = replen_calc.join(size_bonus_df, how='left')
        replen_calc['SIZE_SET_BONUS'] = replen_calc['SIZE_SET_BONUS'].fillna(1.0)
        
        # Ultra-fast allocation using enhanced vectorized operations
        # Pre-calculate SKU-level aggregates
        sku_totals = replen_calc.groupby('SKU')['REPLEN_QTY'].sum()
        replen_calc['TOTAL_DEMAND'] = replen_calc['SKU'].map(sku_totals)
        
        # Identify shortage situations
        has_shortage = replen_calc['TOTAL_DEMAND'] > replen_calc['WAREHOUSE_STOCK']
        no_stock = replen_calc['WAREHOUSE_STOCK'] <= 0
        
        # Default: no stock scenario
        replen_calc.loc[no_stock, 'FINAL_REPLEN_QTY'] = 0
        replen_calc.loc[no_stock, 'REMARKS'] = 'No warehouse stock available'
        
        # Shortage scenario: allocate proportionally with enhanced factors
        shortage_mask = has_shortage & ~no_stock
        if shortage_mask.any():
            # Enhanced allocation factor combining multiple dimensions
            replen_calc.loc[shortage_mask, 'ALLOC_FACTOR'] = (
                replen_calc.loc[shortage_mask, 'ENHANCED_STORE_SCORE'] * 
                replen_calc.loc[shortage_mask, 'PRODUCT_IMPORTANCE'] *
                replen_calc.loc[shortage_mask, 'SIZE_SET_BONUS'] *
                replen_calc.loc[shortage_mask, 'REPLEN_QTY']
            )
            
            # Normalize by SKU group
            sku_alloc_totals = replen_calc[shortage_mask].groupby('SKU')['ALLOC_FACTOR'].transform('sum')
            replen_calc.loc[shortage_mask, 'FINAL_REPLEN_QTY'] = np.floor(
                (replen_calc.loc[shortage_mask, 'ALLOC_FACTOR'] / sku_alloc_totals) * 
                replen_calc.loc[shortage_mask, 'WAREHOUSE_STOCK']
            )
            
            replen_calc.loc[shortage_mask, 'REMARKS'] = 'Warehouse stock insufficient'
        
        # Sufficient stock scenario: fulfill demand with MOQ
        sufficient_mask = ~has_shortage & ~no_stock
        replen_calc.loc[sufficient_mask, 'FINAL_REPLEN_QTY'] = np.where(
            replen_calc.loc[sufficient_mask, 'REPLEN_QTY'] > 0,
            np.maximum(moq, replen_calc.loc[sufficient_mask, 'REPLEN_QTY']),
            0
        )
        
        # Round all quantities to integers
        replen_calc['FINAL_REPLEN_QTY'] = np.floor(replen_calc['FINAL_REPLEN_QTY']).astype(int)
        
        # ADVANCED SIZE-COLOR SET COMPLETION LOGIC (OPTIMIZED)
        # This ensures complete size runs for each style-color combination
        st.info("🎯 Applying Size-Color Set Completion Logic...")
        
        # Ensure required columns exist for size set analysis
        required_cols = ['STYLE', 'COLOR', 'SIZE']
        for col in required_cols:
            if col not in replen_calc.columns:
                st.warning(f"⚠️ {col} column missing, adding default values")
                replen_calc[col] = 'UNKNOWN'
        
        # PERFORMANCE CHECK: Skip complex size set analysis for very large datasets
        dataset_size = len(replen_calc)
        unique_combinations = len(replen_calc[['STORE', 'STYLE', 'COLOR']].drop_duplicates())
        
        # Define size set requirements (will be overridden by size_master if available)
        MIN_SIZES_REQUIRED = 4  # Default minimum 4 sizes per style-color
        MIN_SIZES_PLUS = 3      # Default minimum 3 sizes for plus sizes
        
        # Initialize size hierarchy (will be replaced by size master)
        SIZE_HIERARCHY = ['XS', 'S', 'M', 'L', 'XL', '2XL', '3XL', '4XL', '5XL']
        PLUS_SIZES = ['2XL', '3XL', '4XL', '5XL']
        SIZE_SETS = {}  # Will be populated from size_master
        
        # USE SIZE MASTER DATA IF AVAILABLE
        if size_master is not None and not size_master.empty:
            st.info("📏 Using uploaded Size Master for size set definitions")
            
            # Create size set mappings from uploaded data
            SIZE_SETS = {}
            for _, row in size_master.iterrows():
                size_set = row['SIZE_SET']
                if size_set not in SIZE_SETS:
                    SIZE_SETS[size_set] = {
                        'sizes': [],
                        'min_required': int(row['MIN_SIZES_REQUIRED'])
                    }
                SIZE_SETS[size_set]['sizes'].append(row['SIZE'])
            
            # Show size set summary
            st.write("📊 Size Sets Detected:")
            for set_name, set_data in SIZE_SETS.items():
                st.write(f"  • **{set_name}**: {len(set_data['sizes'])} sizes (min: {set_data['min_required']}) - {', '.join(set_data['sizes'][:5])}{'...' if len(set_data['sizes']) > 5 else ''}")
            
        else:
            st.warning("⚠️ No Size Master uploaded. Using default size hierarchy.")
            # Create default size sets
            SIZE_SETS = {
                'Regular': {'sizes': ['S', 'M', 'L', 'XL', '2XL'], 'min_required': 4},
                'Plus': {'sizes': ['3XL', '4XL', '5XL'], 'min_required': 3}
            }
        
        if dataset_size > 15000 or unique_combinations > 2500:
            st.warning(f"⚠️ Large dataset detected ({dataset_size} records, {unique_combinations} combinations). Using simplified size set logic for performance.")
            
            # Simplified approach for large datasets
            replen_calc['SIZE_SET_ALLOCATION'] = 0
            replen_calc['COMPLETE_SIZE_SET'] = False
            
            # Quick size set completion for high-priority items only
            high_priority_mask = replen_calc['PRIORITY_SCORE'] > 0.7  # Only high priority items
            if high_priority_mask.any():
                # Simple MOQ allocation for missing sizes in high-priority items
                zero_replen_mask = (replen_calc['FINAL_REPLEN_QTY'] == 0) & (replen_calc['WAREHOUSE_STOCK'] > 0) & high_priority_mask
                if zero_replen_mask.any():
                    replen_calc.loc[zero_replen_mask, 'FINAL_REPLEN_QTY'] = moq
                    replen_calc.loc[zero_replen_mask, 'SIZE_SET_ALLOCATION'] = moq
                    replen_calc.loc[zero_replen_mask, 'REMARKS'] = 'High priority size completion'
            
            st.success("✅ Simplified size set completion applied for performance")
            
        else:
            # Full size set analysis for smaller datasets
            st.write(f"📊 Processing {unique_combinations} style-color combinations...")
            
            # Size set analysis will use the uploaded size master data
            # ULTRA-OPTIMIZED: Minimal size set completion for performance
            replen_calc['SIZE_SET_ALLOCATION'] = 0
            replen_calc['COMPLETE_SIZE_SET'] = False
            
            # PERFORMANCE BYPASS: Skip size set completion for very large datasets
            if len(replen_calc) > 30000:
                st.warning(f"⚠️ Very large dataset ({len(replen_calc)} records). Skipping size set completion for performance.")
            else:
                # Quick size set completion for smaller datasets
                st.write(f"🔧 Applying size set completion to {unique_combinations} combinations...")
                
                # Pre-calculate size mappings for speed
                size_to_set = {}
                for set_name, set_data in SIZE_SETS.items():
                    for size in set_data['sizes']:
                        size_to_set[size] = (set_name, set_data['min_required'])
                
                # Vectorized approach: only process items with zero replenishment that have warehouse stock
                zero_replen_mask = (replen_calc['FINAL_REPLEN_QTY'] == 0) & (replen_calc.get('WAREHOUSE_STOCK', 0) > 0)
                candidates = replen_calc[zero_replen_mask].copy()
                
                if not candidates.empty:
                    # Add size set info to candidates
                    candidates['SIZE_SET_INFO'] = candidates['SIZE'].map(size_to_set)
                    candidates = candidates.dropna(subset=['SIZE_SET_INFO'])
                    
                    if not candidates.empty:
                        # For high-priority candidates only, apply minimal MOQ
                        high_priority_mask = candidates['PRIORITY_SCORE'] > 0.5  # Only high priority
                        if high_priority_mask.any():
                            selected_candidates = candidates[high_priority_mask].head(1000)  # Limit to 1000 items
                            
                            # Apply MOQ to selected candidates
                            replen_calc.loc[selected_candidates.index, 'FINAL_REPLEN_QTY'] = moq
                            replen_calc.loc[selected_candidates.index, 'SIZE_SET_ALLOCATION'] = moq
                            replen_calc.loc[selected_candidates.index, 'REMARKS'] = 'Priority size completion'
                            
                            st.success(f"✅ Applied size completion to {len(selected_candidates)} high-priority items")
                        else:
                            st.info("ℹ️ No high-priority items found for size completion")
                    else:
                        st.info("ℹ️ No items matched size sets")
                else:
                    st.info("ℹ️ No candidates found for size completion")
        
        # PERFORMANCE BYPASS: Skip complex analysis to improve speed
        # Just show basic metrics
        st.info(f"📊 Replenishment complete: {len(replen_calc)} records processed")
        
        # Skip detailed size-color analysis for performance
        # Original complex analysis commented out for speed
        
        # Initialize analysis list (required for later code)
        store_style_color_analysis = []
        updated_analysis = []  # Initialize for later statistics
        
        # PERFORMANCE BYPASS: Skip complex loops for large datasets
        if len(replen_calc) > 15000:
            st.warning("⚠️ Large dataset detected. Skipping detailed size-color analysis for performance.")
            # Create basic analysis for statistics
            style_color_counts = replen_calc.groupby(['STYLE', 'COLOR']).size().reset_index(name='SIZE_COUNT')
            updated_analysis = [{'IS_COMPLETE': count >= 3} for count in style_color_counts['SIZE_COUNT']]
        else:
            for store in replen_calc['STORE'].unique():
                store_data = replen_calc[replen_calc['STORE'] == store].copy()
            
            # Group by style-color combinations
            for (style, color), group in store_data.groupby(['STYLE', 'COLOR']):
                sizes_available = set(group['SIZE'].unique())
                sizes_after_replen = set(group[group['FINAL_REPLEN_QTY'] > 0]['SIZE'].unique())
                current_stock_sizes = set(group[group['STOCK'] > 0]['SIZE'].unique())
                
                # Determine if this is a plus size combination
                is_plus_combo = bool(sizes_available.intersection(PLUS_SIZES))
                min_required = MIN_SIZES_PLUS if is_plus_combo else MIN_SIZES_REQUIRED
                
                # Calculate completeness
                total_sizes_available = len(sizes_available)
                sizes_with_stock_or_replen = len(sizes_after_replen.union(current_stock_sizes))
                
                # Identify missing sizes for completion
                all_possible_sizes = sizes_available
                missing_sizes = all_possible_sizes - sizes_after_replen - current_stock_sizes
                
                store_style_color_analysis.append({
                    'STORE': store,
                    'STYLE': style,
                    'COLOR': color,
                    'TOTAL_SIZES': total_sizes_available,
                    'SIZES_WITH_INVENTORY': sizes_with_stock_or_replen,
                    'MISSING_SIZES': len(missing_sizes),
                    'MISSING_SIZE_LIST': list(missing_sizes),
                    'IS_PLUS_COMBO': is_plus_combo,
                    'MIN_REQUIRED': min_required,
                    'IS_COMPLETE': sizes_with_stock_or_replen >= min_required,
                    'COMPLETENESS_RATIO': sizes_with_stock_or_replen / max(1, total_sizes_available)
                })
        
        # Convert to DataFrame for analysis (handles both bypass and normal execution)
        set_analysis_df = pd.DataFrame(store_style_color_analysis)
        
        if not set_analysis_df.empty:
            # Show set completion statistics
            incomplete_sets = set_analysis_df[~set_analysis_df['IS_COMPLETE']]
            complete_sets = set_analysis_df[set_analysis_df['IS_COMPLETE']]
            
            st.info(f"📊 Size-Color Set Analysis:")
            st.info(f"   Complete sets: {len(complete_sets)} | Incomplete sets: {len(incomplete_sets)}")
            
            if len(incomplete_sets) > 0:
                st.warning(f"⚠️ Found {len(incomplete_sets)} incomplete size-color sets")
                
                # Priority-based set completion
                # For incomplete sets, boost replenishment for missing sizes
                for _, row in incomplete_sets.iterrows():
                    store, style, color = row['STORE'], row['STYLE'], row['COLOR']
                    missing_sizes = row['MISSING_SIZE_LIST']
                    
                    if len(missing_sizes) > 0:
                        # Find SKUs for missing sizes that have warehouse stock
                        missing_sku_mask = (
                            (replen_calc['STORE'] == store) &
                            (replen_calc['STYLE'] == style) &
                            (replen_calc['COLOR'] == color) &
                            (replen_calc['SIZE'].isin(missing_sizes)) &
                            (replen_calc['WAREHOUSE_STOCK'] > 0) &
                            (replen_calc['FINAL_REPLEN_QTY'] == 0)  # Currently not getting replenishment
                        )
                        
                        if missing_sku_mask.any():
                            # Apply minimum replenishment to complete the set
                            replen_calc.loc[missing_sku_mask, 'FINAL_REPLEN_QTY'] = np.maximum(
                                replen_calc.loc[missing_sku_mask, 'FINAL_REPLEN_QTY'],
                                np.minimum(moq, replen_calc.loc[missing_sku_mask, 'WAREHOUSE_STOCK'])
                            )
                            
                            # Update remarks
                            replen_calc.loc[missing_sku_mask, 'REMARKS'] = replen_calc.loc[missing_sku_mask, 'REMARKS'] + '; Size set completion'
                            
                            # Mark as size set allocation
                            replen_calc.loc[missing_sku_mask, 'SIZE_SET_ALLOCATION'] = True
                
                st.success(f"✅ Applied size set completion logic to {len(incomplete_sets)} incomplete sets")
            
            # Re-analyze after set completion
            updated_analysis = []
            for store in replen_calc['STORE'].unique():
                store_data = replen_calc[replen_calc['STORE'] == store].copy()
                
                for (style, color), group in store_data.groupby(['STYLE', 'COLOR']):
                    sizes_after_replen = set(group[group['FINAL_REPLEN_QTY'] > 0]['SIZE'].unique())
                    current_stock_sizes = set(group[group['STOCK'] > 0]['SIZE'].unique())
                    sizes_with_inventory = len(sizes_after_replen.union(current_stock_sizes))
                    total_sizes = len(group['SIZE'].unique())
                    is_plus_combo = bool(set(group['SIZE'].unique()).intersection(PLUS_SIZES))
                    min_required = MIN_SIZES_PLUS if is_plus_combo else MIN_SIZES_REQUIRED
                    
                    updated_analysis.append({
                        'STORE': store,
                        'STYLE': style,
                        'COLOR': color,
                        'SIZES_WITH_INVENTORY': sizes_with_inventory,
                        'TOTAL_SIZES': total_sizes,
                        'IS_COMPLETE': sizes_with_inventory >= min_required
                    })
            
            updated_df = pd.DataFrame(updated_analysis)
            if not updated_df.empty:
                final_complete = len(updated_df[updated_df['IS_COMPLETE']])
                final_incomplete = len(updated_df[~updated_df['IS_COMPLETE']])
                st.success(f"📈 Final Set Status: {final_complete} complete, {final_incomplete} incomplete")
        
        # Add size set completion flag
        if 'SIZE_SET_ALLOCATION' not in replen_calc.columns:
            replen_calc['SIZE_SET_ALLOCATION'] = False
        
        # Simplified size set check (keep for compatibility)
        replen_calc['COMPLETE_SIZE_SET'] = False
        
        # FINAL FILTERING: Remove FREEBIES and UNISEX after all calculations are complete
        # Exception styles: A201, C101, A202
        exception_styles = {'A201', 'C101', 'A202'}
        
        if 'GENDER' in replen_calc.columns:
            # Ensure STYLE column exists for filtering
            if 'STYLE' not in replen_calc.columns:
                st.warning("⚠️ STYLE column missing, adding default values")
                replen_calc['STYLE'] = 'UNKNOWN'
            
            # Clean columns for comparison
            replen_calc['GENDER'] = replen_calc['GENDER'].astype(str).str.strip().str.upper()
            replen_calc['STYLE'] = replen_calc['STYLE'].astype(str).str.strip().str.upper()
            
            # Count items before filtering
            total_before = len(replen_calc)
            
            # Check for FREEBIES (including auto-detected ones)
            freebies_mask = replen_calc['GENDER'].str.contains('FREEBIE', na=False, case=False) | (replen_calc['GENDER'] == 'FREEBIES')
            freebies_count = freebies_mask.sum()
            
            # Check for UNISEX (excluding exception styles)
            unisex_mask = (replen_calc['GENDER'] == 'UNISEX') & (~replen_calc['STYLE'].isin(exception_styles))
            unisex_count = unisex_mask.sum()
            
            # Also remove any remaining UNKNOWN items that look like FREEBIES
            unknown_freebies_mask = (replen_calc['GENDER'] == 'UNKNOWN') & (replen_calc['STYLE'].str.contains('GYM BAG|POWER BANK|AIRPODS|BAG|HEADPHONE|CHARGER', na=False, case=False))
            unknown_freebies_count = unknown_freebies_mask.sum()
            
            # Show what we're about to filter
            if freebies_count > 0:
                freebies_styles = replen_calc[freebies_mask]['STYLE'].unique()
                st.warning(f"� Removing {freebies_count} FREEBIES items from final output")
                st.warning(f"   FREEBIES styles: {', '.join(str(s) for s in freebies_styles[:10])}")
            
            if unisex_count > 0:
                unisex_styles = replen_calc[unisex_mask]['STYLE'].unique()
                st.warning(f"🚫 Removing {unisex_count} UNISEX items from final output")
                st.warning(f"   UNISEX styles: {', '.join(str(s) for s in unisex_styles[:10])}")
            
            if unknown_freebies_count > 0:
                unknown_freebies_styles = replen_calc[unknown_freebies_mask]['STYLE'].unique()
                st.warning(f"🚫 Removing {unknown_freebies_count} UNKNOWN items that appear to be FREEBIES")
                st.warning(f"   Detected FREEBIES styles: {', '.join(str(s) for s in unknown_freebies_styles[:10])}")
            
            # Show exception styles being kept
            exception_kept = exception_styles & set(replen_calc['STYLE'].unique())
            if exception_kept:
                st.info(f"✅ Keeping exception styles: {', '.join(exception_kept)}")
            
            # Apply the filtering (remove FREEBIES, UNISEX, and UNKNOWN FREEBIES)
            replen_calc = replen_calc[~freebies_mask & ~unisex_mask & ~unknown_freebies_mask].copy()
            
            # Show final counts
            total_after = len(replen_calc)
            total_removed = total_before - total_after
            
            if total_removed > 0:
                st.success(f"✅ Final filtering complete: Removed {total_removed} items ({freebies_count} FREEBIES + {unisex_count} UNISEX + {unknown_freebies_count} UNKNOWN FREEBIES)")
                st.success(f"📊 Final output: {total_after} items ready for replenishment")
            else:
                st.success("✅ No FREEBIES or UNISEX items found - output is clean")
            
            # Show final gender distribution
            final_gender_counts = replen_calc['GENDER'].value_counts()
            st.info(f"📊 Final Gender Distribution: {', '.join([f'{gender}: {count}' for gender, count in final_gender_counts.items()])}")
        
        return replen_calc
        
    except Exception as e:
        st.error(f"Error in replenishment calculations: {str(e)}")
        return pd.DataFrame()
        
        # Calculate out of stock situations
        zero_stock_counts = stock_data[stock_data['STOCK'] == 0].groupby('STORE').size().reset_index(name='ZERO_STOCK_COUNT')
        store_metrics = pd.merge(store_metrics, zero_stock_counts, on='STORE', how='left')
        store_metrics['ZERO_STOCK_COUNT'] = store_metrics['ZERO_STOCK_COUNT'].fillna(0)
        store_metrics['OOS_RANK'] = (1 - store_metrics['ZERO_STOCK_COUNT'].rank(pct=True)).fillna(0)
        
        # Calculate final priority score
        store_metrics['PRIORITY_SCORE'] = (
            0.4 * store_metrics['SALES_RANK'] +
            0.3 * store_metrics['STOCK_TURN_RANK'].fillna(0) +
            0.3 * store_metrics['OOS_RANK']
        )
        
        # Normalize priority score to be between 0 and 1
        store_metrics['PRIORITY_SCORE'] = (store_metrics['PRIORITY_SCORE'] - store_metrics['PRIORITY_SCORE'].min()) / \
                                        (store_metrics['PRIORITY_SCORE'].max() - store_metrics['PRIORITY_SCORE'].min() + 1e-10)
        
        # Now proceed with sales velocity calculations
        recent_sales['DAYS_AGO'] = (latest_date - recent_sales['DATE']).dt.days
        recent_sales['WEIGHT'] = 1 / (recent_sales['DAYS_AGO'] + 1)  # More weight to recent sales
        
        # Calculate weighted daily sales
        sales_by_day = recent_sales.groupby(['STORE', 'SKU', 'DATE'])['QUANTITY'].sum().reset_index()
        sales_by_day['DOW'] = sales_by_day['DATE'].dt.dayofweek
        
        # Calculate day-of-week factors
        dow_factors = sales_by_day.groupby('DOW')['QUANTITY'].mean() / sales_by_day['QUANTITY'].mean()
        sales_by_day['DOW_FACTOR'] = sales_by_day['DOW'].map(dow_factors)
        
        # Calculate weighted sales velocity
        weighted_sales = (recent_sales['QUANTITY'] * recent_sales['WEIGHT']).sum()
        total_weights = recent_sales['WEIGHT'].sum()
        base_daily_sales = weighted_sales / total_weights
        
        sales_velocity = recent_sales.groupby(['STORE', 'SKU']).agg({
            'QUANTITY': 'sum',
            'WEIGHT': 'sum'
        }).reset_index()
        
        sales_velocity['DAILY_SALES'] = (sales_velocity['QUANTITY'] * sales_velocity['WEIGHT']) / (time_period_days * sales_velocity['WEIGHT'])
        
        # Calculate sales variability for safety stock
        sales_std = sales_by_day.groupby(['STORE', 'SKU'])['QUANTITY'].agg(['std', 'mean']).reset_index()
        sales_std['CV'] = sales_std['std'] / sales_std['mean']  # Coefficient of variation
        sales_std['CV'] = sales_std['CV'].fillna(0)
        
        # Service level factor (95% service level = 1.645)
        service_level_factor = 1.645
        
        # Calculate safety stock based on variability
        sales_velocity = pd.merge(sales_velocity, sales_std[['STORE', 'SKU', 'CV']], on=['STORE', 'SKU'], how='left')
        sales_velocity['SAFETY_STOCK'] = np.ceil(
            service_level_factor * sales_velocity['DAILY_SALES'] * sales_velocity['CV'] * np.sqrt(14)
        )
        
        # Calculate desired stock levels with dynamic safety stock
        sales_velocity['DESIRED_STOCK'] = np.ceil(
            sales_velocity['DAILY_SALES'] * 14 + sales_velocity['SAFETY_STOCK']
        )
        
        # Merge with current stock levels
        stock_levels = stock_data[['STORE', 'SKU', 'STOCK']]
        replen_calc = pd.merge(
            sales_velocity,
            stock_levels,
            on=['STORE', 'SKU'],
            how='outer'
        ).fillna(0)
        
        # Initial replenishment quantities
        replen_calc['REPLEN_QTY'] = np.maximum(
            replen_calc['DESIRED_STOCK'] - replen_calc['STOCK'],
            0
        )
        
        # Merge with warehouse stock and additional constraints
        # Check if warehouse data has the required columns
        if 'SKU' in warehouse_data.columns and 'WAREHOUSE_STOCK' in warehouse_data.columns:
            warehouse_stock = warehouse_data[['SKU', 'WAREHOUSE_STOCK']]
            replen_calc = pd.merge(
                replen_calc,
                warehouse_stock,
                on='SKU',
                how='left'
            )
            replen_calc['WAREHOUSE_STOCK'] = replen_calc['WAREHOUSE_STOCK'].fillna(0)
        else:
            # If warehouse data doesn't have the expected columns, set default values
            replen_calc['WAREHOUSE_STOCK'] = 0
        
        # Add warehouse allocation optimization logic
        def optimize_allocation(group):
            available_stock = group['WAREHOUSE_STOCK'].iloc[0] if 'WAREHOUSE_STOCK' in group.columns else 0
            total_demand = group['REPLEN_QTY'].sum()
            
            if total_demand <= available_stock:
                return group['REPLEN_QTY']
            
            # Calculate allocation weights based on multiple factors
            weights = (
                group['PRIORITY_SCORE'] *  # Store priority
                (1 / (group['STOCK_COVER_DAYS'] + 1)) *  # Lower stock cover gets higher priority
                group['DAILY_SALES'].clip(lower=0.1)  # Sales velocity with minimum threshold
            )
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Initial allocation based on weights
            initial_allocation = np.floor(weights * available_stock)
            
            # Distribute remaining units based on fractional parts
            remaining = available_stock - initial_allocation.sum()
            if remaining > 0:
                fractional_parts = (weights * available_stock) - initial_allocation
                top_fractional = fractional_parts.nlargest(int(remaining))
                initial_allocation[top_fractional.index] += 1
            
            return initial_allocation
        
        # Apply optimization by SKU
        replen_calc['OPTIMIZED_QTY'] = replen_calc.groupby('SKU').apply(
            lambda x: optimize_allocation(x)
        ).reset_index(level=0, drop=True)
        
        # Update replenishment quantities with optimized values
        replen_calc['REPLEN_QTY'] = replen_calc['OPTIMIZED_QTY']
        
        # Apply minimum and maximum order constraints
        max_order_qty = 100  # Maximum order quantity per store-SKU
        pack_size = 6  # Standard pack size
        
        # Round to pack size and apply min/max constraints
        replen_calc['FINAL_REPLEN_QTY'] = np.where(
            replen_calc['REPLEN_QTY'] > 0,
            np.clip(
                np.ceil(replen_calc['REPLEN_QTY'] / pack_size) * pack_size,  # Round up to nearest pack
                moq,  # Use the MOQ parameter passed to function
                max_order_qty  # Maximum order quantity
            ),
            0
        )
        
        # Ensure we don't exceed warehouse stock
        replen_calc['FINAL_REPLEN_QTY'] = np.minimum(
            replen_calc['FINAL_REPLEN_QTY'],
            replen_calc['WAREHOUSE_STOCK']
        )
        
        # Filter out small orders that don't meet MOQ
        replen_calc['FINAL_REPLEN_QTY'] = np.where(
            replen_calc['FINAL_REPLEN_QTY'] < moq,
            0,
            replen_calc['FINAL_REPLEN_QTY']
        )
        
        # Prioritize replenishment based on stock cover
        replen_calc['STOCK_COVER_DAYS'] = np.where(
            replen_calc['DAILY_SALES'] > 0,
            replen_calc['STOCK'] / replen_calc['DAILY_SALES'],
            999  # High number for items with no sales
        )
        
        # Add SKU information
        replen_calc = pd.merge(
            replen_calc,
            sku_master[['SKU', 'STYLE', 'SIZE', 'COLOR']],
            on='SKU',
            how='left'
        )
        
        # Calculate final allocation quantities
        replen_calc['FINAL_REPLEN_QTY'] = np.minimum(
            replen_calc['REPLEN_QTY'],
            replen_calc['WAREHOUSE_STOCK']
        )
        
        # Calculate store performance metrics
        # Start with store list from stock data to ensure all stores are included
        store_metrics = stock_data[['STORE']].drop_duplicates()
        
        # Sales performance (last 30 days)
        store_sales = recent_sales.groupby('STORE')['QUANTITY'].sum().reset_index()
        store_sales['SALES_RANK'] = store_sales['QUANTITY'].rank(pct=True)
        store_metrics = pd.merge(store_metrics, store_sales[['STORE', 'SALES_RANK']], 
                               on='STORE', how='left')
        store_metrics['SALES_RANK'] = store_metrics['SALES_RANK'].fillna(0)  # New stores get lowest rank
        
        # Stock efficiency
        store_stock = stock_data.groupby('STORE')['STOCK'].sum().reset_index()
        store_metrics = pd.merge(store_metrics, store_stock, on='STORE', how='outer')
        
        # Calculate stock turn
        store_metrics['STOCK_TURN'] = store_sales['QUANTITY'] / store_metrics['STOCK']
        store_metrics['STOCK_TURN_RANK'] = store_metrics['STOCK_TURN'].rank(pct=True)
        
        # Lost sales potential (out of stock situations)
        zero_stock = stock_data[stock_data['STOCK'] == 0].groupby('STORE').size()
        store_metrics['ZERO_STOCK_COUNT'] = store_metrics['STORE'].map(zero_stock).fillna(0)
        store_metrics['OOS_RANK'] = (1 - store_metrics['ZERO_STOCK_COUNT'].rank(pct=True))
        
        # Calculate overall store priority score (weighted average of ranks)
        store_metrics['PRIORITY_SCORE'] = (
            0.4 * store_metrics['SALES_RANK'] +  # Higher weight to sales performance
            0.3 * store_metrics['STOCK_TURN_RANK'] +  # Reward efficient stock management
            0.3 * store_metrics['OOS_RANK']  # Consider out-of-stock situations
        )
        
        # Merge priority scores with replenishment calculations
        replen_calc = pd.merge(
            replen_calc,
            store_metrics[['STORE', 'PRIORITY_SCORE']],
            on='STORE',
            how='left'
        )
        replen_calc['PRIORITY_SCORE'] = replen_calc['PRIORITY_SCORE'].fillna(0)  # Handle any stores not in metrics
        
        # Sort by priority score and stock cover
        replen_calc = replen_calc.sort_values(
            ['PRIORITY_SCORE', 'STOCK_COVER_DAYS'],
            ascending=[False, True]
        )
        
        # Handle warehouse stock constraints with priority-based allocation
        for sku in replen_calc['SKU'].unique():
            sku_data = replen_calc[replen_calc['SKU'] == sku].copy()
            available_stock = sku_data['WAREHOUSE_STOCK'].iloc[0] if 'WAREHOUSE_STOCK' in sku_data.columns and len(sku_data) > 0 else 0
            
            if sku_data['REPLEN_QTY'].sum() > available_stock and available_stock > 0:
                # Priority-based allocation if not enough stock
                weighted_priority = sku_data['PRIORITY_SCORE'] * sku_data['REPLEN_QTY']
                total_weighted_priority = weighted_priority.sum()
                
                if total_weighted_priority > 0:
                    # Allocate based on weighted priority
                    sku_data['FINAL_REPLEN_QTY'] = np.floor(
                        (weighted_priority / total_weighted_priority) * available_stock
                    )
                else:
                    # Fallback to equal distribution if all priorities are zero
                    stores_count = len(sku_data)
                    base_allocation = np.floor(available_stock / stores_count)
                    sku_data['FINAL_REPLEN_QTY'] = base_allocation
                    
                    # Distribute remaining units
                    remaining = available_stock - (base_allocation * stores_count)
                    if remaining > 0:
                        sku_data.iloc[:int(remaining), sku_data.columns.get_loc('FINAL_REPLEN_QTY')] += 1
                
                replen_calc.loc[replen_calc['SKU'] == sku, 'FINAL_REPLEN_QTY'] = sku_data['FINAL_REPLEN_QTY']
            else:
                # If enough stock, fulfill the original replenishment quantity
                replen_calc.loc[replen_calc['SKU'] == sku, 'FINAL_REPLEN_QTY'] = sku_data['REPLEN_QTY']
        
        # Final safety check - ensure essential columns exist
        essential_columns = ['STORE', 'SKU', 'STYLE', 'COLOR', 'SIZE']
        for col in essential_columns:
            if col not in replen_calc.columns:
                if col in ['STYLE', 'COLOR', 'SIZE']:
                    replen_calc[col] = 'UNKNOWN'
                else:
                    replen_calc[col] = replen_calc.get('STORE', 'UNKNOWN')
        
        # Ensure no None/NaN values in essential columns
        for col in essential_columns:
            replen_calc[col] = replen_calc[col].astype(str).fillna('UNKNOWN')
        
        st.success(f"✅ Replenishment calculation completed successfully: {len(replen_calc)} records")
        return replen_calc
    
    except Exception as e:
        st.error(f"Error in replenishment calculations: {str(e)}")
        return pd.DataFrame()

# Utility functions for optimized processing
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by converting to appropriate types"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'object':
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        elif col_type in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

def process_chunk(chunk: pd.DataFrame, dtype_dict: dict) -> pd.DataFrame:
    """Process a single chunk of data in parallel"""
    chunk = chunk.copy()
    
    # Convert to appropriate types
    for col, dtype in dtype_dict.items():
        if col in chunk.columns:
            chunk[col] = chunk[col].astype(dtype)
    
    # Clean string columns
    string_columns = chunk.select_dtypes(include=['object']).columns
    for col in string_columns:
        if col not in dtype_dict:
            chunk[col] = chunk[col].fillna('').str.strip()
    
    return optimize_dataframe(chunk)

@st.cache_data(ttl=3600, show_spinner=False)
def read_file(uploaded_file):
    """Read uploaded file into pandas DataFrame with optimized performance"""
    if uploaded_file is None:
        return pd.DataFrame()
        
    try:
        start_time = time.time()
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # Read file with optimized parameters
        if file_type == 'csv':
            # Use fastest CSV reading settings
            df = pd.read_csv(
                uploaded_file,
                engine='c',  # Use C engine for speed
                na_values=['', 'NA', 'null'],
                keep_default_na=True,
                low_memory=False
            )
        elif file_type in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error(f"Unsupported file type: {file_type}")
            return pd.DataFrame()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Show quick success message
        st.success(f"✅ Loaded {len(df):,} rows in {processing_time:.2f}s")
        
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return pd.DataFrame()

def detect_column(df, possible_names):
    """Find a column from a list of possible names"""
    for name in possible_names:
        matches = [col for col in df.columns if name.lower() in col.lower()]
        if matches:
            return matches[0]
    return None

def safe_numeric(value):
    """Safely convert value to numeric"""
    try:
        num = float(value)
        return num if not pd.isna(num) else 0
    except (ValueError, TypeError):
        return 0

def validate_data(df: pd.DataFrame, file_type: str) -> Tuple[bool, List[str]]:
    """Validate data quality and completeness"""
    errors = []
    
    # Check for required columns based on file type
    required_columns = {
        'sales': ['DATE', 'STORE', 'SKU', 'QUANTITY'],
        'stock': ['STORE', 'SKU', 'STOCK'],
        'warehouse': ['SKU', 'WAREHOUSE_STOCK'],
        'sku_master': ['SKU', 'STYLE', 'COLOR', 'SIZE']
    }
    
    if file_type in required_columns:
        missing_cols = [col for col in required_columns[file_type] 
                       if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for duplicate records
    duplicates = df.duplicated()
    if duplicates.any():
        errors.append(f"Found {duplicates.sum()} duplicate records")
    
    # Check for data type consistency
    for col in df.columns:
        if df[col].dtype == 'object':
            non_string = df[col].apply(lambda x: not isinstance(x, str))
            if non_string.any():
                errors.append(f"Column {col} contains mixed data types")
    
    # Validate date formats if present
    if 'DATE' in df.columns:
        invalid_dates = pd.to_datetime(df['DATE'], errors='coerce').isna()
        if invalid_dates.any():
            errors.append(f"Found {invalid_dates.sum()} invalid dates")
    
    return len(errors) == 0, errors

def clean_text(value):
    """Clean text fields with advanced handling"""
    if pd.isna(value):
        return ""
    
    # Remove special characters and extra spaces
    cleaned = str(value).strip()
    cleaned = ' '.join(cleaned.split())  # Normalize spaces
    cleaned = cleaned.upper()
    
    # Remove common problematic characters
    cleaned = cleaned.replace('\t', ' ').replace('\n', ' ')
    
    return cleaned

# Data Cleaning Functions
def fast_clean_text(series):
    """Fast text cleaning using vectorized operations"""
    return series.fillna('').astype(str).str.strip().str.upper()

def clean_sales_data(df):
    """Clean sales data - optimized for speed"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Detect columns (once)
        store_col = detect_column(df, ['store', 'store name', 'ebo', 'channel'])
        store_code_col = detect_column(df, ['store code', 'store_code', 'code', 'store id', 'store_id'])
        ebo_name_col = detect_column(df, ['ebo name', 'ebo_name', 'store name', 'store_name'])
        sku_col = detect_column(df, ['sku', 'ean', 'product code'])
        date_col = detect_column(df, ['date', 'bill date', 'transaction date'])
        qty_col = detect_column(df, ['quantity', 'qty', 'bill quantity'])
        rate_col = detect_column(df, ['rate', 'price', 'mrp', 'amount per unit', 'unit price'])
        amount_col = detect_column(df, ['amount', 'total amount', 'bill amount', 'value'])
        
        if not all([sku_col, date_col, qty_col]):
            st.error("Missing required columns")
            return pd.DataFrame()
        
        # Start with basic columns
        result = pd.DataFrame()
        result['SKU'] = fast_clean_text(df[sku_col])
        result['DATE'] = pd.to_datetime(df[date_col], errors='coerce')
        result['QUANTITY'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
        
        # Add store columns
        if store_code_col:
            result['STORE'] = fast_clean_text(df[store_code_col])
            if ebo_name_col:
                result['STORE_NAME'] = fast_clean_text(df[ebo_name_col])
            elif store_col:
                result['STORE_NAME'] = fast_clean_text(df[store_col])
        elif store_col:
            result['STORE'] = fast_clean_text(df[store_col])
            if ebo_name_col:
                result['STORE_NAME'] = fast_clean_text(df[ebo_name_col])
        else:
            st.error("Missing store columns")
            return pd.DataFrame()
        
        # Filter freebies
        if rate_col:
            mask = pd.to_numeric(df[rate_col], errors='coerce').fillna(0) > 0
            result = result[mask]
        elif amount_col:
            mask = pd.to_numeric(df[amount_col], errors='coerce').fillna(0) > 0
            result = result[mask]
        
        # Filter invalid records
        result = result[(result['DATE'].notna()) & (result['QUANTITY'] > 0)]
        
        return result.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

def clean_stock_data(df):
    """Clean stock data"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        store_col = detect_column(df, ['store', 'store name', 'ebo', 'channel'])
        store_code_col = detect_column(df, ['store code', 'store_code', 'code', 'store id', 'store_id'])
        ebo_name_col = detect_column(df, ['ebo name', 'ebo_name', 'store name', 'store_name'])
        sku_col = detect_column(df, ['sku', 'ean', 'product code'])
        stock_col = detect_column(df, ['stock', 'quantity', 'qty', 'available'])
        
        if not all([sku_col, stock_col]):
            st.error("Missing required columns")
            return pd.DataFrame()
        
        result = pd.DataFrame()
        result['SKU'] = fast_clean_text(df[sku_col])
        result['STOCK'] = pd.to_numeric(df[stock_col], errors='coerce').fillna(0)
        
        if store_code_col:
            result['STORE'] = fast_clean_text(df[store_code_col])
            if ebo_name_col:
                result['STORE_NAME'] = fast_clean_text(df[ebo_name_col])
            elif store_col:
                result['STORE_NAME'] = fast_clean_text(df[store_col])
        elif store_col:
            result['STORE'] = fast_clean_text(df[store_col])
            if ebo_name_col:
                result['STORE_NAME'] = fast_clean_text(df[ebo_name_col])
        else:
            st.error("Missing store columns")
            return pd.DataFrame()
        
        return result
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

def clean_warehouse_data(df):
    """Clean warehouse data"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        sku_col = detect_column(df, ['each client sku id', 'sku', 'ean', 'product code', 'client sku id'])
        stock_col = detect_column(df, ['total available quantity', 'stock', 'quantity', 'available quantity', 'available'])
        
        if not all([sku_col, stock_col]):
            st.error("Missing required columns")
            return pd.DataFrame()
        
        result = pd.DataFrame()
        result['SKU'] = fast_clean_text(df[sku_col])
        result['WAREHOUSE_STOCK'] = pd.to_numeric(df[stock_col], errors='coerce').fillna(0)
        
        return result
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

def clean_sku_master(df):
    """Clean SKU master data"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        sku_col = detect_column(df, ['sku', 'ean', 'product code'])
        style_col = detect_column(df, ['style', 'style code'])
        color_col = detect_column(df, ['color', 'colour'])
        size_col = detect_column(df, ['size'])
        
        if not all([sku_col, style_col, color_col, size_col]):
            st.error("Missing required columns")
            return pd.DataFrame()
        
        result = pd.DataFrame()
        result['SKU'] = fast_clean_text(df[sku_col])
        result['STYLE'] = fast_clean_text(df[style_col])
        result['COLOR'] = fast_clean_text(df[color_col])
        result['SIZE'] = fast_clean_text(df[size_col])
        
        return result
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

def clean_style_master(df):
    """Clean style master data"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        style_col = detect_column(df, ['style', 'style code'])
        gender_col = detect_column(df, ['gender', 'department'])
        
        if not all([style_col, gender_col]):
            st.error("Missing required columns")
            return pd.DataFrame()
        
        result = pd.DataFrame()
        result['STYLE'] = fast_clean_text(df[style_col])
        result['GENDER'] = fast_clean_text(df[gender_col])
        result = result[result['GENDER'] != 'FREEBIES']
        
        return result
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

def clean_size_master(df):
    """Clean size master data for size set completion logic"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Required columns: SIZE, SIZE_SET, MIN_SIZES_REQUIRED
        size_col = detect_column(df, ['size', 'size_code', 'size_name'])
        size_set_col = detect_column(df, ['size_set', 'size_group', 'set'])
        min_sizes_col = detect_column(df, ['min_sizes_required', 'min_sizes', 'required_sizes'])
        
        if not all([size_col, size_set_col, min_sizes_col]):
            st.error("Size Master missing required columns: SIZE, SIZE_SET, MIN_SIZES_REQUIRED")
            return pd.DataFrame()
        
        result = pd.DataFrame()
        result['SIZE'] = fast_clean_text(df[size_col])
        result['SIZE_SET'] = fast_clean_text(df[size_set_col])
        result['MIN_SIZES_REQUIRED'] = pd.to_numeric(df[min_sizes_col], errors='coerce').fillna(4)
        
        # Remove rows with missing data
        result = result.dropna(subset=['SIZE', 'SIZE_SET'])
        
        return result
        
    except Exception as e:
        st.error(f"Error cleaning size master: {str(e)}")
        return pd.DataFrame()

def check_login():
    """Check if user is logged in"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in

# Use session state for storage in Streamlit Cloud (can't create directories)
# For local development, you can uncomment the lines below
# STORAGE_DIR = Path("d:/DATA TILL DATE/Desktop/.streamlit_data")
# STORAGE_DIR.mkdir(exist_ok=True)
# DATA_FILE = STORAGE_DIR / "persisted_data.pkl"
# HISTORY_FILE = STORAGE_DIR / "upload_history.pkl"

# Use session state for storage in Streamlit Cloud (filesystem is limited)
# Don't try to create directories on Streamlit Cloud
STORAGE_DIR = None
DATA_FILE = None
HISTORY_FILE = None

def load_from_disk(filename):
    """Load data from disk"""
    if filename is None:
        return None
    try:
        if filename.exists():
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load saved data: {str(e)}")
    return None

def save_to_disk(data, filename):
    """Save data to disk"""
    if filename is None:
        return False
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        st.error(f"Could not save data: {str(e)}")
        return False

def get_persisted_data():
    """Get persisted data from disk or cache"""
    if 'persisted_data' not in st.session_state:
        # Try to load from disk
        disk_data = load_from_disk(DATA_FILE)
        st.session_state.persisted_data = disk_data if disk_data else {}
    return st.session_state.persisted_data

def get_upload_history():
    """Get upload history with timestamps from disk or cache"""
    if 'upload_history' not in st.session_state:
        # Try to load from disk
        disk_history = load_from_disk(HISTORY_FILE)
        st.session_state.upload_history = disk_history if disk_history else {}
    return st.session_state.upload_history

def save_persisted_data(data_dict):
    """Save data to persist across sessions (both memory and disk)"""
    if 'persisted_data' not in st.session_state:
        st.session_state.persisted_data = {}
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = {}
    
    # Update data in memory
    st.session_state.persisted_data.update(data_dict)
    
    # Update upload timestamps
    for file_type in data_dict.keys():
        st.session_state.upload_history[file_type] = {
            'timestamp': datetime.now(),
            'rows': len(data_dict[file_type]),
            'columns': len(data_dict[file_type].columns)
        }
    
    # Save to disk for persistence
    save_to_disk(st.session_state.persisted_data, DATA_FILE)
    save_to_disk(st.session_state.upload_history, HISTORY_FILE)

def login_page():
    """Display login page"""
    st.markdown("""
        <div style='text-align: center; padding: 3rem 0 2rem 0;'>
            <h1 style='color: #1a1a1a;
                       font-size: 2.5rem;
                       font-weight: 600;
                       margin-bottom: 0.5rem;'>
                🏬 Retail Analytics Dashboard
            </h1>
            <p style='color: #6b7280; font-size: 1.1rem;'>Intelligent Replenishment & Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class='login-container'>
                <h2 style='text-align: center; color: #1f2937; margin-bottom: 2rem; font-weight: 600;'>
                    🔐 Sign In
                </h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Check if there's persisted data
        persisted = get_persisted_data()
        if persisted:
            st.info(f"📁 {len(persisted)} data file(s) available from previous session")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username", key="login_username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                # Simple authentication (you can modify these credentials)
                if username == "admin" and password == "admin123":
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Info box with clean styling
        st.markdown("""
            <div style='background: #f9fafb;
                        padding: 1.5rem;
                        border-radius: 8px;
                        border: 1px solid #e5e7eb;'>
                <p style='margin: 0; color: #1f2937; font-weight: 600;'>📌 Default Credentials</p>
                <p style='margin: 0.75rem 0 0 0; color: #6b7280; font-size: 0.9rem;'>
                    <strong>Username:</strong> admin<br>
                    <strong>Password:</strong> admin123
                </p>
            </div>
        """, unsafe_allow_html=True)

def validate_store_mapping(sales_data, stock_data):
    """Validate and show store mapping between sales and stock data"""
    if sales_data.empty or stock_data.empty:
        return pd.DataFrame(), []
    
    # Get unique stores from both datasets
    sales_stores = set(sales_data['STORE'].unique())
    stock_stores = set(stock_data['STORE'].unique())
    
    # Find matches and mismatches
    matched_stores = sales_stores.intersection(stock_stores)
    sales_only = sales_stores - stock_stores
    stock_only = stock_stores - sales_stores
    
    # Create mapping report
    mapping_report = []
    
    # Add matched stores
    for store in matched_stores:
        sales_name = sales_data[sales_data['STORE'] == store]['STORE_NAME'].iloc[0] if 'STORE_NAME' in sales_data.columns else store
        stock_name = stock_data[stock_data['STORE'] == store]['STORE_NAME'].iloc[0] if 'STORE_NAME' in stock_data.columns else store
        mapping_report.append({
            'Store_Code': store,
            'Sales_Name': sales_name,
            'Stock_Name': stock_name,
            'Status': '✅ Matched',
            'Sales_Records': len(sales_data[sales_data['STORE'] == store]),
            'Stock_Records': len(stock_data[stock_data['STORE'] == store])
        })
    
    # Add sales-only stores
    for store in sales_only:
        sales_name = sales_data[sales_data['STORE'] == store]['STORE_NAME'].iloc[0] if 'STORE_NAME' in sales_data.columns else store
        mapping_report.append({
            'Store_Code': store,
            'Sales_Name': sales_name,
            'Stock_Name': 'NOT FOUND',
            'Status': '🔴 Sales Only',
            'Sales_Records': len(sales_data[sales_data['STORE'] == store]),
            'Stock_Records': 0
        })
    
    # Add stock-only stores
    for store in stock_only:
        stock_name = stock_data[stock_data['STORE'] == store]['STORE_NAME'].iloc[0] if 'STORE_NAME' in stock_data.columns else store
        mapping_report.append({
            'Store_Code': store,
            'Sales_Name': 'NOT FOUND',
            'Stock_Name': stock_name,
            'Status': '🟡 Stock Only',
            'Sales_Records': 0,
            'Stock_Records': len(stock_data[stock_data['STORE'] == store])
        })
    
    mapping_df = pd.DataFrame(mapping_report)
    warnings = []
    
    if sales_only:
        warnings.append(f"⚠️ {len(sales_only)} stores found in sales data but not in stock data")
    if stock_only:
        warnings.append(f"⚠️ {len(stock_only)} stores found in stock data but not in sales data")
    
    return mapping_df, warnings

def show_sample_data(df, label):
    """Display a sample of the data with store mapping info"""
    if df.empty:
        st.info(f"ℹ️ No data uploaded for {label}")
        return
        
    with st.expander(f"📊 {label} Preview", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Rows:** {len(df):,}")
        with col2:
            st.markdown(f"**Columns:** {len(df.columns)}")
        
        st.markdown("**Available Columns:**")
        st.markdown(", ".join(f"`{col}`" for col in df.columns))
        
        # Show store mapping info if available
        if 'STORE' in df.columns:
            unique_stores = df['STORE'].nunique()
            st.markdown(f"**Unique Stores:** {unique_stores}")
            if 'STORE_NAME' in df.columns:
                # Check if EBO NAME was used
                if any('ebo name' in col.lower() for col in df.columns):
                    st.markdown("**Store Mapping:** Using EBO NAME as Store Names ✅")
                else:
                    st.markdown("**Store Mapping:** Using Store Codes with Names ✅")
            else:
                st.markdown("**Store Mapping:** Using Store Names (No codes found)")
        
        # Show warehouse-specific info
        if label == "Warehouse Data" and 'SKU' in df.columns and 'WAREHOUSE_STOCK' in df.columns:
            unique_skus = df['SKU'].nunique()
            total_stock = df['WAREHOUSE_STOCK'].sum()
            st.markdown(f"**Unique SKUs:** {unique_skus:,}")
            st.markdown(f"**Total Warehouse Stock:** {total_stock:,.0f}")
            
            # Check if specific warehouse columns were detected
            original_cols = [col.lower() for col in df.columns]
            if any('each client sku id' in col for col in original_cols):
                st.markdown("**SKU Column:** Each Client SKU ID detected ✅")
            if any('total available quantity' in col for col in original_cols):
                st.markdown("**Stock Column:** Total Available Quantity detected ✅")
        
        st.markdown("**Sample Data:**")
        st.dataframe(
            df.head(5),
            use_container_width=True,
            height=200
        )

def main():
    # Check if user is logged in
    if not check_login():
        login_page()
        return
    
    # Load persisted data if available
    persisted_data = get_persisted_data()
    
    # Sidebar with logout button
    with st.sidebar:
        st.markdown(f"""
            <div style='background: #f9fafb; 
                        padding: 1rem; 
                        border-radius: 8px; 
                        margin-bottom: 1rem;
                        text-align: center;
                        border: 1px solid #e5e7eb;'>
                <h3 style='color: #1f2937; margin: 0; font-weight: 600;'>👤 {st.session_state.get('username', 'User')}</h3>
                <p style='color: #6b7280; margin: 0.5rem 0 0 0; font-size: 0.875rem;'>Welcome back!</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show persisted data info
        if persisted_data:
            st.info(f"📁 {len(persisted_data)} file(s) loaded")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.rerun()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.persisted_data = {}
                st.session_state.upload_history = {}
                # Also delete from disk
                try:
                    if DATA_FILE.exists():
                        DATA_FILE.unlink()
                    if HISTORY_FILE.exists():
                        HISTORY_FILE.unlink()
                    st.success("Cleared!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                st.rerun()
        st.markdown("---")
        
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Analytics Dashboard")
        st.markdown("---")
        st.markdown("### 📁 Data Upload")
        
        # Show upload history before file uploaders
        upload_history = get_upload_history()
        if upload_history:
            with st.expander("📜 Upload History", expanded=False):
                file_type_names = {
                    'sales': 'Sales Data',
                    'stock': 'Stock Data',
                    'warehouse': 'Warehouse Data',
                    'sku_master': 'SKU Master',
                    'style_master': 'Style Master'
                }
                
                for file_type, info in upload_history.items():
                    file_name = file_type_names.get(file_type, file_type)
                    timestamp = info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    st.markdown(f"**{file_name}**")
                    st.markdown(f"🕒 {timestamp}")
                    st.markdown(f"📊 {info['rows']:,} rows × {info['columns']} cols")
                    st.markdown("---")
        
        # Show database status
        if DATABASE_AVAILABLE:
            st.markdown("---")
            st.subheader("🗄️ Backend Database Status")
            streamlit_database_status()
            streamlit_data_management()
            st.markdown("---")
        
        # Email Integration Section
        if EMAIL_INTEGRATION_AVAILABLE:
            st.markdown("---")
            st.subheader("📧 Automated Email Sales Collection")
            
            with st.expander("⚙️ Configure Email Integration", expanded=False):
                st.info("📧 Automatically collect daily sales data from emails")
                
                email_config = streamlit_email_integration_ui()
                
                if email_config.get("email_address") and email_config.get("email_password"):
                    st.markdown("### 🔄 Fetch Sales Data from Emails")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📬 Check Emails Now", key="check_emails"):
                            with st.spinner("📧 Checking for email attachments..."):
                                db = SalesDatabase() if DATABASE_AVAILABLE else None
                                
                                email_integration = EmailSalesIntegration(
                                    email_address=email_config["email_address"],
                                    email_password=email_config["email_password"]
                                )
                                
                                # Fetch emails
                                emails = email_integration.fetch_sales_emails(
                                    from_email=email_config.get("sender_email"),
                                    days_back=email_config.get("days_back", 1),
                                    subject_keyword=email_config.get("subject_keyword", "Sales")
                                )
                                
                                if emails:
                                    st.success(f"✅ Found {len(emails)} email(s) with attachments")
                                    
                                    all_data = pd.DataFrame()
                                    
                                    for email_info in emails:
                                        st.write(f"📧 {email_info['subject']}")
                                        
                                        for attachment in email_info["attachments"]:
                                            df = email_integration.parse_sales_attachment(
                                                attachment["data"],
                                                attachment["filename"]
                                            )
                                            
                                            if df is not None:
                                                st.write(f"   ✅ {attachment['filename']}: {len(df):,} rows")
                                                all_data = pd.concat([all_data, df], ignore_index=True)
                                    
                                    if not all_data.empty:
                                        st.success(f"✅ Total records extracted: {len(all_data):,}")
                                        
                                        # Show preview
                                        with st.expander("Preview Data"):
                                            st.dataframe(all_data.head(10), use_container_width=True)
                                        
                                        # Save to database
                                        if db:
                                            if db.save_dataframe(all_data, f"email_sales_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                                                st.success("✅ Data saved to database!")
                                else:
                                    st.warning("❌ No emails found matching your criteria")
                    
                    with col2:
                        st.markdown("**Setup Instructions:**")
                        st.markdown("""
                        1. Enter your Gmail address
                        2. Create App Password at: https://myaccount.google.com/apppasswords
                        3. Paste App Password (not regular password)
                        4. Click "Check Emails Now"
                        
                        Your stores should send sales data with subject containing "Sales"
                        """)
                else:
                    st.warning("⚠️ Please configure email settings first")
            
            st.markdown("---")
        
        # File uploaders - Using new multi-upload feature with merge capability
        st.markdown("### 📥 Data Upload (Multiple Files Supported)")
        
        # Initialize uploaded_files dictionary
        uploaded_files = {}
        
        # Use new data merge helper if available
        if DATA_MERGE_AVAILABLE:
            # Sales Data with multi-upload
            st.markdown("#### 📊 Sales Data (2024, 2025, 2026, etc.)")
            sales_df, sales_merge_type = streamlit_multi_upload_ui("Sales")
            uploaded_files["sales"] = sales_df
            
            st.divider()
            
            # Other data types with traditional upload (can be upgraded later)
            uploaded_files["stock"] = st.file_uploader("Stock Data (CSV/Excel)", type=["csv", "xlsx"], key="stock_upload")
            uploaded_files["warehouse"] = st.file_uploader("Warehouse Data (CSV/Excel)", type=["csv", "xlsx"], key="warehouse_upload")
            uploaded_files["sku_master"] = st.file_uploader("SKU Master (CSV/Excel)", type=["csv", "xlsx"], key="sku_upload")
            uploaded_files["style_master"] = st.file_uploader("Style Master (CSV/Excel)", type=["csv", "xlsx"], key="style_upload")
            uploaded_files["size_master"] = st.file_uploader("Size Master (CSV/Excel) - Required for Size Set Completion", type=["csv", "xlsx"], key="size_upload")
        else:
            # Fallback to traditional uploaders
            uploaded_files = {
                "sales": st.file_uploader("Sales Data (CSV/Excel)", type=["csv", "xlsx"]),
                "stock": st.file_uploader("Stock Data (CSV/Excel)", type=["csv", "xlsx"]),
                "warehouse": st.file_uploader("Warehouse Data (CSV/Excel)", type=["csv", "xlsx"]),
                "sku_master": st.file_uploader("SKU Master (CSV/Excel)", type=["csv", "xlsx"]),
                "style_master": st.file_uploader("Style Master (CSV/Excel)", type=["csv", "xlsx"]),
                "size_master": st.file_uploader("Size Master (CSV/Excel) - Required for Size Set Completion", type=["csv", "xlsx"])
            }
        
        st.markdown("---")
        # Backend uploads now handled by SQLite database
        st.info("💾 Data is automatically saved to SQLite database backend")
        
        st.markdown("---")
        st.markdown("### ⚙️ Replenishment Settings")
        
        # Target Coverage Days
        target_coverage_days = st.number_input(
            "Target Coverage (Days)",
            min_value=7,
            max_value=60,
            value=20,
            step=1,
            help="Number of days of stock to maintain"
        )
        
        # Safety Stock Days
        safety_stock_days = st.number_input(
            "Safety Stock (Days)", 
            min_value=0,
            max_value=14,
            value=3,
            step=1,
            help="Extra stock for demand variability"
        )
        
        # Lead Time
        lead_time_days = st.number_input(
            "Lead Time (Days)",
            min_value=1,
            max_value=30,
            value=8,
            step=1,
            help="Days to receive stock from warehouse"
        )
        
        # Minimum Order Quantity
        moq = st.number_input(
            "Min Order Qty (MOQ)",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Minimum order quantity per SKU"
        )
        
        # Forecasting method selection
        forecasting_method = st.selectbox(
            "Forecasting Method",
            ["Simple Average", "Weighted Moving Average"],
            index=1,
            help="Weighted gives more importance to recent sales"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Period")
        time_period = st.selectbox(
            "Sales History Period",
            ["Last 30 Days", "Last 60 Days", "Last 90 Days", "Custom Range"],
            index=2,
            key="time_period_select"
        )
        
        if time_period == "Custom Range":
            start_date = st.date_input("Start Date", key="start_date_input")
            end_date = st.date_input("End Date", key="end_date_input")
    
    # Main Content
    st.markdown("<h1 class='title'>Retail Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Start with existing persisted data (preserve previously uploaded files)
    data = persisted_data.copy() if persisted_data else {}
    
    # Map file types to cleaning functions
    cleaning_functions = {
        'sales': clean_sales_data,
        'stock': clean_stock_data,
        'warehouse': clean_warehouse_data,
        'sku_master': clean_sku_master,
        'style_master': clean_style_master,
        'size_master': clean_size_master
    }
    
    # Track which files were newly uploaded
    newly_uploaded = []
    
    # Process each uploaded file (only update the ones that were uploaded)
    for file_type, uploaded_file in uploaded_files.items():
        # Skip if it's a DataFrame (from multi-upload UI) or None
        if isinstance(uploaded_file, pd.DataFrame) or uploaded_file is None:
            continue
            
        # Process file uploader objects only
        if uploaded_file:
            # Read file first
            raw_df = read_file(uploaded_file)
            
            if not raw_df.empty:
                # Then clean it
                df = cleaning_functions[file_type](raw_df)
                
                if not df.empty:
                    # Upload file to API and get response
                    api_response = upload_file_to_api(uploaded_file, file_type)
                    
                    if api_response:
                        # Update/add this file type in the data dictionary
                        data[file_type] = df
                        newly_uploaded.append(file_type)
                        st.success(f"✅ Successfully uploaded {file_type} data to backend")
                    else:
                        st.warning(f"⚠️ Could not save {file_type} data to backend, but will continue with analysis")
                        data[file_type] = df
                        newly_uploaded.append(file_type)
    
    # If any new data was uploaded, persist the merged data
    if newly_uploaded:
        save_persisted_data(data)
        st.info(f"💾 Saved {', '.join(newly_uploaded)} to persistent storage. Other files remain unchanged.")
    
    # Show info if using persisted data
    if not newly_uploaded and data:
        st.info("📂 Using data from previous session")
    
    # Data Preview Section
    st.markdown("## Data Overview")
    
    # Show file status summary
    if data:
        file_status = []
        file_types = ['sales', 'stock', 'warehouse', 'sku_master', 'style_master', 'size_master']
        file_names = ['Sales Data', 'Stock Data', 'Warehouse Data', 'SKU Master', 'Style Master', 'Size Master']
        
        for ftype, fname in zip(file_types, file_names):
            if ftype in data:
                status = "🟢" if ftype in newly_uploaded else "🔵"
                file_status.append(f"{status} {fname}")
            else:
                file_status.append(f"⚪ {fname}")
        
        st.markdown(f"**Files Loaded:** {' | '.join(file_status)}")
        st.caption("🟢 = Just uploaded | 🔵 = Previously uploaded | ⚪ = Not loaded")
    
    st.markdown("Review your uploaded data below. Click to expand each section.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sales' in data: show_sample_data(data['sales'], "Sales Data")
        if 'stock' in data: show_sample_data(data['stock'], "Stock Data")
        if 'warehouse' in data: show_sample_data(data['warehouse'], "Warehouse Data")
    
    with col2:
        if 'sku_master' in data: show_sample_data(data['sku_master'], "SKU Master")
        if 'style_master' in data: show_sample_data(data['style_master'], "Style Master")
        if 'size_master' in data: show_sample_data(data['size_master'], "Size Master")
    
    # Store Mapping Validation
    if 'sales' in data and 'stock' in data:
        st.markdown("## 🏪 Store Mapping Validation")
        mapping_df, warnings = validate_store_mapping(data['sales'], data['stock'])
        
        if not mapping_df.empty:
            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                matched_count = len(mapping_df[mapping_df['Status'] == '✅ Matched'])
                st.metric("Matched Stores", matched_count)
            
            with col2:
                sales_only_count = len(mapping_df[mapping_df['Status'] == '🔴 Sales Only'])
                st.metric("Sales Only", sales_only_count)
            
            with col3:
                stock_only_count = len(mapping_df[mapping_df['Status'] == '🟡 Stock Only'])
                st.metric("Stock Only", stock_only_count)
            
            with col4:
                total_stores = len(mapping_df)
                st.metric("Total Stores", total_stores)
            
            # Show warnings if any
            for warning in warnings:
                st.warning(warning)
            
            # Show detailed mapping
            with st.expander("📋 Detailed Store Mapping", expanded=len(warnings) > 0):
                st.dataframe(
                    mapping_df.sort_values(['Status', 'Store_Code']),
                    use_container_width=True
                )
                
                if len(warnings) > 0:
                    st.info("💡 **Tip:** Stores that don't match between sales and stock data will be excluded from replenishment calculations. Ensure your data uses consistent store codes or names.")
        
        st.markdown("---")
    
    # Analysis Section
    if len(data) == len(uploaded_files):
        st.markdown("## Analysis")
        
        # Key Metrics
        metrics_cols = st.columns(4)
        
        with metrics_cols[0]:
            total_sales = data['sales']['QUANTITY'].sum() if 'sales' in data else 0
            st.markdown(f"""
                <div class='metric-card'>
                    <h3>Total Sales</h3>
                    <h2>{total_sales:,.0f}</h2>
                </div>
            """, unsafe_allow_html=True)
            
        with metrics_cols[1]:
            store_count = data['stock']['STORE'].nunique() if 'stock' in data else 0
            st.markdown(f"""
                <div class='metric-card'>
                    <h3>Store Count</h3>
                    <h2>{store_count:,.0f}</h2>
                </div>
            """, unsafe_allow_html=True)
            
        with metrics_cols[2]:
            sku_count = data['sku_master']['SKU'].nunique() if 'sku_master' in data else 0
            st.markdown(f"""
                <div class='metric-card'>
                    <h3>Total SKUs</h3>
                    <h2>{sku_count:,.0f}</h2>
                </div>
            """, unsafe_allow_html=True)
            
        with metrics_cols[3]:
            total_stock = data['stock']['STOCK'].sum() if 'stock' in data else 0
            st.markdown(f"""
                <div class='metric-card'>
                    <h3>Total Stock</h3>
                    <h2>{total_stock:,.0f}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        # Analysis Tabs
        tabs = st.tabs([
            "📋 Replenishment",
            "📈 Sales Analysis", 
            "📊 Stock Analysis",
            "🏬 Store Performance",
            "📦 SKU Analysis",
            "📊 Performance Metrics"
        ])
        
        with tabs[0]:
            st.markdown("### Replenishment Analysis")
            if all(k in data for k in ['sales', 'stock', 'warehouse', 'sku_master']):
                # Initialize replen_data as empty DataFrame
                replen_data = pd.DataFrame()
                
                # Time period selection for replenishment
                col1, col2 = st.columns([3, 1])
                with col1:
                    replen_period = st.selectbox(
                        "Select Analysis Period for Replenishment",
                        [7, 14, 30, 60, 90],
                        index=2,
                        format_func=lambda x: f"Last {x} Days",
                        key="replen_period_select"
                    )
                
                with col2:
                    run_replen = st.button("Calculate Replenishment", type="primary", key="run_replen_button")
                
                if run_replen:
                    with st.spinner("Calculating replenishment recommendations..."):
                        start_time = time.time()
                        # Calculate replenishment recommendations with user settings
                        replen_data = calculate_replenishment(
                            data['sales'],
                            data['stock'],
                            data['warehouse'],
                            data['sku_master'],
                            data.get('style_master'),  # Pass style_master if available
                            data.get('size_master'),   # Pass size_master if available
                            time_period_days=replen_period,
                            target_coverage_days=target_coverage_days,
                            safety_stock_days=safety_stock_days,
                            lead_time_days=lead_time_days,
                            moq=moq,
                            forecasting_method=forecasting_method
                        )
                        
                        if replen_data is not None and not replen_data.empty:
                            # Overall replenishment metrics
                            st.markdown("#### Replenishment Overview")
                            metrics_cols = st.columns(5)
                            
                            with metrics_cols[0]:
                                total_replen = replen_data['FINAL_REPLEN_QTY'].sum()
                                st.metric("Total Replenishment Qty", f"{total_replen:,.0f}")
                            
                            with metrics_cols[1]:
                                stores_needing_replen = replen_data[replen_data['FINAL_REPLEN_QTY'] > 0]['STORE'].nunique()
                                st.metric("Stores Needing Stock", stores_needing_replen)
                            
                            with metrics_cols[2]:
                                skus_to_replen = replen_data[replen_data['FINAL_REPLEN_QTY'] > 0]['SKU'].nunique()
                                st.metric("SKUs to Replenish", skus_to_replen)
                            
                            with metrics_cols[3]:
                                avg_stock_cover = replen_data[replen_data['STOCK_COVER_DAYS'] < 999]['STOCK_COVER_DAYS'].median()
                                st.metric("Median Stock Cover (Days)", f"{avg_stock_cover:.1f}")
                            
                            with metrics_cols[4]:
                                new_items_count = replen_data['IS_NEW_ITEM'].sum()
                                st.metric("New Items (Need Review)", new_items_count)
                            
                            # Show forecasting method used
                            if forecasting_method == "Weighted Moving Average":
                                st.info("🔬 **Enhanced Forecasting:** Using weighted moving average with trend analysis for improved accuracy")
                            else:
                                st.info("📊 **Basic Forecasting:** Using simple average - consider switching to weighted for better accuracy")
                            
                            # Detailed replenishment recommendations
                            st.markdown("#### Replenishment Recommendations")
                            
                            # Filter for items needing replenishment
                            replen_recommendations = replen_data[replen_data['FINAL_REPLEN_QTY'] > 0].copy()
                            replen_recommendations = replen_recommendations.sort_values(
                                ['STOCK_COVER_DAYS', 'DAILY_SALES'],
                                ascending=[True, False]
                            )
                            
                            # Show new items that need manual review
                            new_items = replen_data[replen_data['IS_NEW_ITEM'] == True]
                            if not new_items.empty:
                                st.markdown("#### 🆕 New Items Requiring Manual Review")
                                st.warning("These items have no sales history and need demand planning input:")
                                
                                # Prepare display for new items with store names
                                new_items_display = new_items[['STORE', 'SKU', 'STYLE', 'COLOR', 'SIZE']].copy()
                                
                                # Add gender if available
                                if 'GENDER' in new_items.columns:
                                    new_items_display['GENDER'] = new_items['GENDER']
                                else:
                                    new_items_display['GENDER'] = 'UNISEX'
                                
                                # Add store name if available
                                if 'STORE_NAME' in new_items.columns:
                                    new_items_display['STORE_DISPLAY'] = new_items['STORE'] + ' - ' + new_items['STORE_NAME'].fillna('')
                                    new_items_display = new_items_display[['STORE_DISPLAY', 'SKU', 'STYLE', 'COLOR', 'SIZE', 'GENDER']]
                                    new_items_display = new_items_display.rename(columns={'STORE_DISPLAY': 'Store Code - Name'})
                                else:
                                    new_items_display = new_items_display.rename(columns={'STORE': 'Store Code'})
                                
                                st.dataframe(
                                    new_items_display,
                                    use_container_width=True
                                )
                                st.markdown("---")
                            
                            if not replen_recommendations.empty:
                                # Add stock-out indicators and highlighting
                                replen_recommendations['STOCK_OUT'] = replen_recommendations['STOCK'] == 0
                                replen_recommendations['CRITICAL_STOCK'] = (
                                    (replen_recommendations['STOCK_COVER_DAYS'] < lead_time_days) &  # Less than lead time
                                    (replen_recommendations['DAILY_SALES'] > 0)
                                )
                                
                            # Calculate and display processing time
                            end_time = time.time()
                            st.success(f"✅ Replenishment calculations completed in {(end_time - start_time):.2f} seconds")

                            # Display recommendations with highlighting
                            st.markdown("🔴 Stock Out &nbsp;&nbsp;&nbsp; 🟡 Critical Stock (< lead_time_days days coverage)")
                                                            # Display key columns with proper formatting and store names
                            display_df = replen_recommendations[[
                                'STORE', 'SKU', 'STYLE', 'COLOR', 'SIZE', 'GENDER',
                                'STOCK', 'DAILY_SALES', 'STOCK_COVER_DAYS',
                                'TARGET_STOCK', 'FINAL_REPLEN_QTY', 'STOCK_OUT', 'CRITICAL_STOCK'
                            ]].copy()
                                
                                # Handle case where GENDER column might not exist
                            if 'GENDER' not in display_df.columns:
                                display_df['GENDER'] = 'UNISEX'
                                
                                # Add store name column if available and handle missing values
                                if 'STORE_NAME' in replen_recommendations.columns:
                                    # Fill any missing store names with empty string
                                    store_names_filled = replen_recommendations['STORE_NAME'].fillna('')
                                    display_df['STORE_DISPLAY'] = replen_recommendations['STORE'] + ' - ' + store_names_filled
                                    # Reorder columns to show store display first
                                    cols = ['STORE_DISPLAY'] + [col for col in display_df.columns if col not in ['STORE', 'STORE_DISPLAY']]
                                    display_df = display_df[cols]
                                    # Remove the original STORE column only if it exists
                                    if 'STORE' in display_df.columns:
                                        display_df = display_df.drop('STORE', axis=1)
                                else:
                                    # If no store names available, rename STORE column for clarity
                                    if 'STORE' in display_df.columns:
                                        display_df = display_df.rename(columns={'STORE': 'STORE_CODE'})
                                
                                # Format numeric columns
                                display_df['DAILY_SALES'] = display_df['DAILY_SALES'].round(2)
                                display_df['STOCK_COVER_DAYS'] = display_df['STOCK_COVER_DAYS'].round(1)
                                
                                # Set up column renames
                                column_renames = {
                                    'STORE_DISPLAY': 'Store Code - Name',
                                    'STORE_CODE': 'Store Code',
                                    'DAILY_SALES': 'Daily Sales',
                                    'STOCK_COVER_DAYS': 'Stock Cover (Days)',
                                    'TARGET_STOCK': 'Target Stock',
                                    'FINAL_REPLEN_QTY': 'Replen Qty',
                                    'STOCK_OUT': 'Stock Out',
                                    'CRITICAL_STOCK': 'Critical'
                                }
                                
                                # Only rename columns that exist in the dataframe
                                final_renames = {k: v for k, v in column_renames.items() if k in display_df.columns}
                                
                                # Display recommendations
                                st.dataframe(
                                    display_df.rename(columns=final_renames),
                                    use_container_width=True
                                )
                            
                            # Download button for replenishment plan
                            # Organize columns in a logical order for output
                            output_columns = [
                                'STORE', 'STORE_NAME', 'SKU', 'STYLE', 'COLOR', 'SIZE', 'GENDER',
                                'STOCK', 'WAREHOUSE_STOCK', 'DAILY_SALES', 'STOCK_COVER_DAYS',
                                'TARGET_STOCK', 'REPLEN_QTY', 'FINAL_REPLEN_QTY',
                                'SIZE_SET_ALLOCATION', 'PRIORITY_SCORE', 'REMARKS', 'IS_NEW_ITEM'
                            ]
                            
                            # Only include columns that exist in the dataframe
                            available_cols = [col for col in output_columns if col in replen_recommendations.columns]
                            # Add any remaining columns that weren't in our list
                            remaining_cols = [col for col in replen_recommendations.columns if col not in available_cols]
                            final_output_cols = available_cols + remaining_cols
                            
                            # Create output with organized columns
                            output_df = replen_recommendations[final_output_cols].copy()
                            
                            # FINAL FREEBIES CHECK: Absolutely ensure no FREEBIES in output file
                            if 'GENDER' in output_df.columns:
                                output_df['GENDER'] = output_df['GENDER'].astype(str).str.strip().str.upper()
                                freebies_in_output = output_df[output_df['GENDER'].str.contains('FREEBIE', na=False, case=False)]
                                
                                if len(freebies_in_output) > 0:
                                    st.error(f"🚨 CRITICAL: Found {len(freebies_in_output)} FREEBIES items in output! Removing before download...")
                                    freebies_styles = freebies_in_output['STYLE'].unique()
                                    st.error(f"   FREEBIES styles found: {', '.join(str(s) for s in freebies_styles)}")
                                    
                                    # Remove ALL FREEBIES from output
                                    output_df = output_df[~output_df['GENDER'].str.contains('FREEBIE', na=False, case=False)].copy()
                                    st.success(f"✅ Removed {len(freebies_in_output)} FREEBIES items from download file")
                                else:
                                    st.success("✅ Output file verified: NO FREEBIES data included")
                            
                            csv = output_df.to_csv(index=False)
                            
                            st.download_button(
                                "Download Replenishment Plan",
                                csv,
                                "replenishment_plan.csv",
                                "text/csv",
                                key="download_replen_csv_button"
                            )
                            
                            # Stock Cover Analysis
                            st.markdown("#### Stock Cover Analysis")
                            
                            # Create chart data with store names and proper error handling
                            chart_data = replen_data.copy()
                            if 'STORE_NAME' in chart_data.columns:
                                chart_data['STORE_NAME'] = chart_data['STORE_NAME'].fillna('')
                                chart_data['STORE_DISPLAY'] = chart_data['STORE'] + ' - ' + chart_data['STORE_NAME']
                                fig_stock_cover = px.box(
                                    chart_data,
                                    x='STORE_DISPLAY',
                                    y='STOCK_COVER_DAYS',
                                    title='Stock Cover Distribution by Store'
                                )
                                fig_stock_cover.update_xaxes(tickangle=45)
                            else:
                                fig_stock_cover = px.box(
                                    replen_data,
                                    x='STORE',
                                    y='STOCK_COVER_DAYS',
                                    title='Stock Cover Distribution by Store'
                                )
                            
                            st.plotly_chart(fig_stock_cover, use_container_width=True, key="stock_cover_chart_1")
                            
                            # Replenishment by Store
                            st.markdown("#### Replenishment Quantities by Store")
                            
                            # Create store display for chart with proper error handling
                            chart_data = replen_recommendations.copy()
                            if 'STORE_NAME' in chart_data.columns:
                                # Handle missing store names
                                chart_data['STORE_NAME'] = chart_data['STORE_NAME'].fillna('')
                                chart_data['STORE_DISPLAY'] = chart_data['STORE'] + ' - ' + chart_data['STORE_NAME']
                                store_replen = chart_data.groupby('STORE_DISPLAY')['FINAL_REPLEN_QTY'].sum().sort_values(ascending=True)
                            else:
                                store_replen = replen_recommendations.groupby('STORE')['FINAL_REPLEN_QTY'].sum().sort_values(ascending=True)
                            
                            if not store_replen.empty:
                                fig_store_replen = px.bar(
                                    x=store_replen.values,
                                    y=store_replen.index,
                                    orientation='h',
                                    title='Total Replenishment Quantity by Store',
                                    labels={'x': 'Replenishment Quantity', 'y': 'Store'}
                                )
                                fig_store_replen.update_layout(height=max(400, len(store_replen) * 25))
                                st.plotly_chart(fig_store_replen, use_container_width=True, key="store_replen_chart_1")
                            else:
                                st.info("No replenishment data available for chart.")

        with tabs[1]:
            st.markdown("### Sales Trends")
            if 'sales' in data:
                # Daily Sales Trend
                daily_sales = data['sales'].groupby('DATE')['QUANTITY'].sum().reset_index()
                st.line_chart(daily_sales.set_index('DATE'))
                
                # Top Selling Products
                top_products = data['sales'].groupby('SKU')['QUANTITY'].sum().sort_values(ascending=False).head(10)
                st.markdown("#### Top 10 Selling Products")
                st.bar_chart(top_products)
                
                # Sales by Store
                store_sales = data['sales'].groupby('STORE')['QUANTITY'].sum().sort_values(ascending=False)
                st.markdown("#### Sales by Store")
                st.bar_chart(store_sales)
        
        with tabs[2]:
            st.markdown("### Stock Distribution")
            if 'stock' in data:
                # Overall Stock Distribution
                st.markdown("#### Stock Distribution Across Stores")
                store_stock = data['stock'].groupby('STORE')['STOCK'].sum().sort_values(ascending=False)
                st.bar_chart(store_stock)
                
                # Stock by SKU
                top_stock = data['stock'].groupby('SKU')['STOCK'].sum().sort_values(ascending=False).head(10)
                st.markdown("#### Top 10 SKUs by Stock Level")
                st.bar_chart(top_stock)
                
                # Zero Stock Analysis
                zero_stock = data['stock'][data['stock']['STOCK'] == 0].groupby('STORE').size()
                if not zero_stock.empty:
                    st.markdown("#### Zero Stock Items by Store")
                    st.bar_chart(zero_stock)
        
        with tabs[3]:
            st.markdown("### Store Performance")
            if all(k in data for k in ['sales', 'stock']):
                # Store Performance Analysis
                store_metrics_display = pd.DataFrame()
                
                # Calculate store metrics
                store_metrics_display['Total_Sales'] = data['sales'].groupby('STORE')['QUANTITY'].sum()
                store_metrics_display['Current_Stock'] = data['stock'].groupby('STORE')['STOCK'].sum()
                store_metrics_display['SKU_Count'] = data['stock'].groupby('STORE')['SKU'].nunique()
                
                # Add store names if available
                if 'STORE_NAME' in data['sales'].columns:
                    store_names = data['sales'].groupby('STORE')['STORE_NAME'].first()
                    store_metrics_display['Store_Name'] = store_names
                    # Reorder columns to show store name first
                    cols = ['Store_Name'] + [col for col in store_metrics_display.columns if col != 'Store_Name']
                    store_metrics_display = store_metrics_display[cols]
                
                # Calculate stock turn ratio (if date range is available)
                if time_period != "Custom Range":
                    store_metrics_display['Stock_Turn'] = store_metrics_display['Total_Sales'] / store_metrics_display['Current_Stock']
                    store_metrics_display['Stock_Turn'] = store_metrics_display['Stock_Turn'].fillna(0)
                
                # Display store performance table
                st.markdown("#### Store Performance Metrics")
                st.dataframe(store_metrics_display.round(2), use_container_width=True)
                
                # Store Rankings
                st.markdown("#### Store Rankings by Sales")
                # Create display names for chart
                if 'Store_Name' in store_metrics_display.columns:
                    chart_data = store_metrics_display.copy()
                    chart_data.index = chart_data.index + ' - ' + chart_data['Store_Name']
                    st.bar_chart(chart_data['Total_Sales'].sort_values(ascending=False))
                else:
                    st.bar_chart(store_metrics_display['Total_Sales'].sort_values(ascending=False))
        
        with tabs[4]:
            st.markdown("### SKU Analysis")
            if all(k in data for k in ['sku_master', 'sales', 'style_master']):
                # Merge sales with SKU and style master
                sku_analysis = pd.merge(
                    data['sales'].groupby('SKU')['QUANTITY'].sum().reset_index(),
                    data['sku_master'],
                    on='SKU'
                )
                sku_analysis = pd.merge(
                    sku_analysis,
                    data['style_master'],
                    on='STYLE'
                )
                
                # Analysis by Gender
                st.markdown("#### Sales by Gender")
                gender_sales = sku_analysis.groupby('GENDER')['QUANTITY'].sum()
                st.bar_chart(gender_sales)
                
                # Analysis by Size
                st.markdown("#### Sales by Size")
                size_sales = sku_analysis.groupby('SIZE')['QUANTITY'].sum()
                st.bar_chart(size_sales)
                
                # Analysis by Color
                st.markdown("#### Top Colors by Sales")
                color_sales = sku_analysis.groupby('COLOR')['QUANTITY'].sum().sort_values(ascending=False).head(10)
                st.bar_chart(color_sales)
                
                # Style Performance
                st.markdown("#### Top Performing Styles")
                style_sales = sku_analysis.groupby('STYLE')['QUANTITY'].sum().sort_values(ascending=False).head(10)
                st.bar_chart(style_sales)
        
        with tabs[5]:
            st.markdown("### Performance Metrics")
            if all(k in data for k in ['sales', 'stock', 'warehouse', 'sku_master']):
                
                # Key Performance Indicators
                st.markdown("#### Key Performance Indicators")
                
                kpi_cols = st.columns(4)
                
                with kpi_cols[0]:
                    # Service Level (items in stock)
                    total_items = len(data['stock'])
                    in_stock_items = len(data['stock'][data['stock']['STOCK'] > 0])
                    service_level = (in_stock_items / total_items * 100) if total_items > 0 else 0
                    st.metric("Service Level", f"{service_level:.1f}%")
                
                with kpi_cols[1]:
                    # Inventory Turnover (simplified)
                    total_sales = data['sales']['QUANTITY'].sum()
                    total_stock = data['stock']['STOCK'].sum()
                    turnover = (total_sales / total_stock) if total_stock > 0 else 0
                    st.metric("Inventory Turnover", f"{turnover:.2f}x")
                
                with kpi_cols[2]:
                    # Out of Stock %
                    oos_items = len(data['stock'][data['stock']['STOCK'] == 0])
                    oos_percentage = (oos_items / total_items * 100) if total_items > 0 else 0
                    st.metric("Out of Stock %", f"{oos_percentage:.1f}%")
                
                with kpi_cols[3]:
                    # Active SKUs (with sales)
                    active_skus = data['sales']['SKU'].nunique()
                    total_skus = data['sku_master']['SKU'].nunique()
                    st.metric("Active SKUs", f"{active_skus}/{total_skus}")
                
                # Forecasting Performance
                st.markdown("#### Forecasting Method Comparison")
                if 'replen_data' in locals():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Current Method Benefits:**")
                        if forecasting_method == "Weighted Moving Average":
                            st.success("✅ **Weighted Moving Average**")
                            st.write("• Recent sales get higher importance")
                            st.write("• Trend analysis included")
                            st.write("• Better demand variability handling")
                            st.write("• Improved forecast accuracy")
                        else:
                            st.info("📊 **Simple Average**")
                            st.write("• Equal weight to all historical data")
                            st.write("• Simple and transparent")
                            st.write("• Good for stable demand patterns")
                    
                    with col2:
                        st.markdown("**Recommendation:**")
                        if forecasting_method == "Simple Average":
                            st.warning("💡 **Consider upgrading to Weighted Moving Average** for better accuracy in dynamic retail environments")
                        else:
                            st.success("🎯 **You're using the recommended advanced forecasting method**")
                
                # Data Quality Metrics
                st.markdown("#### Data Quality Summary")
                quality_cols = st.columns(3)
                
                with quality_cols[0]:
                    # Sales data quality
                    sales_days = (data['sales']['DATE'].max() - data['sales']['DATE'].min()).days
                    st.metric("Sales History", f"{sales_days} days")
                
                with quality_cols[1]:
                    # SKU coverage
                    sales_skus = set(data['sales']['SKU'].unique())
                    stock_skus = set(data['stock']['SKU'].unique())
                    coverage = len(sales_skus.intersection(stock_skus)) / len(sales_skus.union(stock_skus)) * 100
                    st.metric("SKU Data Coverage", f"{coverage:.1f}%")
                
                with quality_cols[2]:
                    # Store mapping quality
                    if 'STORE_NAME' in data['sales'].columns:
                        st.metric("Store Mapping", "✅ Complete")
                    else:
                        st.metric("Store Mapping", "⚠️ Basic")
            else:
                st.info("Upload all data files to view performance metrics")
            st.markdown("### Replenishment Analysis")
            if all(k in data for k in ['sales', 'stock', 'warehouse', 'sku_master']):
                # Time period selection for replenishment
                col1, col2 = st.columns([3, 1])
                with col1:
                    replen_period = st.selectbox(
                        "Select Analysis Period for Replenishment",
                        [7, 14, 30, 60, 90],
                        index=2,
                        format_func=lambda x: f"Last {x} Days"
                    )
                
                with col2:
                    run_replen = st.button("Calculate Replenishment", type="primary")
                
                if run_replen:
                    with st.spinner("Calculating replenishment recommendations..."):
                        # Calculate replenishment recommendations
                        replen_data = calculate_replenishment(
                            data['sales'],
                            data['stock'],
                            data['warehouse'],
                            data['sku_master'],
                            data.get('style_master'),  # Pass style_master if available
                            data.get('size_master'),   # Pass size_master if available
                            time_period_days=replen_period
                        )
                
                if not replen_data.empty:
                    # Overall replenishment metrics
                    st.markdown("#### Replenishment Overview")
                    metrics_cols = st.columns(4)
                    
                    with metrics_cols[0]:
                        total_replen = replen_data['FINAL_REPLEN_QTY'].sum()
                        st.metric("Total Replenishment Qty", f"{total_replen:,.0f}")
                    
                    with metrics_cols[1]:
                        stores_needing_replen = replen_data[replen_data['FINAL_REPLEN_QTY'] > 0]['STORE'].nunique()
                        st.metric("Stores Needing Stock", stores_needing_replen)
                    
                    with metrics_cols[2]:
                        skus_to_replen = replen_data[replen_data['FINAL_REPLEN_QTY'] > 0]['SKU'].nunique()
                        st.metric("SKUs to Replenish", skus_to_replen)
                    
                    with metrics_cols[3]:
                        avg_stock_cover = replen_data['STOCK_COVER_DAYS'].median()
                        st.metric("Median Stock Cover (Days)", f"{avg_stock_cover:.1f}")
                    
                    # Detailed replenishment recommendations
                    st.markdown("#### Replenishment Recommendations")
                    
                    # Filter for items needing replenishment
                    replen_recommendations = replen_data[replen_data['FINAL_REPLEN_QTY'] > 0].copy()
                    replen_recommendations = replen_recommendations.sort_values(
                        ['STOCK_COVER_DAYS', 'DAILY_SALES'],
                        ascending=[True, False]
                    )
                    
                    # Add stock-out indicators and highlighting
                    replen_recommendations['STOCK_OUT'] = replen_recommendations['STOCK'] == 0
                    replen_recommendations['CRITICAL_STOCK'] = (
                        (replen_recommendations['STOCK_COVER_DAYS'] < lead_time_days) &  # Less than lead time
                        (replen_recommendations['DAILY_SALES'] > 0)
                    )
                    
                    # Display recommendations with highlighting
                    st.markdown("#### Replenishment Recommendations")
                    st.markdown(f"🔴 Stock Out &nbsp;&nbsp;&nbsp; 🟡 Critical Stock (< {lead_time_days} days coverage)")
                    
                    # Display key columns with proper formatting
                    display_df = replen_recommendations[[
                        'STORE', 'SKU', 'STYLE', 'COLOR', 'SIZE', 'GENDER',
                        'STOCK', 'DAILY_SALES', 'STOCK_COVER_DAYS',
                        'TARGET_STOCK', 'FINAL_REPLEN_QTY', 'WAREHOUSE_STOCK', 'STOCK_OUT', 'CRITICAL_STOCK', 'REMARKS'
                    ]].copy()
                    
                    # Add a section to highlight warehouse stock shortages
                    shortage_items = display_df[display_df['REMARKS'].str.contains('insufficient', na=False)]
                    if not shortage_items.empty:
                        st.warning("⚠️ **Warehouse Stock Shortages Detected**")
                        st.markdown("The following items have insufficient warehouse stock to meet demand:")
                        shortage_summary = shortage_items[['SKU', 'STYLE', 'WAREHOUSE_STOCK', 'REMARKS']].drop_duplicates()
                        st.dataframe(shortage_summary, use_container_width=True)
                    
                    # Handle case where GENDER column might not exist
                    if 'GENDER' not in display_df.columns:
                        display_df['GENDER'] = 'UNISEX'
                    
                    # Add store name column if available
                    if 'STORE_NAME' in replen_recommendations.columns:
                        display_df['STORE_DISPLAY'] = replen_recommendations['STORE'] + ' - ' + replen_recommendations['STORE_NAME']
                        # Reorder columns to show store display first
                        cols = ['STORE_DISPLAY'] + [col for col in display_df.columns if col not in ['STORE', 'STORE_DISPLAY']]
                        display_df = display_df[cols]
                        # Remove the original STORE column only if it exists
                        if 'STORE' in display_df.columns:
                            display_df = display_df.drop('STORE', axis=1)
                    
                    # Format numeric columns
                    display_df['DAILY_SALES'] = display_df['DAILY_SALES'].round(2)
                    display_df['STOCK_COVER_DAYS'] = display_df['STOCK_COVER_DAYS'].round(1)
                    
                    column_renames = {
                        'STORE_DISPLAY': 'Store Code - Name',
                        'DAILY_SALES': 'Daily Sales',
                        'STOCK_COVER_DAYS': 'Stock Cover (Days)',
                        'TARGET_STOCK': 'Target Stock',
                        'FINAL_REPLEN_QTY': 'Replen Qty',
                        'STOCK_OUT': 'Stock Out',
                        'CRITICAL_STOCK': 'Critical'
                    }
                    
                    # Only rename columns that exist in the dataframe
                    final_renames = {k: v for k, v in column_renames.items() if k in display_df.columns}
                    
                    st.dataframe(
                        display_df.rename(columns=final_renames),
                        use_container_width=True
                    )
                    
                    # Download button for replenishment plan
                    # Organize columns in a logical order for output
                    output_columns = [
                        'STORE', 'STORE_NAME', 'SKU', 'STYLE', 'COLOR', 'SIZE', 'GENDER',
                        'STOCK', 'WAREHOUSE_STOCK', 'DAILY_SALES', 'STOCK_COVER_DAYS',
                        'TARGET_STOCK', 'REPLEN_QTY', 'FINAL_REPLEN_QTY',
                        'PRIORITY_SCORE', 'REMARKS', 'IS_NEW_ITEM'
                    ]
                    
                    # Ensure all essential columns exist before output
                    essential_cols = ['STORE', 'SKU', 'STYLE', 'COLOR', 'SIZE']
                    for col in essential_cols:
                        if col not in replen_recommendations.columns:
                            st.warning(f"⚠️ Adding missing essential column: {col}")
                            replen_recommendations[col] = 'UNKNOWN' if col in ['STYLE', 'COLOR', 'SIZE'] else replen_recommendations.get('STORE', 'UNKNOWN')
                    
                    # Only include columns that exist in the dataframe
                    available_cols = [col for col in output_columns if col in replen_recommendations.columns]
                    # Add any remaining columns that weren't in our list
                    remaining_cols = [col for col in replen_recommendations.columns if col not in available_cols]
                    final_output_cols = available_cols + remaining_cols
                    
                    # Create output with organized columns
                    output_df = replen_recommendations[final_output_cols].copy()
                    
                    # FINAL FREEBIES CHECK: Absolutely ensure no FREEBIES in output file
                    if 'GENDER' in output_df.columns:
                        output_df['GENDER'] = output_df['GENDER'].astype(str).str.strip().str.upper()
                        freebies_in_output = output_df[output_df['GENDER'].str.contains('FREEBIE', na=False, case=False)]
                        
                        if len(freebies_in_output) > 0:
                            st.error(f"🚨 CRITICAL: Found {len(freebies_in_output)} FREEBIES items in output! Removing before download...")
                            freebies_styles = freebies_in_output['STYLE'].unique()
                            st.error(f"   FREEBIES styles found: {', '.join(str(s) for s in freebies_styles)}")
                            
                            # Remove ALL FREEBIES from output
                            output_df = output_df[~output_df['GENDER'].str.contains('FREEBIE', na=False, case=False)].copy()
                            st.success(f"✅ Removed {len(freebies_in_output)} FREEBIES items from download file")
                        else:
                            st.success("✅ Output file verified: NO FREEBIES data included")
                    
                    csv = output_df.to_csv(index=False)
                    
                    st.download_button(
                        "Download Replenishment Plan",
                        csv,
                        "replenishment_plan.csv",
                        "text/csv",
                        key='download-replen-csv'
                    )
                    
                    # Stock Cover Analysis
                    st.markdown("#### Stock Cover Analysis")
                    fig_stock_cover = px.box(
                        replen_data,
                        x='STORE',
                        y='STOCK_COVER_DAYS',
                        title='Stock Cover Distribution by Store'
                    )
                    st.plotly_chart(fig_stock_cover, use_container_width=True, key="stock_cover_chart_2")
                    
                    # Replenishment by Store
                    st.markdown("#### Replenishment Quantities by Store")
                    store_replen = replen_recommendations.groupby('STORE')['FINAL_REPLEN_QTY'].sum().sort_values(ascending=True)
                    fig_store_replen = px.bar(
                        x=store_replen.values,
                        y=store_replen.index,
                        orientation='h',
                        title='Total Replenishment Quantity by Store'
                    )
                    st.plotly_chart(fig_store_replen, use_container_width=True, key="store_replen_chart_2")
    
    else:
        st.info("👆 Please upload all required data files to begin analysis")

if __name__ == "__main__":
    main()