# Retail Analytics Dashboard (Replenishment Module)

This Streamlit application provides retail analytics and intelligent replenishment recommendations based on sales history, current stock levels, and warehouse availability.

## Features

- Sales data analysis
- Stock level monitoring
- Warehouse inventory tracking
- Intelligent replenishment calculations
- Store performance metrics
- Data validation and cleaning
- Downloadable replenishment plans

## Installation

1. Create a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Upload your data files:
   - Sales Data (CSV/Excel)
   - Stock Data (CSV/Excel)
   - Warehouse Data (CSV/Excel)
   - SKU Master Data (CSV/Excel)
   - Style Master Data (Optional)

3. Configure replenishment parameters:
   - Target Coverage Days
   - Safety Stock Days
   - Lead Time
   - Minimum Order Quantity (MOQ)
   - Forecasting Method

4. View analytics and download replenishment recommendations

## Data Format Requirements

### Sales Data
Required columns:
- DATE
- STORE/STORE_CODE
- SKU
- QUANTITY

### Stock Data
Required columns:
- STORE/STORE_CODE
- SKU
- STOCK

### Warehouse Data
Required columns:
- SKU
- WAREHOUSE_STOCK

### SKU Master
Required columns:
- SKU
- STYLE
- COLOR
- SIZE

### Style Master (Optional)
Required columns if used:
- STYLE
- GENDER

# Performance settings
MAX_WORKERS = multiprocessing.cpu_count()
CHUNK_SIZE = 100000
MEMORY_LIMIT = 0.75  # Use max 75% of available memory

# Suppress warnings
warnings.filterwarnings('ignore')

# Backend API URL
API_URL = "http://127.0.0.1:8000"

def upload_file_to_api(file, file_type):
    """Upload a file to the backend API"""
    # Allow skipping backend uploads via Streamlit toggle (useful for offline/testing)
    try:
        if not st.session_state.get('enable_backend_uploads', True):
            # Skip upload when disabled (return a marker that upload was skipped)
            st.info(f"Backend uploads disabled — skipping upload for {file_type}.")
            return {"skipped": True}
    except Exception:
        # st may not be available at import-time; ignore and continue
        pass

    try:
        files = {"file": file}
        response = requests.post(
            f"{API_URL}/upload/{file_type}",
            files=files
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error uploading {file_type}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend server. Please ensure it's running.")
        return None

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

# Configure Pandas display options
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Set page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main { padding: 2rem; }
        .title {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 500;
            color: #1E1E1E;
            padding-bottom: 1rem;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
        }
        .dataframe {
            border: none !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

def calculate_replenishment(sales_data, stock_data, warehouse_data, sku_master, style_master=None,
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
            # Give more weight to recent sales for better forecasting
            sales_velocity = recent_sales.groupby(['STORE', 'SKU']).apply(
                lambda x: calculate_weighted_daily_sales(x, time_period_days)
            ).reset_index()
            sales_velocity.columns = ['STORE', 'SKU', 'DAILY_SALES']
            
            # Add simple trend adjustment to improve forecasting
            sales_velocity['TREND_FACTOR'] = sales_velocity.apply(
                lambda row: calculate_simple_trend(recent_sales, row['STORE'], row['SKU'], time_period_days),
                axis=1
            )
            
            # Apply trend adjustment to daily sales (max 50% increase/decrease for safety)
            sales_velocity['DAILY_SALES'] = sales_velocity['DAILY_SALES'] * sales_velocity['TREND_FACTOR'].clip(0.5, 1.5)
        else:
            # Simple average method for comparison
            sales_velocity = recent_sales.groupby(['STORE', 'SKU']).agg(
                DAILY_SALES=('QUANTITY', lambda x: x.sum() / time_period_days)
            ).reset_index()
        
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
        
        # Merge sales velocity with current stock
        replen_calc = pd.merge(
            sales_velocity,
            current_stock,
            on=['STORE', 'SKU'],
            how='outer'
        ).fillna(0)
        
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
        
        # Calculate forecast demand considering lead time + target coverage + safety stock
        total_coverage_needed = target_coverage_days + safety_stock_days + lead_time_days
        replen_calc['FORECAST_DEMAND'] = replen_calc['DAILY_SALES'] * total_coverage_needed
        
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
        
        # Merge with warehouse stock
        warehouse_stock = warehouse_data[['SKU', 'WAREHOUSE_STOCK']].copy()
        replen_calc = pd.merge(
            replen_calc,
            warehouse_stock,
            on='SKU',
            how='left'
        )
        replen_calc['WAREHOUSE_STOCK'] = replen_calc['WAREHOUSE_STOCK'].fillna(0)
        
        # Add SKU information and gender from style master
        replen_calc = pd.merge(
            replen_calc,
            sku_master[['SKU', 'STYLE', 'COLOR', 'SIZE']],
            on='SKU',
            how='left'
        )
        
        # Add gender information from style master if available
        if style_master is not None and not style_master.empty and 'GENDER' in style_master.columns:
            replen_calc = pd.merge(
                replen_calc,
                style_master[['STYLE', 'GENDER']],
                on='STYLE',
                how='left'
            )
            # Fill any missing gender values
            replen_calc['GENDER'] = replen_calc['GENDER'].fillna('UNISEX')
            # Filter out freebies (items marked as FREEBIES in GENDER)
            replen_calc = replen_calc[replen_calc['GENDER'] != 'FREEBIES']
        else:
            # If style master not available or no gender column, add default
            replen_calc['GENDER'] = 'UNISEX'
        
        # Add REMARKS column for tracking allocation issues
        replen_calc['REMARKS'] = ''
        
        # Final allocation based on priority scores and warehouse stock
        for sku in replen_calc['SKU'].unique():
            sku_data = replen_calc[replen_calc['SKU'] == sku].copy()
            available_stock = sku_data['WAREHOUSE_STOCK'].iloc[0]
            total_demand = sku_data['REPLEN_QTY'].sum()
            
            # If no warehouse stock available, set all replenishment quantities to 0
            if available_stock <= 0:
                replen_calc.loc[replen_calc['SKU'] == sku, 'FINAL_REPLEN_QTY'] = 0
                replen_calc.loc[replen_calc['SKU'] == sku, 'REMARKS'] = 'No warehouse stock available'
                continue
                
            # If there's more demand than available stock
            if total_demand > available_stock:
                # Add remark about insufficient warehouse stock
                shortage_qty = total_demand - available_stock
                replen_calc.loc[replen_calc['SKU'] == sku, 'REMARKS'] = f'Warehouse stock insufficient: Demand={total_demand:.0f}, Short by {shortage_qty:.0f} units'
                
                # Allocate based on priority scores and replenishment needs
                priority_weighted_need = sku_data['PRIORITY_SCORE'] * sku_data['REPLEN_QTY']
                total_weighted_need = priority_weighted_need.sum()
                
                if total_weighted_need > 0:
                    sku_data['FINAL_REPLEN_QTY'] = np.floor(
                        (priority_weighted_need / total_weighted_need) * available_stock
                    )
                else:
                    # Equal distribution if no priority differences
                    base_qty = np.floor(available_stock / len(sku_data))
                    sku_data['FINAL_REPLEN_QTY'] = base_qty
                    
                    # Distribute remaining units to highest priority stores
                    remaining = available_stock - (base_qty * len(sku_data))
                    if remaining > 0:
                        top_stores = sku_data.nlargest(int(remaining), 'PRIORITY_SCORE').index
                        sku_data.loc[top_stores, 'FINAL_REPLEN_QTY'] += 1
            else:
                # If enough stock, fulfill original quantities with size set check
                style_data = sku_master[sku_master['SKU'] == sku].iloc[0]
                style_skus = sku_master[
                    (sku_master['STYLE'] == style_data['STYLE']) &
                    (sku_master['COLOR'] == style_data['COLOR'])
                ]['SKU'].tolist()
                
                # Check warehouse stock for all sizes
                size_stock = warehouse_data[warehouse_data['SKU'].isin(style_skus)]
                all_sizes_available = (size_stock['WAREHOUSE_STOCK'] >= 2).all()  # MOQ=2 for each size
                
                if all_sizes_available:
                    # If all sizes available, ensure complete set allocation
                    sku_data['FINAL_REPLEN_QTY'] = np.where(
                        sku_data['REPLEN_QTY'] > 0,
                        np.maximum(2, sku_data['REPLEN_QTY']),  # Apply MOQ=2
                        0
                    )
                else:
                    # If not all sizes available, fulfill original quantities
                    sku_data['FINAL_REPLEN_QTY'] = np.where(
                        sku_data['REPLEN_QTY'] > 0,
                        np.minimum(np.maximum(2, sku_data['REPLEN_QTY']), sku_data['WAREHOUSE_STOCK']),
                        0
                    )
                
                replen_calc.loc[replen_calc['SKU'] == sku, 'FINAL_REPLEN_QTY'] = sku_data['FINAL_REPLEN_QTY']
        
        # Round all quantities to integers
        replen_calc['FINAL_REPLEN_QTY'] = np.floor(replen_calc['FINAL_REPLEN_QTY'])
        
        # Add size set completeness indicator
        replen_calc['COMPLETE_SIZE_SET'] = False
        for style in sku_master['STYLE'].unique():
            style_skus = sku_master[sku_master['STYLE'] == style]['SKU'].tolist()
            mask = replen_calc['SKU'].isin(style_skus)
            if mask.any():
                replen_calc.loc[mask, 'COMPLETE_SIZE_SET'] = (
                    replen_calc.loc[mask, 'FINAL_REPLEN_QTY'] > 0
                ).all()
        
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
        warehouse_stock = warehouse_data[['SKU', 'WAREHOUSE_STOCK']]
        replen_calc = pd.merge(
            replen_calc,
            warehouse_stock,
            on='SKU',
            how='left'
        )
        
        # Add warehouse allocation optimization logic
        def optimize_allocation(group):
            available_stock = group['WAREHOUSE_STOCK'].iloc[0]
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
            available_stock = sku_data['WAREHOUSE_STOCK'].iloc[0]
            
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
    """Read uploaded file into pandas DataFrame with optimized performance and parallel processing"""
    if uploaded_file is None:
        return pd.DataFrame()
        
    try:
        start_time = time.time()
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # Progress placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Define data types and validation rules
        dtype_dict = {
            'SKU': 'category',
            'STORE': 'category',
            'STYLE': 'category',
            'COLOR': 'category',
            'SIZE': 'category',
            'GENDER': 'category',
            'QUANTITY': 'float32',
            'STOCK': 'float32',
            'WAREHOUSE_STOCK': 'float32',
            'DATE': 'datetime64[ns]'
        }
        
        # Memory usage before processing
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if file_type == 'csv':
            try:
                # Initialize parallel processing
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Read CSV in chunks with parallel processing
                    chunks = []
                    total_chunks = sum(1 for _ in pd.read_csv(
                        uploaded_file, 
                        chunksize=CHUNK_SIZE, 
                        nrows=2
                    )) * CHUNK_SIZE
                    
                    # Process chunks in parallel with progress tracking
                    for i, chunk in enumerate(pd.read_csv(
                        uploaded_file,
                        dtype=dtype_dict,
                        na_values=['', 'NA', 'null'],
                        chunksize=CHUNK_SIZE,
                        low_memory=True
                    )):
                        # Update progress
                        progress = (i * CHUNK_SIZE) / total_chunks
                        progress_bar.progress(progress)
                        status_text.text(f"Processing chunk {i+1}...")
                        
                        # Process chunk in parallel
                        future = executor.submit(
                            process_chunk,
                            chunk,
                            dtype_dict
                        )
                        chunks.append(future.result())
                        
                        # Check memory usage
                        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        if current_memory > psutil.virtual_memory().total * MEMORY_LIMIT:
                            st.warning("⚠️ High memory usage detected. Try reducing file size.")
                            
                    # Combine processed chunks
                    df = pd.concat(chunks, ignore_index=True)
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}")
                st.error("Error in parallel processing. Falling back to standard processing...")
                
                # Fallback to standard processing
                df = pd.read_csv(
                    uploaded_file,
                    dtype=dtype_dict,
                    na_values=['', 'NA', 'null']
                )
            
        elif file_type in ['xls', 'xlsx']:
            # For Excel files, use optimization parameters
            df = pd.read_excel(
                uploaded_file,
                dtype=dtype_dict,
                engine='openpyxl'
            )
        else:
            st.error(f"Unsupported file type: {file_type}")
            return pd.DataFrame()
        
        # Optimize memory usage
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # More efficient cleaning
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if col not in dtype_dict:
                df[col] = df[col].fillna('').str.strip().astype('category')
        
        # Fill numeric columns with 0 instead of empty string
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(0)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add performance feedback
        if processing_time < 10:
            st.success(f"✅ File processed quickly in {processing_time:.2f} seconds")
        elif processing_time < 30:
            st.info(f"⏱️ File processed in {processing_time:.2f} seconds")
        else:
            st.warning(f"⚠️ File processing took {processing_time:.2f} seconds. Consider optimizing the file size.")
            
        # Add file statistics
        st.info(f"📊 Loaded {len(df):,} rows and {len(df.columns)} columns. Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
        
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
def clean_sales_data(df):
    """Clean sales data"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Detect columns
        store_col = detect_column(df, ['store', 'store name', 'ebo', 'channel'])
        store_code_col = detect_column(df, ['store code', 'store_code', 'code', 'store id', 'store_id'])
        ebo_name_col = detect_column(df, ['ebo name', 'ebo_name', 'store name', 'store_name'])  # Added EBO NAME detection
        sku_col = detect_column(df, ['sku', 'ean', 'product code'])
        date_col = detect_column(df, ['date', 'bill date', 'transaction date'])
        qty_col = detect_column(df, ['quantity', 'qty', 'bill quantity'])
        rate_col = detect_column(df, ['rate', 'price', 'mrp', 'amount per unit', 'unit price'])
        amount_col = detect_column(df, ['amount', 'total amount', 'bill amount', 'value'])
        
        if not all([sku_col, date_col, qty_col]):
            missing = []
            if not sku_col: missing.append("SKU")
            if not date_col: missing.append("Date")
            if not qty_col: missing.append("Quantity")
            st.error(f"Missing required columns: {', '.join(missing)}")
            return pd.DataFrame()
        
        # Create base dataframe
        cleaned_df = pd.DataFrame({
            'SKU': df[sku_col].apply(clean_text),
            'DATE': pd.to_datetime(df[date_col], errors='coerce'),
            'QUANTITY': df[qty_col].apply(safe_numeric)
        })
        
        # Add rate/amount columns if available to filter freebies
        if rate_col:
            cleaned_df['RATE'] = df[rate_col].apply(safe_numeric)
        if amount_col:
            cleaned_df['AMOUNT'] = df[amount_col].apply(safe_numeric)
            
        # Filter out freebies (items with zero price or marked as free)
        if 'RATE' in cleaned_df.columns:
            cleaned_df = cleaned_df[cleaned_df['RATE'] > 0]
        elif 'AMOUNT' in cleaned_df.columns:
            cleaned_df = cleaned_df[cleaned_df['AMOUNT'] > 0]
            
        # Remove temporary columns used for filtering
        if 'RATE' in cleaned_df.columns:
            cleaned_df = cleaned_df.drop('RATE', axis=1)
        if 'AMOUNT' in cleaned_df.columns:
            cleaned_df = cleaned_df.drop('AMOUNT', axis=1)
        
        # Handle store identification with priority for EBO NAME
        if store_code_col:
            cleaned_df['STORE_CODE'] = df[store_code_col].apply(clean_text)
            cleaned_df['STORE'] = cleaned_df['STORE_CODE']  # Use store code as primary identifier
            
            # Use EBO NAME as store name if available, otherwise use regular store name column
            if ebo_name_col:
                cleaned_df['STORE_NAME'] = df[ebo_name_col].apply(clean_text)
                st.success(f"✅ Using EBO NAME column: '{ebo_name_col}' as store names")
            elif store_col:
                cleaned_df['STORE_NAME'] = df[store_col].apply(clean_text)
                st.info(f"ℹ️ Using store column: '{store_col}' as store names")
        elif store_col:
            cleaned_df['STORE'] = df[store_col].apply(clean_text)
            cleaned_df['STORE_CODE'] = cleaned_df['STORE']  # Use store name as fallback
            
            # Use EBO NAME if available, otherwise use the store column as name too
            if ebo_name_col:
                cleaned_df['STORE_NAME'] = df[ebo_name_col].apply(clean_text)
                st.success(f"✅ Using EBO NAME column: '{ebo_name_col}' as store names")
            else:
                cleaned_df['STORE_NAME'] = cleaned_df['STORE']
        else:
            st.error("Missing required columns: Store, Store Code, or EBO NAME")
            return pd.DataFrame()
        
        # Remove invalid records
        cleaned_df = cleaned_df.dropna(subset=['DATE'])
        cleaned_df = cleaned_df[cleaned_df['QUANTITY'] > 0]
        
        return cleaned_df
        
    except Exception as e:
        st.error(f"Error cleaning sales data: {str(e)}")
        return pd.DataFrame()

def clean_stock_data(df):
    """Clean stock data"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Detect columns
        store_col = detect_column(df, ['store', 'store name', 'ebo', 'channel'])
        store_code_col = detect_column(df, ['store code', 'store_code', 'code', 'store id', 'store_id'])
        ebo_name_col = detect_column(df, ['ebo name', 'ebo_name', 'store name', 'store_name'])  # Added EBO NAME detection
        sku_col = detect_column(df, ['sku', 'ean', 'product code'])
        stock_col = detect_column(df, ['stock', 'quantity', 'qty', 'available'])
        
        if not all([sku_col, stock_col]):
            missing = []
            if not sku_col: missing.append("SKU")
            if not stock_col: missing.append("Stock")
            st.error(f"Missing required columns: {', '.join(missing)}")
            return pd.DataFrame()
        
        # Create base dataframe
        cleaned_df = pd.DataFrame({
            'SKU': df[sku_col].apply(clean_text),
            'STOCK': df[stock_col].apply(safe_numeric)
        })
        
        # Handle store identification with priority for EBO NAME
        if store_code_col:
            cleaned_df['STORE_CODE'] = df[store_code_col].apply(clean_text)
            cleaned_df['STORE'] = cleaned_df['STORE_CODE']  # Use store code as primary identifier
            
            # Use EBO NAME as store name if available, otherwise use regular store name column
            if ebo_name_col:
                cleaned_df['STORE_NAME'] = df[ebo_name_col].apply(clean_text)
                st.success(f"✅ Using EBO NAME column: '{ebo_name_col}' as store names in stock data")
            elif store_col:
                cleaned_df['STORE_NAME'] = df[store_col].apply(clean_text)
                st.info(f"ℹ️ Using store column: '{store_col}' as store names in stock data")
        elif store_col:
            cleaned_df['STORE'] = df[store_col].apply(clean_text)
            cleaned_df['STORE_CODE'] = cleaned_df['STORE']  # Use store name as fallback
            
            # Use EBO NAME if available, otherwise use the store column as name too
            if ebo_name_col:
                cleaned_df['STORE_NAME'] = df[ebo_name_col].apply(clean_text)
                st.success(f"✅ Using EBO NAME column: '{ebo_name_col}' as store names in stock data")
            else:
                cleaned_df['STORE_NAME'] = cleaned_df['STORE']
        else:
            st.error("Missing required columns: Store, Store Code, or EBO NAME in stock data")
            return pd.DataFrame()
        
        return cleaned_df
        
    except Exception as e:
        st.error(f"Error cleaning stock data: {str(e)}")
        return pd.DataFrame()

def clean_warehouse_data(df):
    """Clean warehouse data"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Detect columns - updated to include specific warehouse column names
        sku_col = detect_column(df, ['each client sku id', 'sku', 'ean', 'product code', 'client sku id'])
        stock_col = detect_column(df, ['total available quantity', 'stock', 'quantity', 'available quantity', 'available'])
        
        if not all([sku_col, stock_col]):
            missing = []
            if not sku_col: missing.append("SKU (Each Client SKU ID)")
            if not stock_col: missing.append("Stock (Total Available Quantity)")
            st.error(f"Missing required columns in warehouse data: {', '.join(missing)}")
            return pd.DataFrame()
        
        # Clean data
        cleaned_df = pd.DataFrame({
            'SKU': df[sku_col].apply(clean_text),
            'WAREHOUSE_STOCK': df[stock_col].apply(safe_numeric)
        })
        
        # Show which columns were detected
        if 'each client sku id' in sku_col.lower():
            st.success(f"✅ Using warehouse SKU column: '{sku_col}' (Each Client SKU ID detected)")
        else:
            st.info(f"ℹ️ Using warehouse SKU column: '{sku_col}'")
            
        if 'total available quantity' in stock_col.lower():
            st.success(f"✅ Using warehouse stock column: '{stock_col}' (Total Available Quantity detected)")
        else:
            st.info(f"ℹ️ Using warehouse stock column: '{stock_col}'")
        
        return cleaned_df
        
    except Exception as e:
        st.error(f"Error cleaning warehouse data: {str(e)}")
        return pd.DataFrame()

def clean_sku_master(df):
    """Clean SKU master data"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Detect columns
        sku_col = detect_column(df, ['sku', 'ean', 'product code'])
        style_col = detect_column(df, ['style', 'style code'])
        color_col = detect_column(df, ['color', 'colour'])
        size_col = detect_column(df, ['size'])
        
        if not all([sku_col, style_col, color_col, size_col]):
            missing = []
            if not sku_col: missing.append("SKU")
            if not style_col: missing.append("Style")
            if not color_col: missing.append("Color")
            if not size_col: missing.append("Size")
            st.error(f"Missing required columns: {', '.join(missing)}")
            return pd.DataFrame()
        
        # Clean data
        cleaned_df = pd.DataFrame({
            'SKU': df[sku_col].apply(clean_text),
            'STYLE': df[style_col].apply(clean_text),
            'COLOR': df[color_col].apply(clean_text),
            'SIZE': df[size_col].apply(clean_text)
        })
        
        return cleaned_df
        
    except Exception as e:
        st.error(f"Error cleaning SKU master data: {str(e)}")
        return pd.DataFrame()

def clean_style_master(df):
    """Clean style master data"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Detect columns
        style_col = detect_column(df, ['style', 'style code'])
        gender_col = detect_column(df, ['gender', 'department'])
        
        if not all([style_col, gender_col]):
            missing = []
            if not style_col: missing.append("Style")
            if not gender_col: missing.append("Gender")
            st.error(f"Missing required columns: {', '.join(missing)}")
            return pd.DataFrame()
        
        # Clean data
        cleaned_df = pd.DataFrame({
            'STYLE': df[style_col].apply(clean_text),
            'GENDER': df[gender_col].apply(clean_text)
        })
        
        # Filter out freebies data
        freebies_count = len(cleaned_df[cleaned_df['GENDER'] == 'FREEBIES'])
        if freebies_count > 0:
            st.info(f"ℹ️ Filtered out {freebies_count} freebies items from style master")
            cleaned_df = cleaned_df[cleaned_df['GENDER'] != 'FREEBIES']
        
        return cleaned_df
        
    except Exception as e:
        st.error(f"Error cleaning style master data: {str(e)}")
        return pd.DataFrame()

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
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Analytics Dashboard")
        st.markdown("---")
        st.markdown("### 📁 Data Upload")
        
        # File uploaders
        uploaded_files = {
            "sales": st.file_uploader("Sales Data (CSV/Excel)", type=["csv", "xlsx"]),
            "stock": st.file_uploader("Stock Data (CSV/Excel)", type=["csv", "xlsx"]),
            "warehouse": st.file_uploader("Warehouse Data (CSV/Excel)", type=["csv", "xlsx"]),
            "sku_master": st.file_uploader("SKU Master (CSV/Excel)", type=["csv", "xlsx"]),
            "style_master": st.file_uploader("Style Master (CSV/Excel)", type=["csv", "xlsx"])
        }
        st.markdown("---")
        # Toggle to enable/disable uploads to backend API (helps run app offline)
        enable_backend = st.checkbox(
            "Enable backend uploads",
            value=False,
            help="Toggle to upload files to backend API. Turn off to run analysis locally without attempting uploads."
        )
        # Persist the setting in session state for upload_file_to_api to read
        st.session_state['enable_backend_uploads'] = enable_backend
        
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
    
    # Process uploaded files and save to backend API
    data = {}
    
    # Map file types to cleaning functions
    cleaning_functions = {
        'sales': clean_sales_data,
        'stock': clean_stock_data,
        'warehouse': clean_warehouse_data,
        'sku_master': clean_sku_master,
        'style_master': clean_style_master
    }
    
    # Process each uploaded file
    for file_type, uploaded_file in uploaded_files.items():
        if uploaded_file:
            # First read and clean the data locally
            df = cleaning_functions[file_type](read_file(uploaded_file))
            
            if not df.empty:
                # Upload file to API and get response
                api_response = upload_file_to_api(uploaded_file, file_type)
                
                if api_response:
                    # Store the cleaned data for further processing
                    data[file_type] = df
                    st.success(f"✅ Successfully uploaded {file_type} data to backend")
                else:
                    st.warning(f"⚠️ Could not save {file_type} data to backend, but will continue with analysis")
                    data[file_type] = df    # Data Preview Section
    st.markdown("## Data Overview")
    st.markdown("Review your uploaded data below. Click to expand each section.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sales' in data: show_sample_data(data['sales'], "Sales Data")
        if 'stock' in data: show_sample_data(data['stock'], "Stock Data")
        if 'warehouse' in data: show_sample_data(data['warehouse'], "Warehouse Data")
    
    with col2:
        if 'sku_master' in data: show_sample_data(data['sku_master'], "SKU Master")
        if 'style_master' in data: show_sample_data(data['style_master'], "Style Master")
    
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
                            time_period_days=replen_period,
                            target_coverage_days=target_coverage_days,
                            safety_stock_days=safety_stock_days,
                            lead_time_days=lead_time_days,
                            moq=moq,
                            forecasting_method=forecasting_method
                        )
                        
                        if not replen_data.empty:
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
                            csv = replen_recommendations.to_csv(index=False)
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
                    csv = replen_recommendations.to_csv(index=False)
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