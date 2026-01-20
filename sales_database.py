"""
Sales Database Module - SQLite backend for persistent data storage
"""
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Optional, Dict, List

# Database setup
DB_DIR = Path(tempfile.gettempdir()) / "streamlit_app_db"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "sales_data.db"


class SalesDatabase:
    """SQLite database manager for sales data"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sales_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sales_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    store TEXT,
                    sku TEXT,
                    quantity REAL,
                    amount REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create upload_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS upload_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    rows_count INTEGER,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def save_dataframe(self, df: pd.DataFrame, source_name: str = "upload") -> bool:
        """Save dataframe to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql('sales_data', conn, if_exists='append', index=False)
            
            # Record in upload history
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO upload_history (filename, rows_count, status) VALUES (?, ?, ?)',
                (source_name, len(df), 'completed')
            )
            conn.commit()
            conn.close()
            
            # Clear cache
            st.cache_data.clear()
            return True
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")
            return False
    
    def load_all_data(self) -> Optional[pd.DataFrame]:
        """Load all data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('SELECT * FROM sales_data ORDER BY date', conn)
            conn.close()
            return df if len(df) > 0 else None
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def get_summary(self) -> Dict:
        """Get quick summary of database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total_rows FROM sales_data')
            total = cursor.fetchone()[0]
            
            cursor.execute('SELECT MIN(date) as min_date, MAX(date) as max_date FROM sales_data')
            min_date, max_date = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_rows': total,
                'min_date': min_date or 'N/A',
                'max_date': max_date or 'N/A'
            }
        except Exception as e:
            return {'total_rows': 0, 'min_date': 'N/A', 'max_date': 'N/A'}


# Cached function for quick summary (module level - ONLY definition)
@st.cache_data(ttl=3600)
def _cached_get_quick_summary() -> Dict:
    """Get cached database summary - single source of truth"""
    try:
        db = SalesDatabase()
        return db.get_summary()
    except Exception as e:
        return {'total_rows': 0, 'min_date': 'N/A', 'max_date': 'N/A', 'error': str(e)}


# Streamlit UI functions
def streamlit_database_status():
    """Display database status in Streamlit"""
    summary = _cached_get_quick_summary()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{summary['total_rows']:,}")
    with col2:
        st.metric("Min Date", summary['min_date'])
    with col3:
        st.metric("Max Date", summary['max_date'])


def streamlit_data_management():
    """Data management UI section"""
    st.subheader("📊 Data Management")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    if st.button("📥 Download Data"):
        db = SalesDatabase()
        df = db.load_all_data()
        if df is not None:
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "sales_data.csv", "text/csv")
