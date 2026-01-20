"""
Data Merge Helper - Handles merging multiple data uploads
Version: 2.0 - Fixed _cached_get_quick_summary() calls
"""
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Optional, Tuple
import time
import os
import tempfile
import pickle
from pathlib import Path

# Import progress tracker if available
try:
    from progress_tracker import ProcessingProgressTracker, display_processing_progress
    PROGRESS_TRACKER_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKER_AVAILABLE = False

# Import database if available
try:
    from sales_database import SalesDatabase, _cached_get_quick_summary
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    _cached_get_quick_summary = None

# Create persistent storage directory
CACHE_DIR = Path(tempfile.gettempdir()) / "streamlit_sales_cache"
CACHE_DIR.mkdir(exist_ok=True)


def save_dataframe_persistent(df: pd.DataFrame, key: str) -> bool:
    """Save dataframe to persistent storage"""
    try:
        file_path = CACHE_DIR / f"{key}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False


def load_dataframe_persistent(key: str) -> Optional[pd.DataFrame]:
    """Load dataframe from persistent storage"""
    try:
        file_path = CACHE_DIR / f"{key}.pkl"
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Error loading data: {str(e)}")
    return None


def clear_persistent_data(key: str) -> bool:
    """Clear persistent data for a key"""
    try:
        file_path = CACHE_DIR / f"{key}.pkl"
        if file_path.exists():
            file_path.unlink()
        return True
    except Exception as e:
        st.error(f"Error clearing data: {str(e)}")
        return False

class DataMergeManager:
    """Manage multiple data uploads and merging"""
    
    @staticmethod
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        """Load CSV or Excel file with progress tracking"""
        try:
            file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
            
            with st.spinner(f"📂 Loading {uploaded_file.name}..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    return None
                
                # Show success with details
                st.success(f"✅ Loaded {uploaded_file.name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("File Size", f"{file_size:.2f} MB")
                
                return df
            
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            return None
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> dict:
        """Get summary statistics of dataframe"""
        if df is None or df.empty:
            return None
        
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "file_size": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        }
        
        # Add date range if date column exists
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                if df[date_col].dtype == 'object':
                    df[date_col] = pd.to_datetime(df[date_col])
                summary["min_date"] = df[date_col].min()
                summary["max_date"] = df[date_col].max()
            except:
                pass
        
        return summary
    
    @staticmethod
    def merge_datasets(old_df: Optional[pd.DataFrame], 
                       new_df: pd.DataFrame,
                       merge_type: str = "append") -> pd.DataFrame:
        """
        Merge old and new datasets with progress tracking
        
        Args:
            old_df: Previously uploaded data
            new_df: New data to add
            merge_type: 'append' (add all), 'update' (overwrite by date), or 'dedupe' (remove duplicates)
        """
        
        if old_df is None or old_df.empty:
            return new_df.copy()
        
        # Show progress
        if PROGRESS_TRACKER_AVAILABLE:
            total_rows = len(old_df) + len(new_df)
            tracker = ProcessingProgressTracker(
                total_rows=total_rows,
                total_bytes=total_rows * 1000,
                operation_name="Merging Data"
            )
            progress_col = st.empty()
            tracker.update(int(total_rows * 0.3))  # Start at 30%
            with progress_col.container():
                display_processing_progress(tracker, show_details=False)
            time.sleep(0.1)
        
        if merge_type == "append":
            # Simply concatenate all data
            merged = pd.concat([old_df, new_df], ignore_index=True)
            if PROGRESS_TRACKER_AVAILABLE:
                tracker.update(total_rows)
                with progress_col.container():
                    display_processing_progress(tracker, show_details=False)
                progress_col.empty()
            return merged
        
        elif merge_type == "update":
            # Update by date - newer data overwrites older
            date_cols = [col for col in old_df.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                try:
                    merged = pd.concat([old_df, new_df], ignore_index=True)
                    merged = merged.sort_values(by=date_col, ascending=False)
                    # Keep first occurrence (most recent)
                    merged = merged.drop_duplicates(subset=[col for col in merged.columns if col != date_col], keep='first')
                    if PROGRESS_TRACKER_AVAILABLE:
                        tracker.update(total_rows)
                        with progress_col.container():
                            display_processing_progress(tracker, show_details=False)
                        progress_col.empty()
                    return merged.sort_values(by=date_col)
                except:
                    pass
            
            # Fallback to append if date column not found
            return pd.concat([old_df, new_df], ignore_index=True)
        
        elif merge_type == "dedupe":
            # Combine and remove exact duplicates
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(keep='first')
            return merged
        
        return pd.concat([old_df, new_df], ignore_index=True)
    
    @staticmethod
    def display_comparison(old_df: Optional[pd.DataFrame], 
                          new_df: pd.DataFrame,
                          title: str = "Data Comparison"):
        """Display side-by-side comparison of old and new data"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Previous Data")
            if old_df is not None and not old_df.empty:
                old_summary = DataMergeManager.get_data_summary(old_df)
                st.metric("Rows", f"{old_summary['rows']:,}")
                st.metric("Columns", old_summary['columns'])
                if "min_date" in old_summary and "max_date" in old_summary:
                    st.text(f"Date Range: {old_summary['min_date'].date()} to {old_summary['max_date'].date()}")
                with st.expander("Preview"):
                    st.dataframe(old_df.head(10), use_container_width=True)
            else:
                st.info("No previous data")
        
        with col2:
            st.subheader("📁 New Data")
            new_summary = DataMergeManager.get_data_summary(new_df)
            st.metric("Rows", f"{new_summary['rows']:,}")
            st.metric("Columns", new_summary['columns'])
            if "min_date" in new_summary and "max_date" in new_summary:
                st.text(f"Date Range: {new_summary['min_date'].date()} to {new_summary['max_date'].date()}")
            with st.expander("Preview"):
                st.dataframe(new_df.head(10), use_container_width=True)
        
        # Merge summary
        if old_df is not None and not old_df.empty:
            merged = DataMergeManager.merge_datasets(old_df, new_df, "append")
            st.divider()
            st.subheader("✅ Merged Result (Preview)")
            st.metric("Total Rows", f"{len(merged):,}")
            st.metric("Total Rows Added", f"{len(new_df):,}")
            if len(merged) > 0:
                st.success(f"✓ Successfully prepared {len(new_df):,} new records for merge!")


def streamlit_multi_upload_ui(data_type: str = "Sales") -> Tuple[Optional[pd.DataFrame], str]:
    """
    Streamlit UI for multi-file uploads with clear, step-by-step flow
    
    Returns:
        Tuple of (merged_dataframe, merge_type)
    """
    
    st.subheader(f"📂 {data_type} Data Management")
    
    # Step 1: Show current database status (from database, not session)
    if DATABASE_AVAILABLE and _cached_get_quick_summary:
        summary = _cached_get_quick_summary()
        
        if summary.get('total_rows', 0) > 0:
            st.success(f"✅ **Database has {summary['total_rows']:,} rows** ({summary['min_date']} to {summary['max_date']})")
        else:
            st.info("📦 Database is empty - upload data to get started")
        st.divider()
    
    # Step 2: Upload new files
    st.markdown("### Step 1️⃣: Upload Files")
    st.caption("Select one or more files to upload and merge with existing data")
    
    uploaded_files = st.file_uploader(
        f"Choose {data_type} files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key=f"uploader_{data_type}"
    )
    
    if not uploaded_files:
        st.info("👆 Select files to continue")
        return None, "append"
    
    # Step 3: Load files
    st.markdown("### Step 2️⃣: Load & Combine Files")
    
    new_data_list = []
    total_new_rows = 0
    
    with st.spinner("📂 Loading files..."):
        for uploaded_file in uploaded_files:
            df = DataMergeManager.load_file(uploaded_file)
            if df is not None:
                new_data_list.append(df)
                total_new_rows += len(df)
    
    if not new_data_list:
        st.error("❌ Could not load any files")
        return None, "append"
    
    # Combine multiple files
    if len(new_data_list) > 1:
        st.info(f"✅ Loaded {len(new_data_list)} files with {total_new_rows:,} total rows")
        new_combined = pd.concat(new_data_list, ignore_index=True)
    else:
        st.success(f"✅ Loaded {uploaded_files[0].name}: {len(new_data_list[0]):,} rows")
        new_combined = new_data_list[0]
    
    st.divider()
    
    # Step 4: Choose merge option
    st.markdown("### Step 3️⃣: Merge Strategy")
    st.caption("How should new data be combined with existing data?")
    
    merge_type = st.radio(
        "Select merge option:",
        options=["append", "update", "dedupe"],
        format_func=lambda x: {
            "append": "📎 **Append** - Add all new rows to existing data",
            "update": "🔄 **Update** - Newer data overwrites older matching records",
            "dedupe": "🔀 **Deduplicate** - Remove exact duplicate rows"
        }.get(x, x),
        horizontal=False
    )
    
    st.divider()
    
    # Step 5: Preview data
    st.markdown("### Step 4️⃣: Data Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 New Data (to upload)")
        st.metric("Rows", f"{len(new_combined):,}")
        st.metric("Columns", len(new_combined.columns))
        with st.expander("View sample"):
            st.dataframe(new_combined.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📊 What will happen")
        if DATABASE_AVAILABLE:
            summary = _cached_get_quick_summary()
            current_rows = summary.get('total_rows', 0)
        else:
            current_rows = 0
        
        if current_rows > 0:
            if merge_type == "append":
                final_rows = current_rows + len(new_combined)
                st.metric("Current rows", f"{current_rows:,}")
                st.metric("New rows", f"+ {len(new_combined):,}")
                st.metric("Result", f"= {final_rows:,}")
            elif merge_type == "dedupe":
                st.metric("Current rows", f"{current_rows:,}")
                st.metric("New rows", f"{len(new_combined):,}")
                st.info("Duplicates will be removed")
            else:  # update
                st.metric("Current rows", f"{current_rows:,}")
                st.metric("Updated rows", f"~{len(new_combined):,}")
                st.info("Newer data will overwrite older")
        else:
            st.metric("Result", f"{len(new_combined):,} rows")
            st.caption("(First upload)")
    
    st.divider()
    
    # Step 6: Save button
    st.markdown("### Step 5️⃣: Save to Database")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Save & Store", key=f"save_btn_{data_type}", use_container_width=True):
            with st.spinner("💾 Saving to database..."):
                if DATABASE_AVAILABLE:
                    db = SalesDatabase()
                    if db.save_dataframe(new_combined, f"{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                        st.success("✅ **SUCCESS!** Data saved to database")
                        st.balloons()
                    else:
                        st.error("❌ Failed to save data")
                else:
                    st.error("Database not available")
    
    with col2:
        csv = new_combined.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{data_type}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        if st.button("🔄 Clear", key=f"clear_btn_{data_type}", use_container_width=True):
            st.rerun()
    
    return new_combined, merge_type


def save_merged_data_to_csv(df: pd.DataFrame, filename: str = "merged_data.csv") -> bytes:
    """Convert dataframe to CSV bytes for download"""
    return df.to_csv(index=False).encode()
