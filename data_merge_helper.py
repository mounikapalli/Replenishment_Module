"""
Data Merge Helper - SIMPLIFIED VERSION (No external dependencies)
Version: 3.1 - FORCE REDEPLOY - Completely self-contained, no dependency issues
"""
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Optional, Tuple
import time


def get_database_summary() -> dict:
    """Get database summary - safe version with comprehensive error handling"""
    try:
        # Try to import and use the sales_database
        from sales_database import SalesDatabase
        db = SalesDatabase()
        return db.get_summary()
    except Exception as e:
        # Return empty summary on any error
        return {
            'total_rows': 0,
            'min_date': 'N/A',
            'max_date': 'N/A',
            'error': str(e)
        }


def streamlit_multi_upload_ui(data_type: str = "Sales") -> Tuple[Optional[pd.DataFrame], str]:
    """
    Streamlit UI for multi-file uploads with clear, step-by-step flow
    SIMPLIFIED VERSION - Self-contained with minimal dependencies
    
    Returns:
        Tuple of (merged_dataframe, merge_type)
    """
    
    st.subheader(f"📂 {data_type} Data Management")
    
    # Step 1: Show current database status (SAFE VERSION)
    try:
        summary = get_database_summary()
        
        if summary.get('total_rows', 0) > 0:
            st.success(
                f"✅ **Database has {summary['total_rows']:,} rows** "
                f"({summary['min_date']} to {summary['max_date']})"
            )
        else:
            st.info("📦 Database is empty - upload data to get started")
    except Exception as e:
        st.warning(f"⚠️ Could not load database status: {str(e)}")
    
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
            try:
                file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
                
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error(f"❌ Unsupported file type: {uploaded_file.name}")
                    continue
                
                # Show success with details
                st.success(f"✅ Loaded {uploaded_file.name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("File Size", f"{file_size:.2f} MB")
                
                new_data_list.append(df)
                total_new_rows += len(df)
                
            except Exception as e:
                st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
                continue
    
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
        try:
            summary = get_database_summary()
            current_rows = summary.get('total_rows', 0)
        except Exception as e:
            st.warning(f"⚠️ Could not retrieve database status: {str(e)}")
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
                try:
                    from sales_database import SalesDatabase
                    db = SalesDatabase()
                    if db.save_dataframe(new_combined, f"{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                        st.success("✅ **SUCCESS!** Data saved to database")
                        st.balloons()
                    else:
                        st.error("❌ Failed to save data")
                except Exception as e:
                    st.error(f"❌ Error saving data: {str(e)}")
    
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
