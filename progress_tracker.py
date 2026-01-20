"""
Progress Tracker - Display file processing progress
"""
import streamlit as st
import time
from datetime import datetime, timedelta
from typing import Optional


class ProcessingProgressTracker:
    """Track file processing progress"""
    
    def __init__(self, total_rows: int, total_bytes: int, operation_name: str = "Processing"):
        self.total_rows = total_rows
        self.total_bytes = total_bytes
        self.operation_name = operation_name
        self.processed_rows = 0
        self.processed_bytes = 0
        self.start_time = datetime.now()
        self.errors = []
    
    def update(self, rows: int = 0, bytes_processed: int = 0, error: Optional[str] = None):
        """Update progress"""
        self.processed_rows += rows
        self.processed_bytes += bytes_processed
        if error:
            self.errors.append(error)
    
    def get_progress_percentage(self) -> float:
        """Get progress percentage"""
        if self.total_rows == 0:
            return 0
        return min((self.processed_rows / self.total_rows) * 100, 100)
    
    def get_elapsed_time(self) -> timedelta:
        """Get elapsed time"""
        return datetime.now() - self.start_time
    
    def get_eta(self) -> Optional[str]:
        """Get estimated time to completion"""
        elapsed = self.get_elapsed_time().total_seconds()
        if elapsed == 0 or self.processed_rows == 0:
            return "calculating..."
        
        rate = self.processed_rows / elapsed
        remaining_rows = self.total_rows - self.processed_rows
        eta_seconds = remaining_rows / rate
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)
        
        if eta_seconds < 60:
            return f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            return f"{int(eta_seconds / 60)}m {int(eta_seconds % 60)}s"
        else:
            hours = int(eta_seconds / 3600)
            mins = int((eta_seconds % 3600) / 60)
            return f"{hours}h {mins}m"
    
    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.processed_rows >= self.total_rows


def display_processing_progress(tracker: ProcessingProgressTracker, show_details: bool = True):
    """Display processing progress in Streamlit"""
    
    progress_bar = st.progress(tracker.get_progress_percentage() / 100)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Progress", f"{tracker.get_progress_percentage():.1f}%")
    
    with col2:
        elapsed = tracker.get_elapsed_time()
        elapsed_str = f"{int(elapsed.total_seconds())}s"
        st.metric("Elapsed", elapsed_str)
    
    with col3:
        eta = tracker.get_eta()
        st.metric("ETA", eta or "N/A")
    
    if show_details and tracker.processed_rows > 0:
        st.caption(f"Processing {tracker.operation_name}: {tracker.processed_rows:,} / {tracker.total_rows:,} rows")
    
    if tracker.errors:
        with st.expander(f"⚠️ Errors ({len(tracker.errors)})"):
            for error in tracker.errors[-10:]:  # Show last 10 errors
                st.error(error)
