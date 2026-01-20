"""
Quick debug to check STYLE column issues
"""

import pandas as pd

def debug_style_issue():
    print("🔍 Debugging STYLE column issue...")
    
    # Load sales data
    sales_file = r'D:\DATA TILL DATE\Desktop\EBO FOLDER\EBO SALES FOLDER\EBO SALES DATA.xlsx'
    sales_raw = pd.read_excel(sales_file)
    
    print("Sales data shape:", sales_raw.shape)
    print("Sales columns:", list(sales_raw.columns))
    
    # Check STYLE column
    print("\nSTYLE column info:")
    print("Type:", type(sales_raw['STYLE'].iloc[0]))
    print("Sample values:", sales_raw['STYLE'].head(10).tolist())
    print("Unique styles count:", sales_raw['STYLE'].nunique())
    print("Null values:", sales_raw['STYLE'].isnull().sum())
    
    # Check if there are any issues with STYLE column
    style_issues = sales_raw[sales_raw['STYLE'].isnull() | (sales_raw['STYLE'] == '')]
    print(f"Rows with null/empty STYLE: {len(style_issues)}")
    
    # Create the processed dataframe like in the main script
    sales_df = pd.DataFrame({
        'DATE': pd.to_datetime(sales_raw['BILL_DATE'], errors='coerce'),
        'STORE': sales_raw['store_code'].astype(str).str.strip(),
        'STORE_NAME': sales_raw['EBO NAME'].astype(str).str.strip(),
        'SKU': sales_raw['SKU'].astype(str).str.strip(),
        'QUANTITY': pd.to_numeric(sales_raw['BILL_QUANTITY'], errors='coerce').fillna(0),
        'STYLE': sales_raw['STYLE'].astype(str).str.strip(),
        'COLOR': sales_raw['COLOR'].astype(str).str.strip(),
        'SIZE': sales_raw['SIZE'].astype(str).str.strip(),
        'DEPARTMENT': sales_raw['DEPARTMENT'].astype(str).str.strip()
    })
    
    # Filter like in main script
    sales_df = sales_df.dropna(subset=['DATE'])
    sales_df = sales_df[sales_df['QUANTITY'] > 0]
    sales_df = sales_df[sales_df['SKU'] != '']
    
    print(f"\nProcessed sales data shape: {sales_df.shape}")
    print("STYLE column in processed data:")
    print("Sample values:", sales_df['STYLE'].head(10).tolist())
    print("Data types:", sales_df.dtypes)
    
    # Check if STYLE column exists and has correct name
    print("\nColumns in processed DataFrame:")
    for i, col in enumerate(sales_df.columns):
        print(f"  {i}: '{col}' (type: {sales_df[col].dtype})")

if __name__ == "__main__":
    debug_style_issue()