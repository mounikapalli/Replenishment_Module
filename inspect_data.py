"""
Data inspector to see actual column names in EBO files
"""

import pandas as pd

def inspect_ebo_data():
    """Inspect the actual structure of EBO data files"""
    
    file_paths = {
        'sales': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\EBO SALES FOLDER\EBO SALES DATA.xlsx',
        'stock': r'D:\DATA TILL DATE\Desktop\EBO FOLDER\STOCK\EBo Stock Data.xlsx',
        'warehouse': r'D:\DATA TILL DATE\Downloads\Inventory Available for Sales - OMS-2025-10-31T12_38_35.558+05_30.csv',
    }
    
    print("🔍 INSPECTING EBO DATA STRUCTURE")
    print("=" * 50)
    
    # Inspect Sales Data
    print("\n📊 SALES DATA STRUCTURE:")
    try:
        sales_df = pd.read_excel(file_paths['sales'])
        print(f"Shape: {sales_df.shape}")
        print("Columns:")
        for i, col in enumerate(sales_df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print("\nFirst few rows:")
        print(sales_df.head(3).to_string())
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Inspect Stock Data
    print("\n\n📦 STOCK DATA STRUCTURE:")
    try:
        stock_df = pd.read_excel(file_paths['stock'])
        print(f"Shape: {stock_df.shape}")
        print("Columns:")
        for i, col in enumerate(stock_df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print("\nFirst few rows:")
        print(stock_df.head(3).to_string())
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Inspect Warehouse Data
    print("\n\n🏭 WAREHOUSE DATA STRUCTURE:")
    try:
        warehouse_df = pd.read_csv(file_paths['warehouse'])
        print(f"Shape: {warehouse_df.shape}")
        print("Columns:")
        for i, col in enumerate(warehouse_df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print("\nFirst few rows:")
        print(warehouse_df.head(3).to_string())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    inspect_ebo_data()