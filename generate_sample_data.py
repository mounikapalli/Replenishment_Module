"""
Test script to generate sample data and run Store Style Efficiency Analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducible results
random.seed(42)
np.random.seed(42)

def generate_sample_data():
    """Generate realistic sample data for testing Store Style Efficiency Analysis"""
    
    # Generate date range (last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Store data
    stores = [
        ('ST001', 'Delhi Main Store'),
        ('ST002', 'Mumbai Central'),
        ('ST003', 'Bangalore Forum'),
        ('ST004', 'Chennai Express'),
        ('ST005', 'Kolkata Mall'),
        ('ST006', 'Pune City'),
        ('ST007', 'Hyderabad Hub'),
        ('ST008', 'Ahmedabad Plaza'),
        ('ST009', 'Gurgaon Select'),
        ('ST010', 'Noida Fashion')
    ]
    
    # Style data with different performance levels
    styles = [
        ('STY001', 'MENS', 'Casual Shirt'),
        ('STY002', 'WOMENS', 'Summer Dress'),
        ('STY003', 'MENS', 'Formal Trouser'),
        ('STY004', 'WOMENS', 'Ethnic Wear'),
        ('STY005', 'MENS', 'T-Shirt'),
        ('STY006', 'WOMENS', 'Jeans'),
        ('STY007', 'MENS', 'Blazer'),
        ('STY008', 'WOMENS', 'Top'),
        ('STY009', 'MENS', 'Shorts'),
        ('STY010', 'WOMENS', 'Skirt'),
        ('STY011', 'MENS', 'Jacket'),
        ('STY012', 'WOMENS', 'Kurti'),
        ('STY013', 'MENS', 'Polo'),
        ('STY014', 'WOMENS', 'Saree'),
        ('STY015', 'MENS', 'Hoodie')
    ]
    
    colors = ['Black', 'White', 'Blue', 'Red', 'Green', 'Navy', 'Grey', 'Beige']
    sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    
    # Generate SKU Master
    sku_master = []
    sku_id = 1
    
    for style_code, gender, style_name in styles:
        for color in colors[:4]:  # 4 colors per style
            for size in sizes:
                sku = f"SKU{sku_id:05d}"
                sku_master.append({
                    'SKU': sku,
                    'STYLE': style_code,
                    'COLOR': color,
                    'SIZE': size
                })
                sku_id += 1
    
    sku_master_df = pd.DataFrame(sku_master)
    
    # Generate Style Master
    style_master_df = pd.DataFrame([
        {'STYLE': style_code, 'GENDER': gender} 
        for style_code, gender, _ in styles
    ])
    
    # Generate Sales Data with realistic patterns
    sales_data = []
    
    # Define performance tiers for styles
    high_performing_styles = ['STY001', 'STY002', 'STY005', 'STY006', 'STY012']
    medium_performing_styles = ['STY003', 'STY004', 'STY008', 'STY013']
    low_performing_styles = ['STY007', 'STY009', 'STY010', 'STY011', 'STY014', 'STY015']
    
    # Define store performance levels
    high_performing_stores = ['ST001', 'ST002', 'ST003']
    medium_performing_stores = ['ST004', 'ST005', 'ST006', 'ST007']
    low_performing_stores = ['ST008', 'ST009', 'ST010']
    
    for date in date_range:
        for store_code, store_name in stores:
            # Determine store performance multiplier
            if store_code in high_performing_stores:
                store_multiplier = random.uniform(1.5, 2.0)
            elif store_code in medium_performing_stores:
                store_multiplier = random.uniform(1.0, 1.5)
            else:
                store_multiplier = random.uniform(0.5, 1.0)
            
            # Generate sales for different styles with varying probabilities
            for style_code, _, _ in styles:
                # Determine style performance
                if style_code in high_performing_styles:
                    base_probability = 0.7
                    base_quantity_range = (2, 8)
                elif style_code in medium_performing_styles:
                    base_probability = 0.4
                    base_quantity_range = (1, 5)
                else:
                    base_probability = 0.2
                    base_quantity_range = (1, 3)
                
                # Adjust probability by store performance
                sale_probability = base_probability * store_multiplier
                sale_probability = min(sale_probability, 0.9)  # Cap at 90%
                
                if random.random() < sale_probability:
                    # Select random SKUs from this style
                    style_skus = sku_master_df[sku_master_df['STYLE'] == style_code]['SKU'].tolist()
                    selected_skus = random.sample(style_skus, min(random.randint(1, 3), len(style_skus)))
                    
                    for sku in selected_skus:
                        quantity = random.randint(*base_quantity_range)
                        sales_data.append({
                            'DATE': date,
                            'STORE': store_code,
                            'STORE_NAME': store_name,
                            'SKU': sku,
                            'QUANTITY': quantity
                        })
    
    sales_df = pd.DataFrame(sales_data)
    
    # Generate Stock Data
    stock_data = []
    
    for store_code, store_name in stores:
        for sku in sku_master_df['SKU']:
            # Calculate stock based on sales history for this store-SKU combination
            sku_sales = sales_df[(sales_df['STORE'] == store_code) & (sales_df['SKU'] == sku)]
            
            if len(sku_sales) > 0:
                avg_daily_sales = sku_sales['QUANTITY'].sum() / 90
                # Stock between 5-30 days of coverage
                stock_level = int(avg_daily_sales * random.uniform(5, 30))
                # Some items might be out of stock
                if random.random() < 0.15:  # 15% chance of stockout
                    stock_level = 0
            else:
                # Items with no sales get random low stock
                stock_level = random.randint(0, 5)
                if random.random() < 0.3:  # 30% chance of no stock for non-selling items
                    stock_level = 0
            
            stock_data.append({
                'STORE': store_code,
                'STORE_NAME': store_name,
                'SKU': sku,
                'STOCK': stock_level
            })
    
    stock_df = pd.DataFrame(stock_data)
    
    # Generate Warehouse Data
    warehouse_data = []
    
    for sku in sku_master_df['SKU']:
        # Calculate total sales across all stores
        total_sku_sales = sales_df[sales_df['SKU'] == sku]['QUANTITY'].sum()
        
        if total_sku_sales > 0:
            # Warehouse stock based on total demand
            warehouse_stock = int(total_sku_sales * random.uniform(0.8, 2.0))
        else:
            # Non-selling items get lower warehouse stock
            warehouse_stock = random.randint(0, 20)
        
        warehouse_data.append({
            'SKU': sku,
            'WAREHOUSE_STOCK': warehouse_stock
        })
    
    warehouse_df = pd.DataFrame(warehouse_data)
    
    return sales_df, stock_df, warehouse_df, sku_master_df, style_master_df

def save_sample_data():
    """Generate and save sample data to CSV files"""
    print("Generating sample data...")
    
    sales_df, stock_df, warehouse_df, sku_master_df, style_master_df = generate_sample_data()
    
    # Save to CSV files
    sales_df.to_csv('sample_sales_data.csv', index=False)
    stock_df.to_csv('sample_stock_data.csv', index=False)
    warehouse_df.to_csv('sample_warehouse_data.csv', index=False)
    sku_master_df.to_csv('sample_sku_master.csv', index=False)
    style_master_df.to_csv('sample_style_master.csv', index=False)
    
    # Print summary statistics
    print(f"\n✅ Sample data generated successfully!")
    print(f"📊 Sales Records: {len(sales_df):,}")
    print(f"🏪 Stores: {sales_df['STORE'].nunique()}")
    print(f"📦 SKUs: {sku_master_df['SKU'].nunique()}")
    print(f"👔 Styles: {style_master_df['STYLE'].nunique()}")
    print(f"📈 Date Range: {sales_df['DATE'].min()} to {sales_df['DATE'].max()}")
    print(f"💰 Total Sales Quantity: {sales_df['QUANTITY'].sum():,}")
    print(f"📋 Total Stock: {stock_df['STOCK'].sum():,}")
    print(f"🏭 Total Warehouse Stock: {warehouse_df['WAREHOUSE_STOCK'].sum():,}")
    
    # Show sample data
    print(f"\n📋 Sample Sales Data:")
    print(sales_df.head())
    
    print(f"\n📦 Sample Stock Data:")
    print(stock_df.head())
    
    print(f"\n🏭 Sample Warehouse Data:")
    print(warehouse_df.head())
    
    return sales_df, stock_df, warehouse_df, sku_master_df, style_master_df

if __name__ == "__main__":
    save_sample_data()