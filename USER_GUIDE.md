# Store Style Efficiency Analysis - User Guide

## 🎯 Overview
This tool analyzes your retail data to determine the optimal number of styles each store should carry to maximize efficiency and profitability.

## 📁 Required Data Files

### 1. Sales Data (CSV/Excel)
**Required columns:**
- `DATE` - Transaction date (YYYY-MM-DD format)
- `STORE` - Store identifier/code
- `SKU` - Product SKU/barcode
- `QUANTITY` - Quantity sold

**Optional columns:**
- `STORE_NAME` - Store display name

**Example:**
```
DATE,STORE,SKU,QUANTITY,STORE_NAME
2025-10-01,ST001,SKU12345,5,Mumbai Central
2025-10-01,ST001,SKU12346,2,Mumbai Central
```

### 2. Stock Data (CSV/Excel)
**Required columns:**
- `STORE` - Store identifier/code (must match sales data)
- `SKU` - Product SKU/barcode
- `STOCK` - Current stock quantity

**Example:**
```
STORE,SKU,STOCK
ST001,SKU12345,25
ST001,SKU12346,10
```

### 3. Warehouse Data (CSV/Excel)
**Required columns:**
- `SKU` - Product SKU/barcode
- `WAREHOUSE_STOCK` - Available warehouse stock

**Example:**
```
SKU,WAREHOUSE_STOCK
SKU12345,500
SKU12346,300
```

### 4. SKU Master Data (CSV/Excel)
**Required columns:**
- `SKU` - Product SKU/barcode
- `STYLE` - Style code/identifier
- `COLOR` - Color name
- `SIZE` - Size designation

**Example:**
```
SKU,STYLE,COLOR,SIZE
SKU12345,ST001,BLACK,M
SKU12346,ST001,BLACK,L
```

### 5. Style Master Data (CSV/Excel) - Optional
**Required columns:**
- `STYLE` - Style code (must match SKU master)
- `GENDER` - Gender category (MENS/WOMENS/KIDS/UNISEX)

**Example:**
```
STYLE,GENDER
ST001,MENS
ST002,WOMENS
```

## 🚀 How to Run the Analysis

### Step 1: Prepare Your Data
1. Export your data in CSV or Excel format
2. Ensure column names match the requirements above
3. Place files in the same folder as the analysis scripts

### Step 2: Configure File Paths
1. Open `run_my_analysis.py`
2. Update the `file_paths` dictionary with your actual file paths:

```python
file_paths = {
    'sales': r'C:\your\path\to\sales_data.csv',
    'stock': r'C:\your\path\to\stock_data.xlsx',
    'warehouse': r'C:\your\path\to\warehouse_data.csv',
    'sku_master': r'C:\your\path\to\sku_master.xlsx',
    'style_master': r'C:\your\path\to\style_master.csv'  # Optional
}
```

### Step 3: Adjust Analysis Parameters (Optional)
```python
analysis_params = {
    'time_period_days': 90,        # 30, 60, 90, or 120 days
    'efficiency_threshold': 60     # 30-80 (minimum score for viable styles)
}
```

### Step 4: Run the Analysis
```bash
python run_my_analysis.py
```

## 📊 Understanding the Results

### Excel Output Sheets

#### 1. Store_Recommendations
- **Main summary** of recommendations per store
- **STORE_CATEGORY**: Performance level (High/Good/Average/Needs Improvement)
- **CURRENT_TOTAL_STYLES**: Number of styles currently carried
- **RECOMMENDED_STYLES**: Optimal number of styles
- **EFFICIENCY_IMPROVEMENT**: Change required (+/- styles)

#### 2. Style_Performance
- **Detailed metrics** for every store-style combination
- **STYLE_EFFICIENCY_SCORE**: Overall efficiency (0-100)
- **SALES_VELOCITY**: Daily sales rate
- **STOCK_TURNOVER**: How fast styles move
- **DAYS_STOCK_COVER**: Days of stock remaining

#### 3. Top_Performing_Styles
- **Best 50 styles** across all stores
- Use this to identify star performers

#### 4. Category_Analysis
- **Store category breakdown** with averages
- Compare performance across store types

#### 5. Detailed_Style_Recs
- **Specific style recommendations** for each store
- Exact styles each store should focus on

#### 6. Summary_Statistics
- **Overall analysis summary**
- Key metrics and totals

### Key Metrics Explained

#### Efficiency Score (0-100)
- **30% Sales Velocity**: How fast the style sells daily
- **25% Stock Turnover**: Sales vs stock ratio
- **20% Consistency**: Regular sales pattern
- **15% Revenue Contribution**: Total sales volume
- **10% Stock Efficiency**: Optimal stock levels

#### Store Categories
- **High Performer (≥70 score)**: Can handle more styles efficiently
- **Good Performer (50-69 score)**: Maintain current efficient styles
- **Average Performer (30-49 score)**: Focus on fewer, better styles
- **Needs Improvement (<30 score)**: Significant style reduction needed

## 💡 Business Impact

### Focus Strategy Benefits
- **Improved inventory turnover**
- **Reduced stock-outs on popular items**
- **Lower inventory carrying costs**
- **Simplified operations**
- **Better sales per square foot**

### Implementation Tips
1. **Start with pilot stores** from each performance category
2. **Phase implementation** over 2-3 months
3. **Monitor sales impact** weekly
4. **Adjust based on seasonal patterns**
5. **Re-run analysis quarterly** for optimization

## ❓ Troubleshooting

### Common Issues

#### "File not found" error
- Check file paths are correct
- Ensure files exist in specified locations
- Use forward slashes (/) or raw strings (r'path')

#### "Missing columns" error
- Verify column names match requirements exactly
- Check for extra spaces in column names
- Ensure data types are correct (dates, numbers)

#### "No efficient styles found"
- Lower the efficiency_threshold (try 40-50)
- Check if you have enough sales history
- Verify SKU master data is complete

#### Analysis shows no recommendations
- Increase the time_period_days (try 120)
- Check data quality and completeness
- Ensure sales data includes recent transactions

## 📞 Support
If you encounter issues or need customization:
1. Check your data format matches the examples
2. Verify all required columns are present
3. Ensure data covers the analysis period
4. Review the troubleshooting section above

## 🔄 Regular Usage
Run this analysis:
- **Monthly**: For tactical adjustments
- **Quarterly**: For strategic planning
- **Before seasons**: For seasonal optimization
- **After new launches**: To assess style performance