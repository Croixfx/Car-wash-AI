import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_messy_carwash_dataset(n_records=5000):
    """
    Create a realistic but messy car wash dataset with intentional data quality issues
    """
    np.random.seed(42)
    random.seed(42)
    
    # Customer base with different behavior patterns
    customer_ids = [f'C{str(i).zfill(4)}' for i in range(1, 1001)]
    
    # Service types with realistic pricing
    service_types = {
        'basic wash': {'price_range': (15, 25), 'duration_range': (15, 25)},
        'Basic Wash': {'price_range': (15, 25), 'duration_range': (15, 25)},  # Duplicate with different case
        'PREMIUM WASH': {'price_range': (35, 50), 'duration_range': (30, 45)},
        'Premium Wash': {'price_range': (35, 50), 'duration_range': (30, 45)},
        'deluxe wash': {'price_range': (55, 75), 'duration_range': (45, 60)},
        'Interior Detailing': {'price_range': (80, 120), 'duration_range': (60, 90)},
        'FULL SERVICE': {'price_range': (100, 150), 'duration_range': (90, 120)},
        'full service': {'price_range': (100, 150), 'duration_range': (90, 120)}
    }
    
    # Vehicle types with inconsistencies
    vehicle_types = ['Sedan', 'SUV', 'suv', 'Truck', 'truck', 'Hatchback', 'HATCHBACK', 'Luxury', 'luxury', 'Van']
    
    # Locations
    locations = ['Downtown', 'MALL', 'mall', 'Highway', 'highway', 'Suburban', 'Airport', 'AIRPORT']
    
    # Payment methods with variations
    payment_methods = ['Cash', 'CASH', 'Credit Card', 'credit card', 'Debit Card', 'debit card', 
                      'Mobile Pay', 'MOBILE PAY', 'mobile pay', 'Membership', 'membership']
    
    data = []
    start_date = datetime(2022, 1, 1)
    
    # Create customer behavior patterns
    customer_profiles = {}
    for cust_id in customer_ids:
        customer_profiles[cust_id] = {
            'preferred_service': random.choice(list(service_types.keys())),
            'avg_visit_frequency': random.randint(20, 120),  # days between visits
            'last_visit': start_date - timedelta(days=random.randint(1, 365)),
            'spending_level': random.uniform(0.3, 1.0)
        }
    
    # Generate messy records
    for i in range(n_records):
        # INTENTIONAL DATA QUALITY ISSUES
        
        # 1. Missing values (5-10% missing rate)
        missing_chance = random.random()
        
        # Select customer
        customer_id = random.choice(customer_ids)
        profile = customer_profiles[customer_id]
        
        # Calculate service date with some realistic patterns
        if random.random() < 0.7:  # 70% follow their pattern
            service_date = profile['last_visit'] + timedelta(
                days=profile['avg_visit_frequency'] + random.randint(-14, 14)
            )
        else:
            service_date = start_date + timedelta(days=random.randint(0, 1095))  # 3-year span
        
        # 2. Inconsistent service type selection
        if random.random() < 0.6:
            service_type = profile['preferred_service']
        else:
            service_type = random.choice(list(service_types.keys()))
        
        service_info = service_types[service_type]
        
        # 3. Missing vehicle_type (8% missing)
        if missing_chance > 0.08:
            vehicle_type = random.choice(vehicle_types)
            # 4. Inconsistent text casing (15% inconsistent)
            if random.random() < 0.15:
                if random.random() < 0.5:
                    vehicle_type = vehicle_type.lower()
                else:
                    vehicle_type = vehicle_type.upper()
        else:
            vehicle_type = np.nan
        
        # 5. Service cost with outliers and errors
        base_price_range = service_info['price_range']
        service_cost = round(random.uniform(base_price_range[0], base_price_range[1]), 2)
        
        # Add outliers (3% extreme values)
        if random.random() < 0.03:
            service_cost *= random.uniform(3, 10)
        
        # Add negative values error (1% errors)
        if random.random() < 0.01:
            service_cost = -abs(service_cost)
        
        # 6. Service duration with inconsistencies
        duration_range = service_info['duration_range']
        service_duration = random.randint(duration_range[0], duration_range[1])
        
        # 7. Customer rating with missing values and outliers
        if missing_chance > 0.1:  # 10% missing ratings
            base_rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.2, 0.4, 0.25])[0]
            # Rating outliers (2% invalid ratings)
            if random.random() < 0.02:
                customer_rating = random.choice([0, 6, 7, 10, 99])
            else:
                customer_rating = base_rating
        else:
            customer_rating = np.nan
        
        # 8. Payment method inconsistencies
        payment_method = random.choice(payment_methods)
        
        # 9. Location with inconsistencies
        location = random.choice(locations)
        
        # 10. Intentional date format inconsistencies
        if random.random() < 0.2:  # 20% different date formats
            if random.random() < 0.5:
                date_str = service_date.strftime('%m/%d/%Y')
            else:
                date_str = service_date.strftime('%d-%m-%Y')
        else:
            date_str = service_date.strftime('%Y-%m-%d')
        
        record = {
            'customer_id': customer_id,
            'service_date': date_str,  # Mixed formats
            'vehicle_type': vehicle_type,
            'service_type': service_type,
            'service_cost': service_cost,
            'service_duration': service_duration,
            'customer_rating': customer_rating,
            'payment_method': payment_method,
            'location': location,
            'record_source': random.choice(['Web', 'MOBILE', 'In-Person', 'in-person', 'WEB'])
        }
        
        data.append(record)
        
        # Update customer's last visit
        customer_profiles[customer_id]['last_visit'] = service_date
    
    df = pd.DataFrame(data)
    
    # ADD MORE DATA QUALITY ISSUES
    
    # 11. Duplicate records (7% duplicates)
    print("Adding duplicate records...")
    duplicates = df.sample(frac=0.07)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 12. Add some completely empty rows (1%)
    empty_indices = df.sample(frac=0.01).index
    for col in df.columns:
        if col != 'customer_id':  # Keep at least one identifier
            df.loc[empty_indices, col] = np.nan
    
    # 13. Add inconsistent numeric entries
    numeric_error_indices = df.sample(frac=0.02).index
    df.loc[numeric_error_indices, 'service_cost'] = 'unknown'
    df.loc[numeric_error_indices, 'service_duration'] = 'N/A'
    
    # 14. Add whitespace issues
    whitespace_indices = df.sample(frac=0.05).index
    df.loc[whitespace_indices, 'vehicle_type'] = '  ' + df.loc[whitespace_indices, 'vehicle_type'].astype(str) + '  '
    df.loc[whitespace_indices, 'service_type'] = '  ' + df.loc[whitespace_indices, 'service_type'].astype(str) + '  '
    
    # 15. Add special characters and typos
    typo_indices = df.sample(frac=0.03).index
    df.loc[typo_indices, 'vehicle_type'] = df.loc[typo_indices, 'vehicle_type'].astype(str) + '!'
    df.loc[typo_indices, 'service_type'] = '#' + df.loc[typo_indices, 'service_type'].astype(str)
    
    print("✅ Messy dataset created with intentional data quality issues!")
    return df

def analyze_data_issues(df):
    """
    Analyze and report all data quality issues in the dataset
    """
    print("\n" + "="*60)
    print("DATA QUALITY ISSUES ANALYSIS")
    print("="*60)
    
    print(f"Dataset Shape: {df.shape}")
    
    # 1. Missing Values Analysis
    print("\n🔍 MISSING VALUES:")
    missing_data = df.isnull().sum()
    for col, count in missing_data.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"   • {col}: {count} missing ({percentage:.1f}%)")
    
    # 2. Duplicate Analysis
    print(f"\n🔄 DUPLICATE RECORDS: {df.duplicated().sum()}")

    # 3. Data Type Inconsistencies
    print(f"\n📊 DATA TYPE ISSUES:")
    print(f"   • service_date: Mixed formats (YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY)")
    print(f"   • service_cost: Mixed numeric and text values")
    
    # 4. Text Inconsistencies
    print(f"\n📝 TEXT INCONSISTENCIES:")
    for col in ['vehicle_type', 'service_type', 'payment_method', 'location']:
        unique_vals = df[col].astype(str).str.strip().unique()
        if len(unique_vals) > 10:  # Show if too many variations
            print(f"   • {col}: {len(unique_vals)} unique values (many inconsistencies)")
        else:
            variations = [val for val in unique_vals if pd.notna(val)]
            print(f"   • {col}: Variations include {variations[:3]}...")
    
    # 5. Outlier Analysis
    print(f"\n📈 OUTLIERS AND ERRORS:")
    try:
        numeric_cost = pd.to_numeric(df['service_cost'], errors='coerce')
        cost_outliers = numeric_cost[(numeric_cost < 0) | (numeric_cost > 500)]
        print(f"   • service_cost: {len(cost_outliers)} outliers/errors")
        
        rating_outliers = df['customer_rating'][~df['customer_rating'].isin([1, 2, 3, 4, 5, np.nan])]
        print(f"   • customer_rating: {len(rating_outliers)} invalid ratings")
    except:
        print(f"   • Cannot analyze outliers due to data type issues")
    
    # 6. Whitespace Issues
    whitespace_issues = 0
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].astype(str).str.contains('^\s+|\s+$').any():
            whitespace_issues += 1
    print(f"   • Columns with whitespace issues: {whitespace_issues}")

# Create the messy dataset
print("🚗 CREATING MESSY CAR WASH DATASET...")
print("This dataset will have intentional data quality issues for preprocessing demonstration")

messy_data = create_messy_carwash_dataset(5000)

# Analyze the issues
analyze_data_issues(messy_data)

# Save the messy dataset
output_file = 'smartshine_carwash_messy_data.csv'
messy_data.to_csv(output_file, index=False)

print(f"\n💾 MESSY DATASET SAVED: {output_file}")
print(f"📊 Dataset shape: {messy_data.shape}")

# Show sample of the messy data
print(f"\n👀 SAMPLE OF MESSY DATA (first 10 records):")
print(messy_data.head(10))

print(f"\n📋 COLUMNS IN DATASET:")
for col in messy_data.columns:
    print(f"   • {col}")

print(f"\n🎯 INTENTIONAL DATA QUALITY ISSUES INCLUDED:")
issues = [
    "Missing values in multiple columns",
    "Duplicate records",
    "Inconsistent text casing (SUV vs suv vs SUV)",
    "Mixed date formats",
    "Outliers in service costs",
    "Invalid customer ratings",
    "Whitespace padding",
    "Special characters in text fields",
    "Mixed data types in numeric columns",
    "Empty rows"
]

for issue in issues:
    print(f"   • {issue}")