import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load the Processed Data
df = pd.read_csv('smartshine_carwash_processed_data.csv')

# 2. Select Features (X) and Target (y)
# We use the encoded columns and numerical features
features = [
    'vehicle_type_encoded', 
    'service_type_encoded', 
    'service_cost', 
    'service_duration', 
    'customer_rating', 
    'location_encoded', 
    'payment_method_encoded', 
    'visit_frequency', 
    'total_spent'
]
target = 'days_since_last_visit'

# Drop rows where target is missing (if any)
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# 3. Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale Data (Important for SVM and Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODEL 1: Simple Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# --- MODEL 2: Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # Tree models don't necessarily need scaling
y_pred_rf = rf_model.predict(X_test)

# --- MODEL 3: Support Vector Regressor (SVM) ---
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)

# --- MODEL 4: Gradient Boosting (Ensemble method similar to XGBoost) ---
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# 5. Evaluate and Compare
models = {
    "Linear Regression": y_pred_lr,
    "Random Forest": y_pred_rf,
    "Support Vector Machine": y_pred_svr,
    "Gradient Boosting": y_pred_gb
}

results = []
print(f"{'Model':<25} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
print("-" * 65)

for name, preds in models.items():
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    print(f"{name:<25} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f}")

# Optional: Suggesting the best model based on R2 Score (closer to 1.0 is better)
best_model = max(results, key=lambda x: x['R2'])
print(f"\nRecommended Model for Smartshine: {best_model['Model']}")