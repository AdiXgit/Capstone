"""
Karnataka Paddy Yield Prediction - ML Starter Script
=====================================================

This script demonstrates how to build ML models using the Karnataka Paddy dataset.
Includes: data loading, preprocessing, feature engineering, model training, and evaluation.

Requirements:
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("Karnataka Paddy Yield Prediction - ML Pipeline")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n1. Loading data...")

# Load main dataset
df = pd.read_csv('Karnataka_Paddy_Main_Dataset.csv')
print(f"   ✓ Loaded {len(df):,} records with {len(df.columns)} features")

# Quick exploration
print(f"\n   Data shape: {df.shape}")
print(f"   Date range: {df['Year'].min()} to {df['Year'].max()}")
print(f"   Districts: {df['District'].nunique()}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n2. Data preprocessing...")

# Handle categorical variables
label_encoders = {}
categorical_cols = ['District', 'Region', 'Soil_Quality', 'Season', 
                   'Seed_Variety', 'Sowing_Method', 'Water_Management',
                   'Weed_Management', 'Pest_Incidence', 'Disease_Incidence',
                   'Harvester_Used']

df_encoded = df.copy()

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print(f"   ✓ Encoded {len(categorical_cols)} categorical features")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n3. Feature engineering...")

# Create derived features
df_encoded['Fertilizer_Total'] = (df_encoded['Urea_Applied_kg_per_ha'] + 
                                   df_encoded['DAP_Applied_kg_per_ha'] + 
                                   df_encoded['Potash_Applied_kg_per_ha'])

df_encoded['Rainfall_Irrigation_Index'] = (df_encoded['Rainfall_mm'] * 
                                            df_encoded['Irrigation_Percent'] / 100)

df_encoded['Soil_NPK_Balance'] = (df_encoded['Soil_Nitrogen_kg_per_ha'] + 
                                   df_encoded['Soil_Phosphorus_kg_per_ha'] + 
                                   df_encoded['Soil_Potassium_kg_per_ha'])

df_encoded['Cost_Per_Yield'] = (df_encoded['Cost_of_Cultivation_Rs_per_ha'] / 
                                 (df_encoded['Yield_Kg_per_Ha'] + 1))

df_encoded['Profit_Margin'] = (df_encoded['Net_Income_Rs_per_ha'] / 
                                (df_encoded['Gross_Revenue_Rs_per_ha'] + 1))

print(f"   ✓ Created 5 derived features")

# ============================================================================
# 4. FEATURE SELECTION
# ============================================================================
print("\n4. Feature selection...")

# Select features for modeling
feature_cols = [
    # Environmental
    'Rainfall_mm', 'Temperature_Celsius', 'Humidity_Percent',
    
    # Soil
    'Soil_pH', 'Soil_Organic_Carbon_Percent', 
    'Soil_Nitrogen_kg_per_ha', 'Soil_Phosphorus_kg_per_ha', 'Soil_Potassium_kg_per_ha',
    
    # Inputs
    'Urea_Applied_kg_per_ha', 'DAP_Applied_kg_per_ha', 'Potash_Applied_kg_per_ha',
    'Organic_Fertilizer_kg_per_ha', 'Pesticide_Sprays',
    
    # Management
    'Irrigation_Percent', 'Labor_Days_per_ha', 'Tractor_Hours_per_ha',
    
    # Categorical (encoded)
    'District', 'Region', 'Soil_Quality', 'Season', 'Seed_Variety',
    'Sowing_Method', 'Water_Management', 'Weed_Management',
    
    # Derived
    'Fertilizer_Total', 'Rainfall_Irrigation_Index', 'Soil_NPK_Balance'
]

# Filter available features
feature_cols = [col for col in feature_cols if col in df_encoded.columns]
target_col = 'Yield_Kg_per_Ha'

X = df_encoded[feature_cols]
y = df_encoded[target_col]

print(f"   ✓ Selected {len(feature_cols)} features")
print(f"   Target: {target_col}")

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================
print("\n5. Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   ✓ Training set: {len(X_train)} samples")
print(f"   ✓ Test set: {len(X_test)} samples")

# ============================================================================
# 6. MODEL TRAINING
# ============================================================================
print("\n6. Training models...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mae': mae,
        'rmse': rmse,
        'predictions': y_pred_test
    }
    
    print(f"      Train R²: {train_r2:.4f}")
    print(f"      Test R²:  {test_r2:.4f}")
    print(f"      MAE:      {mae:.2f} kg/ha")
    print(f"      RMSE:     {rmse:.2f} kg/ha")

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("7. Model Comparison")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test R²': [results[m]['test_r2'] for m in results.keys()],
    'MAE (kg/ha)': [results[m]['mae'] for m in results.keys()],
    'RMSE (kg/ha)': [results[m]['rmse'] for m in results.keys()]
})

print("\n", comparison_df.to_string(index=False))

# Best model
best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
print(f"\n✓ Best Model: {best_model_name} (R² = {results[best_model_name]['test_r2']:.4f})")

# ============================================================================
# 8. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n8. Feature importance analysis...")

if best_model_name in ['Random Forest', 'Gradient Boosting']:
    best_model = results[best_model_name]['model']
    
    # Get feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n   Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(15), x='Importance', y='Feature')
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n   ✓ Feature importance plot saved: feature_importance.png")

# ============================================================================
# 9. PREDICTIONS VISUALIZATION
# ============================================================================
print("\n9. Creating visualizations...")

# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, results[best_model_name]['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Yield (kg/ha)')
plt.ylabel('Predicted Yield (kg/ha)')
plt.title(f'Actual vs Predicted Yield - {best_model_name}')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("   ✓ Actual vs Predicted plot saved: actual_vs_predicted.png")

# Residuals
plt.figure(figsize=(10, 6))
residuals = y_test - results[best_model_name]['predictions']
plt.scatter(results[best_model_name]['predictions'], residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Yield (kg/ha)')
plt.ylabel('Residuals (kg/ha)')
plt.title(f'Residual Plot - {best_model_name}')
plt.tight_layout()
plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
print("   ✓ Residual plot saved: residuals.png")

# ============================================================================
# 10. SAVE MODEL
# ============================================================================
print("\n10. Saving model...")

import pickle

# Save best model
model_filename = f'best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(results[best_model_name]['model'], f)

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Save feature list
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"   ✓ Model saved: {model_filename}")
print(f"   ✓ Label encoders saved: label_encoders.pkl")
print(f"   ✓ Feature columns saved: feature_columns.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n✓ Dataset: {len(df):,} records processed")
print(f"✓ Features: {len(feature_cols)} selected")
print(f"✓ Best Model: {best_model_name}")
print(f"✓ Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"✓ MAE: {results[best_model_name]['mae']:.2f} kg/ha")
print(f"✓ RMSE: {results[best_model_name]['rmse']:.2f} kg/ha")
print(f"\n✓ Model ready for deployment!")
print("\n" + "=" * 80)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
print("\n" + "=" * 80)
print("USAGE EXAMPLE - Making Predictions")
print("=" * 80)

usage_code = """
# Load saved model and make predictions

import pickle
import pandas as pd

# Load model and encoders
with open('best_model_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Create new data point
new_data = {
    'District': 'Mandya',
    'Season': 'Kharif',
    'Rainfall_mm': 800,
    'Temperature_Celsius': 28,
    'Irrigation_Percent': 85,
    'Soil_pH': 7.0,
    # ... add all required features
}

# Encode categorical variables
new_data_encoded = new_data.copy()
for col, encoder in label_encoders.items():
    if col in new_data_encoded:
        new_data_encoded[col] = encoder.transform([new_data_encoded[col]])[0]

# Create DataFrame with correct feature order
X_new = pd.DataFrame([new_data_encoded])[feature_cols]

# Predict
predicted_yield = model.predict(X_new)[0]
print(f"Predicted Yield: {predicted_yield:.2f} kg/ha")
"""

print(usage_code)
print("=" * 80)
