import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# 1. Dataset Load karna
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

print("--- Dataset Shape ---")
print(df.shape)
print("\n--- First 5 Rows ---")
print(df.head())

# 2. Features aur Target alag karna
X = df.drop(columns=['charges'])
y = df['charges']

# Numerical & Categorical features identify karna
num_features = ['age', 'bmi', 'children']
cat_features = ['sex', 'smoker', 'region']

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Preprocessing Pipeline setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ]
)

# 5. Multiple Models Train aur Evaluate karna
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []

print("\n--- Model Training & Evaluation ---")
for name, model in models.items():
    # Model Pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

# Comparison Table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 6. Best Model (Random Forest) se naya prediction test karna
best_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
best_pipe.fit(X_train, y_train)

# Example sample input
sample_person = pd.DataFrame({
    'age': [35],
    'sex': ['male'],
    'bmi': [28.5],
    'children': [2],
    'smoker': ['no'],
    'region': ['southeast']
})

predicted_cost = best_pipe.predict(sample_person)
print(f"\nEstimated Insurance Cost for Sample Profile: ${predicted_cost[0]:.2f}")