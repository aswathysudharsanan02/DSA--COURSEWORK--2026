# Import Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv("beer_servings.csv")

# Drop unwanted column
df.drop("Unnamed: 0", axis=1, inplace=True)

# Handle missing values
df["beer_servings"] = df["beer_servings"].fillna(df["beer_servings"].median())
df["spirit_servings"] = df["spirit_servings"].fillna(df["spirit_servings"].median())
df["wine_servings"] = df["wine_servings"].fillna(df["wine_servings"].median())

df = df.dropna(subset=["total_litres_of_pure_alcohol"])

# Features
X = df[
    [
        "country",
        "beer_servings",
        "spirit_servings",
        "wine_servings",
        "continent"
    ]
]

# Target
y = df["total_litres_of_pure_alcohol"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Categorical columns
cat_cols = ["country", "continent"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols
        )
    ],
    remainder="passthrough"
)

# Linear Regression

lr_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression R2:")
print(lr_r2)


# Random Forest

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="r2"
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

rf_pred = best_rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest R2:")
print(rf_r2)

print("\nBest Parameters:")
print(grid_search.best_params_)

import joblib

if lr_r2 >= rf_r2:
    best_model = lr_model
    print("\nBest Model: Linear Regression")
else:
    best_model = best_rf
    print("\nBest Model: Random Forest")

joblib.dump(best_model, "best_model.pkl")

print("\nBest model saved as best_model.pkl")