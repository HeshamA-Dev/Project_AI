import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

DATA_PATH = "data/cleaned/ai_job_market_insights_cleaned.csv"
MODEL_PATH = "model/job_growth_model.pkl"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print(df.head())


# --------------------------------------------------
# 2. Create binary target column
# --------------------------------------------------

df["Job_Growth_Binary"] = df["Job_Growth_Projection"].map({
    "Growth": "Growing",
    "Stable": "Not Growing",
    "Decline": "Not Growing"
})

print("\nNew target distribution:")
print(df["Job_Growth_Binary"].value_counts())


# --------------------------------------------------
# 3. Define features and target
# --------------------------------------------------

target_column = "Job_Growth_Binary"

features = [
    "Job_Title",
    "Company_Size",
    "Location",
    "AI_Adoption_Level",
    "Automation_Risk",
    "Remote_Friendly"
]

X = df[features]
y = df[target_column]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(target_column)


# --------------------------------------------------
# 4. Define categorical and numerical columns
# --------------------------------------------------

categorical_columns = [
    "Job_Title",
    "Company_Size",
    "Location",
    "AI_Adoption_Level",
    "Automation_Risk"
]

numerical_columns = [
    "Remote_Friendly"
]


# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. Preprocessing and model pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("num", "passthrough", numerical_columns)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel trained successfully.")


# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("Accuracy:", round(accuracy, 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 9. Save model
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully.")
print("Saved to:", MODEL_PATH)