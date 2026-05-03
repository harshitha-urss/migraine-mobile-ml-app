import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("data/migraine_symptom_classification/migraine_symptom_classification.csv")

# =========================
# 2. CLEAN DATA
# =========================
df = df.dropna()
df.columns = df.columns.str.strip()

# =========================
# 3. ENCODE CATEGORICAL DATA
# =========================
label_encoders = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# =========================
# 4. SPLIT FEATURES & TARGET
# =========================
X = df.drop("Type", axis=1)
y = df["Type"]

# Save feature order (VERY IMPORTANT 🔥)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "features.pkl")

# =========================
# 5. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. TRAIN MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_acc = 0

print("\nModel Accuracies:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    print(f"{name}: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# =========================
# 7. SAVE MODEL
# =========================
joblib.dump(best_model, "best_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("\n=========================")
print(f"Best Model: {best_name}")
print(f"Best Accuracy: {best_acc:.4f}")
print("Model saved successfully!")
print("=========================")