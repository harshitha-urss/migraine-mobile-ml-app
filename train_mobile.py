import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib

# LOAD DATASET
df = pd.read_csv("data/mobile/Mobile Phone Pricing.csv")

# CLEAN
df = df.dropna()

# SPLIT
X = df.drop("price_range", axis=1)
y = df["price_range"]

# SAVE FEATURES ORDER
feature_columns = X.columns.tolist()

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODELS
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
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name}: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# SAVE
joblib.dump(best_model, "mobile_model.pkl")
joblib.dump(feature_columns, "mobile_features.pkl")

print("\nBest Model:", best_name)
print("Accuracy:", best_acc)