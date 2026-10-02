
import os, json, warnings, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA_FILE = "UAV_Battery_SOC_SOH_Dataset.csv"
OUT = "outputs"
MODEL_DIR = os.path.join(OUT, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

# -------------------- EDA --------------------
plt.figure(figsize=(8,5))
sns.histplot(df["soc_pct"], bins=30, kde=True)
plt.title("SOC Distribution")
plt.xlabel("SOC (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_soc_distribution.png"), dpi=200)
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(df["soh_pct"], bins=30, kde=True)
plt.title("SOH Distribution")
plt.xlabel("SOH (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_soh_distribution.png"), dpi=200)
plt.close()

plt.figure(figsize=(9,5))
sns.countplot(data=df, x="battery_state_category", order=df["battery_state_category"].value_counts().index)
plt.xticks(rotation=25)
plt.title("Battery State Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_battery_state_distribution.png"), dpi=200)
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="cycle_count", y="soh_pct", alpha=0.35)
plt.title("SOH vs Cycle Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_soh_vs_cycle.png"), dpi=200)
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="cycle_count", y="capacity_fade_pct", alpha=0.35)
plt.title("Capacity Fade vs Cycle Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_capacity_fade_vs_cycle.png"), dpi=200)
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="current_a", y="temperature_c", alpha=0.35)
plt.title("Temperature vs Current")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_temperature_vs_current.png"), dpi=200)
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="payload_weight_kg", y="flight_duration_min", alpha=0.35)
plt.title("Payload vs Flight Duration")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "07_payload_vs_flight_duration.png"), dpi=200)
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="power_w", y="flight_duration_min", alpha=0.35)
plt.title("Power vs Flight Duration")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_power_vs_flight_duration.png"), dpi=200)
plt.close()

# -------------------- Common preprocessing --------------------
def make_preprocessor(X):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    return pre, num, cat

# -------------------- SOH REGRESSION: leakage-controlled --------------------
soh_drop = [
    "soh_pct", "soc_pct", "capacity_fade_pct", "resistance_growth_pct",
    "battery_id", "uav_id", "battery_state_category"
]
X_soh = df.drop(columns=soh_drop, errors="ignore")
y_soh = df["soh_pct"]

Xtr, Xte, ytr, yte = train_test_split(X_soh, y_soh, test_size=0.20, random_state=42)
pre, _, _ = make_preprocessor(X_soh)

soh_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR()
}

soh_rows = []
best_soh = None
best_soh_pipe = None

for name, model in soh_models.items():
    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    r2 = r2_score(yte, pred)
    soh_rows.append([name, mae, rmse, r2])
    if best_soh is None or mae < best_soh:
        best_soh, best_soh_pipe = mae, pipe
        best_soh_pred = pred

soh_results = pd.DataFrame(soh_rows, columns=["Model","MAE","RMSE","R2"])
soh_results.to_csv(os.path.join(OUT, "soh_model_comparison.csv"), index=False)

plt.figure(figsize=(9,5))
sns.barplot(data=soh_results, x="Model", y="MAE")
plt.xticks(rotation=25)
plt.title("SOH Model Comparison - MAE")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_soh_mae_comparison.png"), dpi=200)
plt.close()

plt.figure(figsize=(9,5))
sns.barplot(data=soh_results, x="Model", y="RMSE")
plt.xticks(rotation=25)
plt.title("SOH Model Comparison - RMSE")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_soh_rmse_comparison.png"), dpi=200)
plt.close()

plt.figure(figsize=(9,5))
sns.barplot(data=soh_results, x="Model", y="R2")
plt.xticks(rotation=25)
plt.title("SOH Model Comparison - R²")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "11_soh_r2_comparison.png"), dpi=200)
plt.close()

plt.figure(figsize=(7,6))
plt.scatter(yte, best_soh_pred, alpha=0.4)
mn, mx = min(yte.min(), best_soh_pred.min()), max(yte.max(), best_soh_pred.max())
plt.plot([mn,mx], [mn,mx], linestyle="--")
plt.xlabel("Actual SOH (%)")
plt.ylabel("Predicted SOH (%)")
plt.title("Actual vs Predicted SOH")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "12_actual_vs_predicted_soh.png"), dpi=200)
plt.close()

joblib.dump(best_soh_pipe, os.path.join(MODEL_DIR, "soh_model.pkl"))

# -------------------- CLASSIFICATION: leakage-controlled --------------------
clf_drop = [
    "battery_state_category", "soc_pct", "soh_pct",
    "capacity_fade_pct", "resistance_growth_pct",
    "battery_id", "uav_id"
]
X_clf = df.drop(columns=clf_drop, errors="ignore")
y_clf = df["battery_state_category"]

Xtr, Xte, ytr, yte = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)
pre_c, _, _ = make_preprocessor(X_clf)

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
}

clf_rows = []
best_f1 = -1
best_clf_pipe = None
best_clf_pred = None

for name, model in clf_models.items():
    pipe = Pipeline([("preprocessor", pre_c), ("model", model)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    row = [
        name,
        accuracy_score(yte,pred),
        precision_score(yte,pred,average="weighted",zero_division=0),
        recall_score(yte,pred,average="weighted",zero_division=0),
        f1_score(yte,pred,average="weighted",zero_division=0),
        f1_score(yte,pred,average="macro",zero_division=0)
    ]
    clf_rows.append(row)
    if row[-1] > best_f1:
        best_f1 = row[-1]
        best_clf_pipe = pipe
        best_clf_pred = pred

clf_results = pd.DataFrame(
    clf_rows,
    columns=["Model","Accuracy","Precision","Recall","Weighted_F1","Macro_F1"]
)
clf_results.to_csv(os.path.join(OUT, "classification_model_comparison.csv"), index=False)

report = classification_report(yte, best_clf_pred, zero_division=0)
with open(os.path.join(OUT, "classification_report.txt"), "w") as f:
    f.write(report)

labels = sorted(y_clf.unique())
cm = confusion_matrix(yte, best_clf_pred, labels=labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Battery State Confusion Matrix")
plt.xticks(rotation=25)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "13_confusion_matrix.png"), dpi=200)
plt.close()

plt.figure(figsize=(9,5))
sns.barplot(data=clf_results, x="Model", y="Accuracy")
plt.xticks(rotation=25)
plt.title("Classification Accuracy Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "14_classification_accuracy.png"), dpi=200)
plt.close()

plt.figure(figsize=(9,5))
sns.barplot(data=clf_results, x="Model", y="Macro_F1")
plt.xticks(rotation=25)
plt.title("Classification Macro F1 Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "15_classification_macro_f1.png"), dpi=200)
plt.close()

joblib.dump(best_clf_pipe, os.path.join(MODEL_DIR, "battery_state_model.pkl"))

# Save summary
summary = {
    "dataset_shape": list(df.shape),
    "missing_values": int(df.isna().sum().sum()),
    "duplicates": int(df.duplicated().sum()),
    "best_soh_model_by_mae": soh_results.sort_values("MAE").iloc[0].to_dict(),
    "best_classifier_by_macro_f1": clf_results.sort_values("Macro_F1", ascending=False).iloc[0].to_dict(),
}
with open(os.path.join(OUT, "project_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSOH RESULTS")
print(soh_results.round(4).to_string(index=False))
print("\nCLASSIFICATION RESULTS")
print(clf_results.round(4).to_string(index=False))
print("\nBest SOH:", soh_results.sort_values("MAE").iloc[0]["Model"])
print("Best classifier:", clf_results.sort_values("Macro_F1", ascending=False).iloc[0]["Model"])
print("\nOutputs saved in:", OUT)
