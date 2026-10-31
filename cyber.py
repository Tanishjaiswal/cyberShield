

# ============================================================
# CYBERSHIELD
# Network Intrusion Detection Using Machine Learning
# ============================================================

# IMPORTING LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# PASTEL COLOR THEME

pastel_colors = [
    "#B8A9D9",  # Lavender
    "#E8A9B8",  # Dusty Pink
    "#A9C9E8",  # Pastel Blue
    "#A8D5BA",  # Mint Green
    "#F3C6A8"   # Peach
]

sns.set_theme(style="whitegrid")

plt.rcParams["figure.facecolor"] = "#FAF8FC"
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["axes.edgecolor"] = "#D8D2E5"
plt.rcParams["grid.color"] = "#EAE6F0"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 55)
print("CYBERSHIELD - NETWORK INTRUSION DETECTION")
print("=" * 55)

train = pd.read_csv(
    r"C:\Users\ACER\Downloads\UNSW_NB15_training-set.csv"
)

test = pd.read_csv(
    r"C:\Users\ACER\Downloads\UNSW_NB15_testing-set.csv"
)

print("\nTRAINING DATA SHAPE:", train.shape)
print("TESTING DATA SHAPE:", test.shape)

print("\nFIRST FIVE RECORDS:")
print(train.head())

print("\nDATASET INFORMATION:")
print("Total Rows:", train.shape[0])
print("Total Columns:", train.shape[1])

print("\nCOLUMN DATA TYPES:")
print(train.dtypes)


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\nMISSING VALUES BEFORE CLEANING:")
print(train.isnull().sum().sum())

print("\nDUPLICATE RECORDS:")
print(train.duplicated().sum())

# Replace infinite values

train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)

# Separate target and input features

# 0 = Normal Traffic
# 1 = Attack Traffic

X_train = train.drop(
    columns=["id", "label", "attack_cat"],
    errors="ignore"
).copy()

y_train = train["label"].astype(int)

X_test = test.drop(
    columns=["id", "label", "attack_cat"],
    errors="ignore"
).copy()

y_test = test["label"].astype(int)

# Identify numerical and categorical columns

numeric_columns = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

# Fill missing numerical values using training median

for col in numeric_columns:

    median_value = X_train[col].median()

    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

# Fill missing categorical values using training mode

for col in categorical_columns:

    mode_value = X_train[col].mode()[0]

    X_train[col] = X_train[col].fillna(mode_value)
    X_test[col] = X_test[col].fillna(mode_value)

print("\nMISSING VALUES AFTER CLEANING:")
print(X_train.isnull().sum().sum())

print("\nDATA CLEANING COMPLETED!")


# ============================================================
# 3. DATASET DISTRIBUTION
# ============================================================

print("\nNORMAL VS ATTACK DISTRIBUTION:")
print(train["label"].value_counts())

print("\nATTACK CATEGORY DISTRIBUTION:")
print(train["attack_cat"].value_counts())


# ============================================================
# 4. VISUALIZATION 1
# NORMAL VS ATTACK DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 4.5))

class_counts = y_train.value_counts().sort_index()

sns.barplot(
    x=["Normal", "Attack"],
    y=[
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ],
    hue=["Normal", "Attack"],
    palette=pastel_colors[:2],
    legend=False
)

plt.title("Normal vs Attack Traffic", fontsize=13)
plt.xlabel("Traffic Category")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()


# ============================================================
# 5. VISUALIZATION 2
# ATTACK CATEGORY DISTRIBUTION
# ============================================================

attack_counts = train["attack_cat"].value_counts()

plt.figure(figsize=(7, 4.5))

sns.barplot(
    x=attack_counts.values,
    y=attack_counts.index,
    hue=attack_counts.index,
    palette=sns.color_palette(
        pastel_colors,
        n_colors=len(attack_counts)
    ),
    legend=False
)

plt.title("Distribution of Attack Categories", fontsize=13)
plt.xlabel("Number of Records")
plt.ylabel("Attack Category")

plt.tight_layout()
plt.show()


# ============================================================
# 6. VISUALIZATION 3
# NETWORK CONNECTION DURATION
# ============================================================

# Calculate 99th percentile

duration_limit = train["dur"].quantile(0.99)

# Keep values within the limit for visualization only

duration_data = train[
    train["dur"] <= duration_limit
]

plt.figure(figsize=(7, 4.5))

sns.histplot(
    data=duration_data,
    x="dur",
    bins=35,
    color="#B8A9D9",
    edgecolor="white"
)

plt.title(
    "Distribution of Network Connection Duration",
    fontsize=13
)

plt.xlabel("Connection Duration")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()


# ============================================================
# 7. VISUALIZATION 4
# SOURCE PACKET DISTRIBUTION
# ============================================================

# Select 10,000 random records

boxplot_data = train.sample(
    n=min(10000, len(train)),
    random_state=42
).copy()

# Convert labels into readable names

boxplot_data["Traffic"] = boxplot_data["label"].map({
    0: "Normal",
    1: "Attack"
})

# Calculate 99th percentile

packet_limit = boxplot_data["spkts"].quantile(0.99)

# Filter only for visualization

boxplot_clean = boxplot_data[
    boxplot_data["spkts"] <= packet_limit
]

plt.figure(figsize=(7, 4.5))

sns.boxplot(
    data=boxplot_clean,
    x="Traffic",
    y="spkts",
    hue="Traffic",
    order=["Normal", "Attack"],
    palette={
        "Normal": "#B8A9D9",
        "Attack": "#E8A9B8"
    },
    showfliers=False,
    legend=False
)

plt.title(
    "Source Packets: Normal vs Attack Traffic",
    fontsize=13
)

plt.xlabel("Traffic Category")
plt.ylabel("Number of Source Packets")

plt.tight_layout()
plt.show()


# ============================================================
# 8. VISUALIZATION 5
# CORRELATION HEATMAP
# ============================================================

correlation_columns = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "label"
]

correlation_columns = [
    col for col in correlation_columns
    if col in train.columns
]

correlation = train[correlation_columns].corr()

plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap=sns.color_palette(
        ["#E8A9B8", "#FFFFFF", "#B8A9D9"],
        as_cmap=True
    ),
    linewidths=0.4,
    annot_kws={"size": 8},
    square=True,
    cbar_kws={"shrink": 0.75}
)

plt.title(
    "Correlation Between Network Features",
    fontsize=12
)

plt.xticks(fontsize=8, rotation=35, ha="right")
plt.yticks(fontsize=8, rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 9. CATEGORICAL ENCODING
# ============================================================

print("\nENCODING CATEGORICAL FEATURES...")

X_train = pd.get_dummies(
    X_train,
    columns=categorical_columns,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_columns,
    dtype=int
)

# Match testing columns with training columns

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

print("ENCODING COMPLETED!")

print("\nFINAL TRAINING SHAPE:", X_train.shape)
print("FINAL TESTING SHAPE:", X_test.shape)


# ============================================================
# 10. FEATURE SCALING
# ============================================================

print("\nFEATURE SCALING...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("FEATURE SCALING COMPLETED!")


# ============================================================
# 11. MODEL 1 - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 55)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 55)

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_scaled
)

print("\nLOGISTIC REGRESSION RESULTS")

print("Accuracy:",
      round(accuracy_score(y_test, logistic_pred), 4))

print("Precision:",
      round(precision_score(y_test, logistic_pred, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, logistic_pred, zero_division=0), 4))

print("F1 Score:",
      round(f1_score(y_test, logistic_pred, zero_division=0), 4))


# ============================================================
# 12. MODEL 2 - DECISION TREE
# ============================================================

print("\n" + "=" * 55)
print("MODEL 2: DECISION TREE")
print("=" * 55)

tree_model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

tree_model.fit(
    X_train_scaled,
    y_train
)

tree_pred = tree_model.predict(
    X_test_scaled
)

print("\nDECISION TREE RESULTS")

print("Accuracy:",
      round(accuracy_score(y_test, tree_pred), 4))

print("Precision:",
      round(precision_score(y_test, tree_pred, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, tree_pred, zero_division=0), 4))

print("F1 Score:",
      round(f1_score(y_test, tree_pred, zero_division=0), 4))


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],

    "Accuracy": [
        accuracy_score(y_test, logistic_pred),
        accuracy_score(y_test, tree_pred)
    ],

    "Precision": [
        precision_score(y_test, logistic_pred, zero_division=0),
        precision_score(y_test, tree_pred, zero_division=0)
    ],

    "Recall": [
        recall_score(y_test, logistic_pred, zero_division=0),
        recall_score(y_test, tree_pred, zero_division=0)
    ],

    "F1 Score": [
        f1_score(y_test, logistic_pred, zero_division=0),
        f1_score(y_test, tree_pred, zero_division=0)
    ]
})

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

print(results.round(4).to_string(index=False))


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

best_index = results["F1 Score"].idxmax()

if best_index == 0:

    best_name = "Logistic Regression"
    best_pred = logistic_pred

else:

    best_name = "Decision Tree"
    best_pred = tree_pred

print("\nBEST MODEL:", best_name)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_pred,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX:")
print(cm)

# Confusion matrix visualization

plt.figure(figsize=(6, 4.5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=sns.color_palette(
        ["#E8A9B8", "#FFFFFF", "#B8A9D9"],
        as_cmap=True
    ),
    linewidths=1,
    linecolor="white",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"],
    annot_kws={"size": 12}
)

plt.title(
    "Confusion Matrix - " + best_name,
    fontsize=13
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        y_test,
        best_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)


# ============================================================
# 17. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("CYBERSHIELD PROJECT COMPLETED")
print("=" * 55)

print("\nProject: Network Intrusion Detection")

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nAlgorithms Used:")
print("1. Logistic Regression")
print("2. Decision Tree")

print("\nBest Performing Model:", best_name)

print("\nFINAL RESULTS:")
print(results.round(4).to_string(index=False))

print("\nCONFUSION MATRIX:")
print(cm)

print("\nTHANK YOU!")
print("=" * 55)
