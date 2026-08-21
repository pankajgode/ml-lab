import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Create synthetic dataset
np.random.seed(0)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 2. Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# 3. Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Compute evaluation metrics
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1-Score: {f1:.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 6. Visualize predictions
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_test[:, 0],
    y=X_test[:, 1],
    hue=y_pred,
    palette="coolwarm",
    edgecolor="k"
)

plt.title("Logistic Regression Predictions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Predicted Class")
plt.show()  