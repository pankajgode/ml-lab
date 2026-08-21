import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Salary_Prediction_Dataset.csv")

# Features (X) and Target (y)
X = df[['Years of Experience']]
y = df['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nActual Salary:")
print(y_test.values)

print("\nPredicted Salary:")
print(y_pred)