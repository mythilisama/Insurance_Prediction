import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

os.makedirs("../artifacts", exist_ok=True)

# load processed data
X_train = pd.read_csv("../data/processed/X_train.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")

# PRINT TABLES
print("\nTraining Features:\n")
print(X_train.to_string())

print("\nTraining Target:\n")
print(y_train.to_string())

model = LinearRegression()

model.fit(X_train, y_train)

with open("../artifacts/model.pkl","wb") as f:
    pickle.dump(model,f)

print("\nModel trained and saved successfully")