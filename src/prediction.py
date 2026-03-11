import pandas as pd
import pickle

# load trained model
with open("../artifacts/model.pkl","rb") as f:
    model = pickle.load(f)

# load scaler
with open("../artifacts/scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# new customer data
new_customer = pd.DataFrame({
    "Age":[35],
    "Annual_Income_LPA":[8.5],
    "Policy_Term_Years":[25],
    "Sum_Assured_Lakhs":[80]
})

print("\nNew Customer Data:\n")
print(new_customer.to_string(index=False))

# scale data
scaled_data = scaler.transform(new_customer)

# prediction
prediction = model.predict(scaled_data)

print("\nPredicted Annual Premium (Thousands):")
print(prediction[0])