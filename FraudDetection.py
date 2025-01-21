import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\asaik\OneDrive\Documents\ML Projects\FinancialFraudDetectionProject\creditcard.csv")

print(data.head())

print(data.info())

#Data Preprocessing.
6
#Checking for the missing values.

print(data.isnull().sum())

#Standardizing the 'Amount' feature.

scaler = StandardScaler()

data[['Amount']] = scaler.fit_transform(data[['Amount']])

#Displaying the first few features after standardization.

print(data.head())

# Define the features for the model (excluding 'Class' for unsupervised learning)
features = data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 
                 'V27', 'V28', 'Amount']]

# Initialize the Isolation Forest model
isolation_forest = IsolationForest(n_estimators=1000, contamination=0.01, random_state=42)

# Fit the model on the features
isolation_forest.fit(features)

# Predict anomalies
data['anomaly'] = isolation_forest.predict(features)

# Convert anomalies from -1/1 to 0/1 for easier interpretation
data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})

# Display the results
print(data[['Amount', 'anomaly']].head())

# For demonstration, we will use a small sample to evaluate
sample_data = data.sample(n=1000, random_state=42)  # Sample 1000 transactions

# Create a confusion matrix
conf_matrix = confusion_matrix(sample_data['Class'], sample_data['anomaly'])
print("Confusion Matrix:\n", conf_matrix)

# Generate a classification report
class_report = classification_report(sample_data['Class'], sample_data['anomaly'])
print("Classification Report:\n", class_report)

# Scatter plot to visualize anomalies
plt.figure(figsize=(10, 6))
plt.scatter(data[data['anomaly'] == 0].index, data[data['anomaly'] == 0]['Amount'], 
            color='blue', label='Normal Transactions', alpha=0.5)
plt.scatter(data[data['anomaly'] == 1].index, data[data['anomaly'] == 1]['Amount'], 
            color='red', label='Anomalous Transactions', alpha=0.5)
plt.title('Anomaly Detection in Transactions')
plt.xlabel('Transaction Index')
plt.ylabel('Transaction Amount (Standardized)')
plt.legend()
plt.show()