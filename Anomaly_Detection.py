# Import necessary libraries
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_excel("Machine_Data_Set.xlsx")

#drop the string features
df.drop('Power Status', axis = 1,inplace = True)
df.drop('Machine Status', axis = 1, inplace = True)
df.drop('Timestamp', axis = 1, inplace = True)

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Create an Isolation Forest model
clf = IsolationForest(contamination=0.15)  # Contamination is the proportion of outliers in the data

# Fit the model and predict anomalies
clf.fit(scaled_data)
predictions = clf.predict(scaled_data)

# Anomalies are labeled as -1, normal points are labeled as 1
anomaly_indices = np.where(predictions == -1)[0]


for i in df.columns:
    plt.plot(df.index, df[i], c = 'r')
    plt.scatter(anomaly_indices, df[i].loc[anomaly_indices], c = 'b', marker = '*')
    plt.title(i)
    plt.show()
