import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'Annual_Income': [15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 60, 65, 70, 75, 80],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 60, 65, 70, 75, 20, 30, 40, 50, 60]
}

df = pd.DataFrame(data)

X = df[['Annual_Income', 'Spending_Score']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using K-Means')
plt.show()

print(df)
