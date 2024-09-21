import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('house_price.csv')

# Define bins and labels for price ranges
bins = [0, 200000, 500000, float('inf')]
labels = ['Low', 'Medium', 'High']
data['Price_Range'] = pd.cut(data['Median_Price'], bins=bins, labels=labels, right=False)

# Data preprocessing: Drop rows with missing values in essential columns
cleaned_data = data.dropna(subset=['Median_Price', 'Price_Range'])
cleaned_data.loc[:, 'Small_Area'] = cleaned_data['Small_Area'].fillna(cleaned_data['Small_Area'].mode()[0])

# Clustering Data
X_cluster = cleaned_data[['Transfer_Year', 'Transaction_Count']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# -------------------------------------
# Visualization 1: Scatter plot as K-Means Clustering
# -------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add K-Means cluster labels to the dataset
cleaned_data['KMeans_Labels'] = kmeans.labels_

# Evaluate K-Means clusters using silhouette score
silhouette_avg_kmeans = silhouette_score(X_scaled, kmeans.labels_)
print(f'K-Means Silhouette Score: {silhouette_avg_kmeans:.2f}')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=cleaned_data['Transfer_Year'], y=cleaned_data['Transaction_Count'], hue=cleaned_data['KMeans_Labels'], palette='deep')
plt.title('K-Means Clustering of House Data')
plt.xlabel('Transfer Year')
plt.ylabel('Transaction Count')
plt.show()

# -------------------------------------
# Visualization 2: Scatter plot as DBSCAN Clustering
# -------------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add DBSCAN cluster labels to the dataset
cleaned_data['DBSCAN_Labels'] = dbscan_labels

# Evaluate DBSCAN clusters using silhouette score (ignore noise points with label -1)
valid_labels = dbscan_labels != -1  # Exclude noise points (labelled as -1)
if valid_labels.sum() > 0:
    silhouette_avg_dbscan = silhouette_score(X_scaled[valid_labels], dbscan_labels[valid_labels])
    print(f'DBSCAN Silhouette Score: {silhouette_avg_dbscan:.2f}')
else:
    print('No valid clusters found for silhouette score calculation.')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=cleaned_data['Transfer_Year'], y=cleaned_data['Transaction_Count'], hue=cleaned_data['DBSCAN_Labels'], palette='deep', legend='full')
plt.title('DBSCAN Clustering of House Data')
plt.xlabel('Transfer Year')
plt.ylabel('Transaction Count')
plt.show()

# -------------------------------------
# Compare K-Means and DBSCAN cluster labels with Price Range
# -------------------------------------

comparison_kmeans = pd.crosstab(cleaned_data['KMeans_Labels'], cleaned_data['Price_Range'])
print("K-Means vs Price Range:")
print(comparison_kmeans)

comparison_dbscan = pd.crosstab(cleaned_data['DBSCAN_Labels'], cleaned_data['Price_Range'])
print("DBSCAN vs Price Range:")
print(comparison_dbscan)

