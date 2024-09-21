from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('house_price.csv')

bins = [0, 200000, 500000, float('inf')]
labels = ['Low', 'Medium', 'High']

data['Price_Range'] = pd.cut(data['Median_Price'], bins=bins, labels=labels, right=False)
cleaned_data = data.dropna(subset=['Median_Price', 'Price_Range'])
cleaned_data.loc[:, 'Small_Area'] = cleaned_data['Small_Area'].fillna(cleaned_data['Small_Area'].mode()[0])

X = cleaned_data[['Transfer_Year', 'Transaction_Count']]
y = cleaned_data['Price_Range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

accuracy = rf_model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

y_pred = rf_model.predict(X_test)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(12, 7))
bars = plt.barh(feature_names, importances, color=['skyblue', 'lightgreen'])
plt.title('Feature Importance')
plt.xlabel('Importance (%)')
plt.ylabel('Feature')

# Add value annotations to each bar
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center')

plt.show()

# Count occurrences of each label
label_counts = y.value_counts()

# Plot a pie chart for label distribution
plt.figure(figsize=(12, 7))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Distribution of Price Ranges')
plt.show()

# Create a crosstab to analyze the relationship between Transfer_Year and Price Range
crosstab = pd.crosstab(X['Transfer_Year'], y)

# Plot stacked bar chart
crosstab.plot(kind='bar', stacked=True, figsize=(12, 7), color=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Transfer Year vs. Price Range')
plt.ylabel('Count')
plt.show()

