import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
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

# Classification data
X = cleaned_data[['Transfer_Year', 'Transaction_Count']]
y = cleaned_data['Price_Range']

# Initialize LabelEncoder and fit on the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
accuracy = rf_model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Predictions
y_pred = rf_model.predict(X_test)

# Compute metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

# -------------------------------------
# Visualization 1: Crosstab and stacked bar chart
# -------------------------------------
crosstab = pd.crosstab(X['Transfer_Year'], y)
crosstab.plot(kind='bar', stacked=True, figsize=(12, 7), color=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Transfer Year vs. Price Range')
plt.ylabel('Count')
plt.show()

# -------------------------------------
# Visualization 2: Confusion Matrix
# -------------------------------------
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# -------------------------------------
# Visualization 3: Classification Report as heatmap
# -------------------------------------
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(12, 7))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues')
plt.title('Classification Report')
plt.show()

# -------------------------------------
# Visualization 4: Multiclass ROC Curve
# -------------------------------------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
classifier = OneVsRestClassifier(RandomForestClassifier(random_state=42))
y_score = classifier.fit(X_train, label_binarize(y_train, classes=[0, 1, 2])).predict_proba(X_test)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure(figsize=(12, 7))
for i in range(y_test_bin.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multiclass Classification')
plt.legend(loc="lower right")
plt.show()
