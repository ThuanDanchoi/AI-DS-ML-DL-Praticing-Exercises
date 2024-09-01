import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv('air_quality_health.csv')

# Columns to scale
scale_columns = [
    'AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3',
    'Temperature', 'Humidity', 'WindSpeed',
    'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions',
    'HealthImpactScore'
]

scaler = StandardScaler()

data[scale_columns] = scaler.fit_transform(data[scale_columns])

X = data[scale_columns]
y = data['HealthImpactClass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)

print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_prediction))


cm = confusion_matrix(y_test, y_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
