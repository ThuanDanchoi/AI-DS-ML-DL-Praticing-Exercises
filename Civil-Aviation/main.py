import matplotlib.pyplot as mlp
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv('Aviation_CityPairs_2019.csv')


X = data[['City1', 'City2', 'Month', 'Aircraft_Trips', 'Passenger_Load_Factor', 'Distance_GC_(km)', 'ASKs', 'Seats']]
y = data['Passenger_Trips']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Month', 'Aircraft_Trips', 'Passenger_Load_Factor', 'Distance_GC_(km)', 'ASKs', 'Seats']),
        ('cat', OneHotEncoder(), ['City1', 'City2'])
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])


X_transformed = pipeline.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_prediction = model.predict(X_test)


mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


mlp.scatter(y_test, y_prediction)
mlp.xlabel('Actual Passenger Trips')
mlp.ylabel('Predicted Passenger Trips')
mlp.title('Actual vs Predicted Passenger Trips')
mlp.show()

