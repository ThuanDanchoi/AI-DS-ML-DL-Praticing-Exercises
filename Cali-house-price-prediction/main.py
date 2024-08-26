import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('housing.csv')

# Separate numerical and categorical data
numerical_data = data.select_dtypes(include=[np.number])
categorical_data = data.select_dtypes(exclude=[np.number])

# Identify columns with and without missing data
cols_nan = numerical_data.columns[numerical_data.isna().any()].tolist()
cols_no_nan = numerical_data.columns.difference(cols_nan).values

# Impute missing values using KNeighborsRegressor
for col in cols_nan:
    imp_test = numerical_data[numerical_data[col].isna()]
    imp_train = numerical_data.dropna()
    model = KNeighborsRegressor(n_neighbors=5)
    knr = model.fit(imp_train[cols_no_nan], imp_train[col])
    numerical_data.loc[data[col].isna(), col] = knr.predict(imp_test[cols_no_nan])

# Recombine numerical and categorical data
final_data = pd.concat([numerical_data, categorical_data], axis=1)

# Encode categorical data
label_encoder = LabelEncoder()
final_data['ocean_proximity'] = label_encoder.fit_transform(final_data['ocean_proximity'])

# Separate features and target variable
target = 'median_house_value'
X = final_data.drop(target, axis=1)
y = final_data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using regression metrics
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred, squared=False))
print("R-squared (R²):", r2_score(y_test, y_pred))


