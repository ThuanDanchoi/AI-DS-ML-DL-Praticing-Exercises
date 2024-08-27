import pandas as pd
import matplotlip as plt
import numpy as np
import seaborn as sns


from sklearn.model_selection import train_test_split

data = pd.read_csv('weatherHistory.csv')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


print(data.isnull().sum())

