from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('mock_data_capstone.csv')

X = df.drop('Temp', axis=1)
Y = df['Temp']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, shuffle=True)
dt = DecisionTreeRegressor(
    criterion='poisson', min_samples_split=.5, splitter='best')
dt.fit(X_train, Y_train)

prediction = dt.predict(X_test)


sns.displot(Y_train, color="red")
plt.savefig('Temp.png')
