import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('mock_data_capstone.csv')

X = df.drop('PH',axis=1)
Y= df['PH']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.5,shuffle=True)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(criterion= 'poisson', min_samples_split= .5, splitter= 'best')
dt.fit(X_train,Y_train)

prediction=dt.predict(X_test)


sns.displot(Y_train, color="#B2FF66")
plt.savefig('PH.png')

