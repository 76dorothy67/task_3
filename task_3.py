import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

result = pd.read_csv('internship_train.csv')
train_len = int(len(result)*0.8)

df_train = pd.read_csv("internship_train.csv")
X_train = df_train.iloc[:train_len + 1, 0:53].values
y_train = df_train.iloc[: train_len + 1, -1:].values
X_test = df_train.iloc [train_len + 1 : , 0:53].values
y_test = df_train.iloc[train_len + 1:, -1:].values

model = DecisionTreeRegressor().fit(X_train, y_train)
y_train_pred = model.predict(X_train)
r2_train = model.score(X_train, y_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_test_pred = model.predict(X_test)
r2_test = model.score(X_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

df_test = pd.read_csv("internship_hidden_test.csv")
X = df_test.iloc[:, :].values
y_pred = model.predict(X)

df_test['target'] = y_pred

# зберігаємо DataFrame у CSV файл
df_test.to_csv('internship_hidden_test.csv', index=False)



