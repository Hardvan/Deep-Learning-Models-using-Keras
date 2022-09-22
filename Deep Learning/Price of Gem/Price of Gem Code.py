import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Getting Data
df = pd.read_csv("fake_reg.csv")

print(df.head())
print()
print(df.describe())
print()
print(df.info())
print()

# EDA
sns.pairplot(df)

# Splitting Data
from sklearn.model_selection import train_test_split
X = df[["feature1","feature2"]].values
y = df["price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = 42)

print(X_train.shape)
print(X_test.shape)
print()

# Data Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))

# Output Node
model.add(Dense(1))

model.compile(optimizer="rmsprop", loss="mse")

model.fit(x=X_train, y=y_train, epochs=250, verbose=0)

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

# Predictions & Evaluation
print(model.evaluate(X_test, y_test, verbose=0))  # Returns MSE
print(model.evaluate(X_train, y_train, verbose=0))
print()

test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(250,))

pred_df = pd.DataFrame(y_test, columns=["Test True Y"])
pred_df = pd.concat([pred_df,test_predictions], axis=1)
pred_df.columns = ["Test True Y", "Model Predictions"]
print(pred_df.head())
print()

sns.scatterplot(x="Test True Y", y="Model Predictions", data=pred_df)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(pred_df["Test True Y"], pred_df["Model Predictions"])
print("MAE:", mae)
mse = mean_squared_error(pred_df["Test True Y"], pred_df["Model Predictions"])
print("MSE:", mse)
rmse = mean_squared_error(pred_df["Test True Y"], pred_df["Model Predictions"])**0.5
print("RMSE:", rmse)
print()

# Trial Data
new_gem = [[998, 1000]]
new_gem = scaler.transform(new_gem)
model.predict(new_gem, verbose=0)

# Saving Model
from tensorflow.keras.models import load_model
model.save("my_gem_model.h5")

later_model = load_model("my_gem_model.h5")

print("Predicted Price for feature1 = {}, feature2 = {} is:".format(new_gem[0][0], new_gem[0][1]))

print(later_model.predict(new_gem))
















