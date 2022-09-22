import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Getting Data
df = pd.read_csv("kc_house_data.csv")

print(df.head())
print()
print(df.describe())
print()
print(df.info())
print()

# Checking null values
print(df.isnull().sum())

# EDA
plt.figure(figsize=(10,6))
sns.histplot(df["price"], kde = True)

sns.countplot(x=df["bedrooms"], palette="rainbow")

# Correlation
print("Correlation of Price:")
print(df.corr()["price"].sort_values())
print()

plt.figure(figsize=(10,5))
sns.scatterplot(x="price", y="sqft_living", data=df)

plt.figure(figsize=(10,6))
sns.boxplot(x="bedrooms", y="price", data=df)

plt.figure(figsize=(12,8))
sns.scatterplot(x="price", y="long", data=df)

plt.figure(figsize=(12,8))
sns.scatterplot(x="price", y="lat", data=df)

# Resembles location on map of house pricing
plt.figure(figsize=(12,8))
sns.scatterplot(x="long", y="lat", data=df, hue="price", palette="rainbow")

# Excluding top 1% houses as they are too costly, causing error in the model
non_top_1_perc = df.sort_values("price",ascending=False).iloc[round(len(df)*0.01):]

# Better location graph based on house pricing
plt.figure(figsize=(12,8))
sns.scatterplot(x="long", y="lat", data=non_top_1_perc, hue="price", palette="rainbow")

sns.boxplot(x="waterfront", y="price", data=df)

df = df.drop("id",axis=1)   # Not a useful column

# Converting date column to date datatype
df["date"] = pd.to_datetime(df["date"])

# Adding year & month column
df["year"] = df["date"].apply(lambda date: date.year)
df["month"] = df["date"].apply(lambda date: date.month)

print("New Dataset:")
print(df.head())
print()

plt.figure(figsize=(10,6))
sns.boxplot(x="month", y="price", data=df)

df.groupby("year").mean()["price"].plot()

# No need of date column as we have month and year
df = df.drop("date", axis=1)
df = df.drop("zipcode", axis=1)

# Splitting Data
from sklearn.model_selection import train_test_split
X = df.drop("price", axis=1).values
y = df["price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42)

# Scaling Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
print(X_train.shape)

model = Sequential()

model.add(Dense(19, activation="relu")) # no. of columns is 19
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(x=X_train, y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128,
          epochs=400)

losses = pd.DataFrame(model.history.history)
losses.plot()

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)
evs = explained_variance_score(y_test, predictions)
print("Explained Variance Score:", evs)

plt.figure(figsize=(12,6))
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, "r")

# Testing for single data point
single_house = df.drop("price", axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,19))

print(model.predict(single_house))
print()
print(df.head(1)["price"])



















