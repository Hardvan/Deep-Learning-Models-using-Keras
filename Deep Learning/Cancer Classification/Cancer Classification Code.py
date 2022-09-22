import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Getting Data
df = pd.read_csv("cancer_classification.csv")

print(df.head())
print()
print(df.describe())
print()
print(df.info())
print()

# EDA
sns.countplot(x="benign_0__mal_1", data=df)

df.corr()["benign_0__mal_1"][:-1].sort_values().plot(kind="bar")

plt.figure(figsize=(12,12))
sns.heatmap(df.corr())

# Splitting Data
from sklearn.model_selection import train_test_split
X = df.drop("benign_0__mal_1", axis=1).values
y = df["benign_0__mal_1"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

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

model.add(Dense(30, activation="relu")) # no. of columns is 30
model.add(Dense(15, activation="relu"))

model.add(Dense(1, activation="sigmoid"))   # Binary Classification

model.compile(optimizer="adam", loss="binary_crossentropy")

model.fit(x=X_train, y=y_train,
          validation_data=(X_test, y_test),
          epochs=600)

# Overfitting
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.title("Overfitting")

# Model with Early Stopping
model = Sequential()

model.add(Dense(30, activation="relu"))
model.add(Dense(15, activation="relu"))

# Binary Classification
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy")

# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
model.fit(x=X_train, y=y_train,
          validation_data=(X_test, y_test),
          epochs=600,
          callbacks=[early_stop])

# Just right fit
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.title("With Early Stopping only")

# Model with Dropout & Early Stopping
from tensorflow.keras.layers import Dropout
model = Sequential()

model.add(Dense(30, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(15, activation="relu"))
model.add(Dropout(0.5))

# Binary Classification
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy")

model.fit(x=X_train, y=y_train,
          validation_data=(X_test, y_test),
          epochs=600,
          callbacks=[early_stop])

model_loss2 = pd.DataFrame(model.history.history)
model_loss2.plot()
plt.title("With Dropout and Early Stopping")

predictions = (model.predict(X_test) > 0.5)*1

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))
print()















