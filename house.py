import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"/Users/kumkumsingh335/Downloads/House Price Prediction Dataset.csv")

df = df.drop("Id", axis=1)

df["Garage"] = df["Garage"].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df, columns=["Location", "Condition"], drop_first=True)

df = df.fillna(df.mean())

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(X_train, y_train, epochs=50, validation_split=0.2)

loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)

predictions = model.predict(X_test[:5])
print("Predictions:", predictions.flatten())
print("Actual:", y_test[:5].values)