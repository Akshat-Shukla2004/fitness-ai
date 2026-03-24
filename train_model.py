import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error

df = pd.read_csv("data/final_calories.csv")
x=df[["Gender","Age","Height","Weight","Duration"]]
y=df["Calories"]

X_train,X_test,Y_train,Y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth = 10,
    random_state = 42,
    n_jobs=-1
)

model.fit(X_train,Y_train)
y_pred=model.predict(X_test)

r2=r2_score(Y_test,y_pred)
mae = mean_absolute_error(Y_test,y_pred)

print("Model Performance")
print("R2 score:",round(r2,4))
print("mean absolute error:",round(mae,2))

with open("model/calorie_model.pkl","wb") as f:
    pickle.dump((model,scaler),f)

print("Trained successfully")