import pandas as pd

df = pd.read_csv("data/calories.csv")
df = df[["Gender","Age","Height","Weight","Duration","Calories"]]
df["Gender"] = df["Gender"].map({"male":0,"female":1})
df.columns = ["Gender","Age","Height","Weight","Duration","Calories"]
df.to_csv("data/final_calories.csv",index=False)
print("calories.csv cleaned")