import pandas as pd

data = pd.read_csv("data science/heart.csv")
print(data.head())
print(data.info())

# input: everything except target  output: target
x = data[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
y = data["target"]

