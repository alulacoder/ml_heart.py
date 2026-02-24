import pandas as pd
import matplotlib.pyplot as plt
# Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load dataset
data = pd.read_csv("/Users/Alula/Desktop/Machine learning/heart.csv")

# Show basic info
print("First 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# Define features (X) and target (y)
x = data[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]

y = data["target"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = SVC(kernel='rbf')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(matrix, annot = True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nResults:")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))