from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target

model = DecisionTreeClassifier()
model.fit(X, y)

print("Prediction:", iris.target_names[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]])
