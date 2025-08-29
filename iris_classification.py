# Классификация цветов ириса (Iris dataset) с sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Загружаем данные
iris = load_iris()
X = iris.data
y = iris.target

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создаём модель
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка
acc = accuracy_score(y_test, y_pred)
print(f"Точность модели: {acc:.2f}")
