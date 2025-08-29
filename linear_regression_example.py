# Простой пример линейной регрессии на Python

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Данные: X — часы занятий, y — баллы
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Создаём модель
model = LinearRegression()
model.fit(X, y)

# Предсказание
y_pred = model.predict(X)

# Вывод коэффициентов
print("Коэффициент:", model.coef_[0])
print("Смещение:", model.intercept_)

# График
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel("Часы занятий")
plt.ylabel("Баллы")
plt.title("Линейная регрессия")
plt.show()
