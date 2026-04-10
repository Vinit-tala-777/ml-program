import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Make predictions
x_test = np.array([6, 7, 8]).reshape(-1, 1)
x_test_poly = poly.transform(x_test)
y_pred = model.predict(x_test_poly)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Predictions for', x_test.flatten(), '=', y_pred)

# Plot the original data and the polynomial regression curve
x_plot = np.linspace(x.min(), x_test.max(), 100).reshape(-1, 1)
x_plot_poly = poly.transform(x_plot)
y_plot = model.predict(x_plot_poly)

plt.scatter(x, y, color='blue', label='Training data')
plt.plot(x_plot, y_plot, color='red', label='Polynomial fit')
plt.scatter(x_test, y_pred, color='green', marker='x', label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
