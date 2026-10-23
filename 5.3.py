import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([15.3, 20.5, 27.4, 36.6, 49.1, 65.6, 87.87, 117.6])
n = len(x) 

Y = np.log(y)

sum_x = np.sum(x)
sum_Y = np.sum(Y)
sum_x2 = np.sum(x**2)
sum_xY = np.sum(x * Y)

B = (n * sum_xY - sum_x * sum_Y) / (n * sum_x2 - sum_x**2)
A = (sum_Y - B * sum_x) / n

a_linear = np.exp(A)
b_linear = B

y_linear = a_linear * np.exp(b_linear * x)
r2_linear = r2_score(y, y_linear)

print(f"a = {a_linear:.4f}, b = {b_linear:.4f}")
print(f"拟合公式：y = {a_linear:.4f} * e^({b_linear:.4f}x)")
print(f"拟合优度 R² = {r2_linear:.6f}")

x_plot = np.linspace(0, 9, 100)
y_linear_plot = a_linear * np.exp(b_linear * x_plot)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(x, y, color='red', s=60, zorder=5)
plt.plot(x_plot, y_linear_plot, color='blue', linewidth=2)

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('最小二乘法拟合', fontsize=14, pad=20)
plt.grid(linestyle='--', alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
