import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

years = np.arange(1969, 1984)
y = np.array([3.78, 4.19, 4.83, 5.46, 6.71, 7.99, 8.60, 9.24,
              9.67, 9.87, 10.49, 10.92, 10.93, 12.39, 12.59])
t = np.arange(1, 16)

# 三次多项式
coef3 = np.polyfit(t, y, 3)
p3 = np.poly1d(coef3)
print(f"三次多项式: y = {coef3[0]:.4f}*t^3 + {coef3[1]:.4f}*t^2 + {coef3[2]:.4f}*t + {coef3[3]:.4f}")

# 对数多项式
ln_t = np.log(t)
X_log = np.column_stack([ln_t, ln_t**2])
reg_log = LinearRegression().fit(X_log, y)
a_log, b_log, c_log = reg_log.intercept_, reg_log.coef_[0], reg_log.coef_[1]
print(f"对数多项式: y = {a_log:.4f} + {b_log:.4f}*ln(t) + {c_log:.4f}*(ln(t))^2")

# Weibull
def weibull(t, L, y0, alpha, beta):
    return L - (L - y0) * np.exp(-(t/alpha)**beta)

def weibull_init_stable(t, y):
    dy1 = y[-1] - y[-2]
    dy2 = y[-2] - y[-3]
    if dy2 > 0:
        L = y[-1] + dy1**2 / dy2
    else:
        L = y[-1] * 1.05
    L = max(L, y[-1] + 0.5)
    y0 = y[0]
    r = (L - y) / (L - y0)
    r = np.clip(r, 1e-10, 1 - 1e-10)
    lhs = np.log(-np.log(r))
    valid = np.isfinite(lhs)
    if np.sum(valid) < 2:
        return L, y0, 8.0, 2.0
    coeff = np.polyfit(np.log(t[valid]), lhs[valid], 1)
    beta = coeff[0]
    alpha = np.exp(-coeff[1] / beta)
    return L, y0, alpha, beta

p0_wei = weibull_init_stable(t, y)
lower_bounds_wei = [y[-1]+0.1, -np.inf, 1e-5, 0.1]
upper_bounds_wei = [20, np.inf, 100, 10]
popt_wei, _ = curve_fit(weibull, t, y, p0=p0_wei, bounds=(lower_bounds_wei, upper_bounds_wei), maxfev=5000)
Lw, y0w, alphaw, betaw = popt_wei
print(f"Weibull: y = {Lw:.4f} - ({Lw:.4f}-{y0w:.4f}) * exp(-(t/{alphaw:.4f})^{betaw:.4f})")

# Richards
def richards(t, L, k, t0, nu):
    return L * (1 + nu * np.exp(-k * (t - t0)))**(-1/nu)

def richards_init_stable(t, y):
    L = weibull_init_stable(t, y)[0]
    dy1 = np.diff(y)
    dy2 = np.diff(dy1)
    t0 = t[np.argmax(np.abs(dy2)) + 1]
    z = (L - y) / y
    z = np.clip(z, 1e-10, 1e10)
    lhs = np.log(z)
    valid = np.isfinite(lhs)
    if np.sum(valid) < 2:
        return L, 0.2, t0, 1.0
    coeff = np.polyfit(t[valid], lhs[valid], 1)
    k = -coeff[0]
    if k <= 0:
        k = 0.2
    nu = 1.0
    return L, k, t0, nu

p0_rich = richards_init_stable(t, y)
lower_bounds_rich = [y[-1]+0.1, 1e-5, 1, 0.1]
upper_bounds_rich = [20, 5, 15, 10]
popt_rich, _ = curve_fit(richards, t, y, p0=p0_rich, bounds=(lower_bounds_rich, upper_bounds_rich), maxfev=5000)
Lr, kr, t0r, nur = popt_rich
print(f"Richards: y = {Lr:.4f} * (1 + {nur:.4f} * exp(-{kr:.4f}*(t-{t0r:.4f})))^(-1/{nur:.4f})")

def r2_mse(actual, pred):
    r2 = 1 - np.sum((actual - pred)**2) / np.sum((actual - np.mean(actual))**2)
    mse = np.mean((actual - pred)**2)
    return r2, mse

y_logpoly = a_log + b_log*ln_t + c_log*ln_t**2
y_weibull = weibull(t, *popt_wei)
y_rich = richards(t, *popt_rich)

print("\n=== 拟合精度 ===")
print(f"三次多项式: R²={r2_mse(y, p3(t))[0]:.4f}, MSE={r2_mse(y, p3(t))[1]:.4f}")
print(f"对数多项式: R²={r2_mse(y, y_logpoly)[0]:.4f}, MSE={r2_mse(y, y_logpoly)[1]:.4f}")
print(f"Weibull:    R²={r2_mse(y, y_weibull)[0]:.4f}, MSE={r2_mse(y, y_weibull)[1]:.4f}")
print(f"Richards:   R²={r2_mse(y, y_rich)[0]:.4f}, MSE={r2_mse(y, y_rich)[1]:.4f}")

t17, t22 = 17, 22
print("\n=== 预测值 (亿千克) ===")
print(f"三次多项式: 1985年 {p3(t17):.2f}, 1990年 {p3(t22):.2f}")
print(f"对数多项式: 1985年 {a_log + b_log*np.log(t17) + c_log*(np.log(t17))**2:.2f}, 1990年 {a_log + b_log*np.log(t22) + c_log*(np.log(t22))**2:.2f}")
print(f"Weibull:    1985年 {weibull(t17, *popt_wei):.2f}, 1990年 {weibull(t22, *popt_wei):.2f}")
print(f"Richards:   1985年 {richards(t17, *popt_rich):.2f}, 1990年 {richards(t22, *popt_rich):.2f}")

plt.figure(figsize=(10,6))
plt.scatter(years, y, label='实际值', s=50, color='black')
plt.plot(years, y_logpoly, 'r-', linewidth=2, label='对数多项式')
plt.plot(years, p3(t), 'b--', linewidth=2, label='三次多项式')
plt.plot(years, y_weibull, 'g-.', linewidth=2, label='Weibull')
plt.plot(years, y_rich, 'm:', linewidth=2, label='Richards')
plt.xlabel('年份')
plt.ylabel('产量（亿千克）')
plt.title('粮食产量曲线拟合比较')
plt.legend()
plt.grid(True)
plt.savefig('粮食产量拟合对比.png', dpi=300, bbox_inches='tight')
plt.show()