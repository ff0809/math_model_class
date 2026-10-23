import numpy as np
import pandas as pd

years = np.arange(1974, 1982)
y = np.array([80.8, 94.0, 88.4, 101.5, 110.3, 121.5, 134.7, 142.7])
t = np.arange(1, 9)  # 时间序号

# 1. 趋势移动平均法 (二次移动平均, N=3)
def moving_avg(x, n):
    return np.convolve(x, np.ones(n)/n, mode='valid')

# 一次移动平均
M1 = moving_avg(y, 3)   # 对应 t=3,4,5,6,7,8
# 二次移动平均
M2 = moving_avg(M1, 3)  # 对应 t=5,6,7,8

a_ma = 2 * M1[2:] - M2
b_ma = M1[2:] - M2      # 2/(N-1)=1

print("=== 趋势移动平均法 ===")
print("M1:", M1)
print("M2:", M2)
print("a_t (t=5~8):", a_ma)
print("b_t (t=5~8):", b_ma)
print("预测模型 (t=8): y = {:.3f} + {:.3f} * T".format(a_ma[-1], b_ma[-1]))

y_fit_ma = a_ma
y_actual_ma = y[4:8]
mse_ma = np.mean((y_actual_ma - y_fit_ma)**2)
print("1978-1981 拟合误差 MSE = {:.4f}\n".format(mse_ma))

# 2. 直线指数平滑法 (布朗二次指数平滑)
def brown_exp_smoothing(y, alpha, S0):
    n = len(y)
    S1 = np.zeros(n)
    S2 = np.zeros(n)
    a = np.zeros(n)
    b = np.zeros(n)
    S1_prev = S0
    S2_prev = S0
    for i in range(n):
        S1[i] = alpha * y[i] + (1-alpha) * S1_prev
        S2[i] = alpha * S1[i] + (1-alpha) * S2_prev
        a[i] = 2 * S1[i] - S2[i]
        b[i] = alpha/(1-alpha) * (S1[i] - S2[i])
        S1_prev, S2_prev = S1[i], S2[i]
    return a, b

S0 = (y[0] + y[1] + y[2]) / 3
print("初始平滑值 S0 = {:.1f}".format(S0))

# α=0.3
a03, b03 = brown_exp_smoothing(y, 0.3, S0)
print("\n=== α=0.3 指数平滑 ===")
print("a_t (t=1~8):", np.round(a03, 3))
print("预测模型 (t=8): y = {:.3f} + {:.3f} * T".format(a03[-1], b03[-1]))
y_fit03 = a03
y_actual = y
mse03 = np.mean((y_actual[4:8] - y_fit03[4:8])**2)
print("1978-1981 拟合误差 MSE = {:.4f}".format(mse03))

# α=0.6
a06, b06 = brown_exp_smoothing(y, 0.6, S0)
print("\n=== α=0.6 指数平滑 ===")
print("a_t (t=1~8):", np.round(a06, 3))
print("预测模型 (t=8): y = {:.3f} + {:.3f} * T".format(a06[-1], b06[-1]))
y_fit06 = a06
mse06 = np.mean((y_actual[4:8] - y_fit06[4:8])**2)
print("1978-1981 拟合误差 MSE = {:.4f}".format(mse06))

print("\n=== 最优模型预测 (α=0.6) ===")
y_1982 = a06[-1] + b06[-1] * 1
y_1985 = a06[-1] + b06[-1] * 4
print("1982年预测产量: {:.1f} 亿米".format(y_1982))
print("1985年预测产量: {:.1f} 亿米".format(y_1985))