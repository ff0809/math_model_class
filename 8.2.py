import numpy as np

y = np.array([696.9, 607.7, 604.0, 604.5, 638.2, 670.3, 732.8, 770.5,
              737.3, 801.5, 858.0, 929.2, 1023.3, 1106.7, 1163.6, 1271.1,
              1339.4, 1432.8, 1558.6, 1800.0, 2140.0, 2350.0, 2570.0])

def triple_exp_smoothing(y, alpha):
    n = len(y)
    S1 = np.zeros(n)
    S2 = np.zeros(n)
    S3 = np.zeros(n)

    S1[0] = y[0]
    S2[0] = y[0]
    S3[0] = y[0]

    for t in range(1, n):
        S1[t] = alpha * y[t] + (1 - alpha) * S1[t-1]
        S2[t] = alpha * S1[t] + (1 - alpha) * S2[t-1]
        S3[t] = alpha * S2[t] + (1 - alpha) * S3[t-1]

    a = 3*S1 - 3*S2 + S3
    denom = 2 * (1 - alpha)**2
    b = (alpha / denom) * ((6-5*alpha)*S1 - 2*(5-4*alpha)*S2 + (4-3*alpha)*S3)
    c = (alpha**2 / denom) * (S1 - 2*S2 + S3)

    fit = np.zeros(n)
    fit[0] = y[0]
    for t in range(1, n):
        fit[t] = a[t-1] + b[t-1]*1 + c[t-1]*1  

    return a, b, c, fit

# 网格搜索找最优 alpha
alphas = np.arange(0.01, 0.99, 0.005)
best_alpha = 0.3
best_mse = 1e12

for alpha in alphas:
    _, _, _, fit = triple_exp_smoothing(y, alpha)
    mse = np.mean((y[-5:] - fit[-5:]) ** 2)
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha

print(f"最优 α = {best_alpha:.3f}")
print(f"1978-1982 MSE = {best_mse:.1f}\n")

a, b, c, fit = triple_exp_smoothing(y, best_alpha)

print(f"1982 模型参数：a={a[-1]:.1f}, b={b[-1]:.1f}, c={c[-1]:.1f}")
print(f"1983 预测：{a[-1] + b[-1]*1 + c[-1]*1:.1f} 亿元")
print(f"1985 预测：{a[-1] + b[-1]*3 + c[-1]*9:.1f} 亿元\n")

print("1978-1982 拟合效果：")
for idx, i in enumerate(range(-5, 0)):
    year = 1978 + idx
    print(f"{year}年 实际={y[i]:6.1f} 拟合={fit[i]:6.1f} 误差={y[i]-fit[i]:6.1f}")