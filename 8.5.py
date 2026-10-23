import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

data = [269, 321, 585, 871, 1475, 2821, 3928, 5943, 4950, 2577, 523, 98,
        184, 279, 409, 2285, 2685, 3409, 1824, 409, 151, 45, 68, 213,
        546, 1033, 2129, 2536, 957, 361, 377, 225, 360, 731, 1638, 2725,
        2871, 2119, 684, 299, 236, 245, 552, 1623, 3311, 6721, 4254, 687,
        255, 473, 358, 784, 1594, 1676, 2251, 1426, 756, 299, 201, 229,
        469, 736, 2042, 2811, 4431, 2511, 389, 73, 39, 49, 59, 188, 377,
        1292, 4031, 3495, 537, 105, 153, 387, 758, 1307, 3465, 6991, 6313,
        3794, 1836, 345, 382, 808, 1388, 2713, 3800, 309, 2985, 3790, 674,
        71, 80, 108, 229, 399, 1132, 2432, 3575, 2935, 1537, 529, 485,
        662, 1000, 1520, 2657, 3396]

y = np.array(data)
log_y = np.log10(y)

max_lag = 11
model = AutoReg(log_y, lags=max_lag, trend='c')
model_fit = model.fit()

print("模型拟合摘要")
print(model_fit.summary())

resid = model_fit.resid
lb_test = acorr_ljungbox(resid, lags=[12], return_df=True)
print(f"\nLjung-Box 残差检验（滞后12期） p值 = {lb_test['lb_pvalue'].iloc[0]:.4f}")

pred_log = model_fit.predict(start=len(y), end=len(y)+1)
pred_original = 10 ** pred_log

print(f"\n模型使用的自回归阶数: {max_lag}")
print(f"第115年（未来第1年）预测数量: {pred_original[0]:.0f} 只")
print(f"第116年（未来第2年）预测数量: {pred_original[1]:.0f} 只")

plt.figure(figsize=(12,6))
plt.plot(range(1, len(y)+1), y, 'b-', alpha=0.7, label='历史真实数据')
plt.plot([115, 116], pred_original, 'ro--', ms=8, lw=2, label='Log-AR预测值')
fitted_values = 10 ** model_fit.fittedvalues
plt.plot(range(max_lag+1, len(y)+1), fitted_values, 'g--', alpha=0.5, label='历史数据模型拟合')
plt.xlabel('年份序号')
plt.ylabel('山猫数量 (只)')
plt.title('基于对数自回归模型(AR11)的山猫数量预测')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()