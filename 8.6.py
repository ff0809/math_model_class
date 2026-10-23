import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

data = [
    7.5, 8.9, 11.1, 13.4, 15.5, 15.7, 15.6, 16.7, 18.0, 17.4, 17.9, 18.8,
    17.6, 17.0, 16.1, 15.7, 15.9, 17.9, 20.3, 20.4, 20.2, 20.5, 20.9, 20.9,
    21.1, 21.4, 18.2, 20.1, 21.4, 21.3, 21.9, 21.3, 20.4, 20.4, 20.7, 20.7,
    20.9, 23.0, 24.9, 26.5, 25.6, 26.1, 27.0, 27.2, 28.1, 28.0, 29.1, 28.3,
    25.7, 24.5, 24.4, 25.5, 27.0, 28.7, 29.1, 29.0, 29.6, 31.2, 30.6, 29.8,
    27.6, 27.7, 29.0, 30.3, 31.0, 32.1, 33.5, 33.2, 33.2, 33.8, 35.5, 36.8,
    37.9, 39.0, 41.0, 41.6, 43.7, 44.4, 46.6, 48.3, 50.2, 52.1, 54.0, 56.0,
    53.9, 55.6, 55.4, 56.2, 57.9, 57.3, 58.8, 60.4, 63.1, 83.5, 64.8, 65.7,
    64.8, 65.6, 67.2, 62.1
]
y = pd.Series(data)
log_y = np.log(y)

# 网格搜索参数范围
p_range = range(0, 3)
q_range = range(0, 3)
P_range = range(0, 2)
Q_range = range(0, 2)
d = 1
D = 1
m = 4

best_aic = np.inf
best_order = None
best_seasonal_order = None

print("开始网格搜索最优 SARIMA 模型（对数序列，d=1, D=1, m=4）...")
for p in p_range:
    for q in q_range:
        for P in P_range:
            for Q in Q_range:
                try:
                    model = SARIMAX(log_y, order=(p, d, q), seasonal_order=(P, D, Q, m),
                                    simple_differencing=False)
                    fit = model.fit(disp=False)
                    aic = fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_seasonal_order = (P, D, Q, m)
                        print(f"找到更优模型: order={best_order}, seasonal={best_seasonal_order}, AIC={best_aic:.2f}")
                except:
                    continue

print(f"\n最优模型: SARIMA{best_order}{best_seasonal_order}, AIC={best_aic:.2f}")

final_model = SARIMAX(log_y, order=best_order, seasonal_order=best_seasonal_order, simple_differencing=False)
final_fit = final_model.fit(disp=False)
print(final_fit.summary())

# 残差诊断
resid = final_fit.resid
lb_test = acorr_ljungbox(resid, lags=[12], return_df=True)
print(f"Ljung-Box p值 (lag=12): {lb_test['lb_pvalue'].iloc[0]:.4f}")

forecast_log = final_fit.forecast(steps=8)
forecast = np.exp(forecast_log)

pred = final_fit.get_forecast(steps=8)
conf_int_log = pred.conf_int(alpha=0.05)
conf_int = np.exp(conf_int_log)

forecast_index = pd.date_range(start='1971-01-01', periods=8, freq='QE')
print("\n预测结果（1971-1972各季度）：")
for i, date in enumerate(forecast_index):
    print(f"{date.year}年Q{date.quarter}: {forecast.iloc[i]:.1f} 亿美元  95% CI: [{conf_int.iloc[i,0]:.1f}, {conf_int.iloc[i,1]:.1f}]")

plt.figure(figsize=(12,6))
plt.plot(y, 'b-', alpha=0.7, label='历史数据')
plt.plot(range(len(y), len(y)+8), forecast, 'ro-', lw=2, label='预测值')
plt.fill_between(range(len(y), len(y)+8), conf_int.iloc[:,0], conf_int.iloc[:,1],
                 color='pink', alpha=0.3, label='95%置信区间')
plt.xlabel('时间（季度序号）')
plt.ylabel('耐用品支出（亿美元）')
plt.title(f'自动最优 SARIMA{best_order}{best_seasonal_order} 模型预测')
plt.legend()
plt.grid(True)
plt.show()
