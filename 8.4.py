import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

years = np.arange(1952, 1998)
data = [119, 142, 144, 150, 165, 168, 200, 216, 218, 185, 173, 181, 208, 240,
        254, 235, 222, 243, 275, 288, 292, 309, 310, 327, 316, 339, 379, 417,
        460, 489, 525, 580, 682, 853, 956, 1104, 1355, 1512, 1634, 1879, 2287,
        2939, 3923, 4854, 5576, 6079]

y = pd.Series(data)

model = ARIMA(y, order=(2,1,1), trend='t')
fit = model.fit()
print(fit.summary())

phi1 = fit.params['ar.L1']
phi2 = fit.params['ar.L2']
theta1 = fit.params['ma.L1']
c = fit.params['x1']
print(f"\n参数估计值：")
print(f"phi1 = {phi1:.4f}")
print(f"phi2 = {phi2:.4f}")
print(f"theta1 = {theta1:.4f}")
print(f"趋势项 c = {c:.4f}")

# 残差白噪声检验
resid = fit.resid
lb_test = acorr_ljungbox(resid, lags=[12], return_df=True)
print(f"\nLjung-Box检验（滞后12期）p值 = {lb_test['lb_pvalue'].iloc[0]:.4f}")

# 10步预测
forecast_result = fit.get_forecast(steps=10)
forecast_values = forecast_result.predicted_mean.values
conf_int = forecast_result.conf_int(alpha=0.05)
conf_lower = conf_int.iloc[:, 0].values
conf_upper = conf_int.iloc[:, 1].values

forecast_years = np.arange(1998, 2008)
print("\n10步预测结果（1998-2007年）：")
for i, year in enumerate(forecast_years):
    print(f"{year}: 预测值 {forecast_values[i]:.0f} 元/人 (95% CI: [{conf_lower[i]:.0f}, {conf_upper[i]:.0f}])")

plt.figure(figsize=(12,6))
plt.plot(years, y, 'b-o', label='历史数据')
plt.plot(forecast_years, forecast_values, 'r-s', label='预测值')
plt.fill_between(forecast_years, conf_lower, conf_upper,
                 color='pink', alpha=0.3, label='95%置信区间')
plt.xlabel('年份')
plt.ylabel('人均GDP（元/人）')
plt.title('我国人均GDP的ARIMA(2,1,1)模型拟合与预测')
plt.legend()
plt.grid(True)
plt.show()