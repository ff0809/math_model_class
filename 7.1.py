import numpy as np
from scipy import stats

x = np.array([1050, 1100, 1120, 1250, 1280])
n = len(x) 
confidence = 0.90  

mean = np.mean(x)
std = np.std(x, ddof=1)

alpha = 1 - confidence
t_critical = stats.t.ppf(1 - alpha/2, df=n-1)

margin_error = t_critical * std / np.sqrt(n)
confidence_interval = (mean - margin_error, mean + margin_error)

print(f"\n灯泡寿命平均值置信水平为{confidence}的置信区间为：")
print(f"({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}) h")