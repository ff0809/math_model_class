import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data = np.array([
    15.0, 15.8, 15.2, 15.1, 15.9, 14.7, 14.8, 15.5, 15.6, 15.3,
    15.1, 15.3, 15.0, 15.6, 15.7, 14.8, 14.5, 14.2, 14.9, 14.9,
    15.2, 15.0, 15.3, 15.6, 15.1, 14.9, 14.2, 14.6, 15.8, 15.2,
    15.9, 15.2, 15.0, 14.9, 14.8, 14.5, 15.1, 15.5, 15.5, 15.1,
    15.1, 15.0, 15.3, 14.7, 14.5, 15.5, 15.0, 14.7, 14.6, 14.2
])
n = len(data)  
mu = 15.0780  
sigma = 0.4325 
alpha = 0.05 

bins = [14.1, 14.4, 14.7, 15.0, 15.3, 15.6, 15.9, 16.2]
f_obs, _ = np.histogram(data, bins=bins)
k = len(f_obs)

p = []
for i in range(k):
    a = bins[i]
    b = bins[i+1]
    z_a = (a - mu) / sigma
    z_b = (b - mu) / sigma
    p_i = stats.norm.cdf(z_b) - stats.norm.cdf(z_a)
    p.append(p_i)
np_i = n * np.array(p) 

chi_square = np.sum((f_obs - np_i) ** 2 / np_i)

r = 2 
df = k - r - 1  
chi_critical = stats.chi2.ppf(1 - alpha, df) 

print(f"\n卡方统计量 χ² = {chi_square:.4f}")
print(f"卡方临界值 χ²_{alpha}({df}) = {chi_critical:.4f}")

if chi_square < chi_critical:
    print("\n结论：不拒绝原假设H0，认为滚珠直径服从正态分布N(15.0780, 0.4325²)")
else:
    print("\n结论：拒绝原假设H0，认为滚珠直径不服从该正态分布")

