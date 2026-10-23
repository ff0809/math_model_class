import numpy as np
from scipy import stats

group1 = np.array([98, 93, 103, 92, 110])
group2 = np.array([100, 108, 118, 99, 111])
group3 = np.array([129, 140, 108, 105, 116])
data = [group1, group2, group3]

k = len(data)  # 组数k=3
n = [len(g) for g in data]  # 每组样本量n=[5,5,5]
N = sum(n)  # 总样本量N=15

group_means = [np.mean(g) for g in data]
total_mean = np.mean(np.concatenate(data))

S_A = sum(nj * (mj - total_mean)**2 for nj, mj in zip(n, group_means))
S_E = sum(sum((x - mj)**2 for x in g) for g, mj in zip(data, group_means))
S_T = sum((x - total_mean)**2 for g in data for x in g)

df_A = k - 1
df_E = N - k
MS_A = S_A / df_A
MS_E = S_E / df_E
F = MS_A / MS_E

F_critical = 4.6

print("\n平方和分解：")
print(f"组间平方和 S_A = {S_A:.4f}，自由度 df_A = {df_A}")
print(f"组内平方和 S_E = {S_E:.4f}，自由度 df_E = {df_E}")
print(f"总平方和 S_T = {S_T:.4f}（验证：S_A+S_E = {S_A+S_E:.4f}）")
print("\n均方与F统计量：")
print(f"组间均方 MS_A = {MS_A:.4f}")
print(f"组内均方 MS_E = {MS_E:.4f}")
print(f"F统计量 F = {F:.4f}")
print(f"临界值 F_0.05({df_A},{df_E}) = {F_critical}")

if F > F_critical:
    print("\n结论：拒绝原假设H0，认为3个组的理赔额均值有显著差异")
else:
    print("\n结论：不拒绝原假设H0，认为3个组的理赔额均值无显著差异")

f_stat, p_val = stats.f_oneway(group1, group2, group3)
print(f"\nF统计量 = {f_stat:.4f}，p值 = {p_val:.4f}")
if p_val < 0.05:
    print("验证：拒绝H0，均值有显著差异")
else:
    print("验证：不拒绝H0，均值无显著差异")