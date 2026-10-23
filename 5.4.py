import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd

E = 30.24  
D = 57 * E  
S = np.pi * (D/2)**2
rho = 1.0

raw_data = [
    (0, 3175), (3316, 3110), (6635, 3054), (10619, 2994), (13937, 2947),
    (17921, 2892), (21240, 2850), (25223, 2795), (28543, 2752), (32284, 2697),
    # 泵水时段 1: 32284 到 39435
    (39435, 3550), (43318, 3445), (44636, 3350), (49953, 3260), (53936, 3167),
    (57254, 3087), (60574, 3012), (64554, 2927), (68535, 2842), (71854, 2767),
    (75021, 2697),
    # 泵水时段 2: 75021 到 85968
    (85968, 3475), (89953, 3397), (93270, 3340)
]

t_pts = np.array([x[0] for x in raw_data])
h_pts = np.array([x[1] for x in raw_data]) * 0.01 * E 
m_pts = rho * S * h_pts 

pump_idx = [(9, 10), (20, 21)] 

f_samples_t = []
f_samples_v = []

for i in range(len(t_pts)-1):
    is_pumping = any(i == start for start, end in pump_idx)
    
    if not is_pumping:
        dt = t_pts[i+1] - t_pts[i]
        dm = m_pts[i+1] - m_pts[i]
        avg_f = -dm / dt
        f_samples_t.append((t_pts[i] + t_pts[i+1]) / 2)
        f_samples_v.append(avg_f)

spline_f = UnivariateSpline(f_samples_t, f_samples_v, s=0.5, k=3)

def get_f(t):
    val = float(spline_f(t))
    return max(val, 0) 

t_s, t_e = t_pts[9], t_pts[10]
m_s, m_e = m_pts[9], m_pts[10]

t_fine = np.linspace(t_s, t_e, 100)
f_integral = np.trapz([get_f(t) for t in t_fine], t_fine)

Q = (m_e - m_s + f_integral) / (t_e - t_s)
print(f"水泵恒定灌水速度 Q = {Q:.2f} g/s")

t_all = np.linspace(0, 93270, 1000)
f_all = [get_f(t) for t in t_all]

t_list = np.arange(0, 93271, 100)
df = pd.DataFrame({
    "时间t(s)": t_list,
    "流出流量f(t)(g/s)": [round(get_f(t), 2) for t in t_list]
})
df.to_excel("水箱流量结果.xlsx", index=False)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(12, 6))

plt.plot(t_all, f_all, 'r-', linewidth=2, label="拟合流出流量 $f(t)$")
plt.scatter(f_samples_t, f_samples_v, color='blue', s=20, alpha=0.5, label="实测流量样本")

for start, end in pump_idx:
    plt.axvspan(t_pts[start], t_pts[end], color='orange', alpha=0.2, label="水泵工作时段" if start==9 else "")

plt.title("水箱流量 $f(t)$ 随时间变化曲线")
plt.xlabel("时间 t (s)")
plt.ylabel("流量 (g/s)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()