# ==================== 导入库 ====================
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd

z_raw = np.array([
    [1350,1370,1390,1400,1410,960,940,880,800,690,570,430,290,210,150],
    [1370,1390,1410,1430,1440,1140,1110,1050,950,820,690,540,380,300,210],
    [1380,1410,1430,1450,1470,1320,1280,1200,1080,940,780,620,460,370,350],
    [1420,1430,1450,1480,1500,1550,1510,1430,1300,1200,980,850,750,550,500],
    [1430,1450,1460,1500,1550,1600,1550,1600,1600,1600,1550,1500,1500,1550,1550],
    [950,1190,1370,1500,1200,1100,1550,1600,1550,1380,1070,900,1050,1150,1200],
    [910,1090,1270,1500,1200,1100,1350,1450,1200,1150,1010,880,1000,1050,1100],
    [880,1060,1230,1390,1500,1500,1400,900,1100,1060,950,870,900,936,950],
    [830,980,1180,1320,1450,1420,400,1300,700,900,850,810,380,780,750],
    [740,880,1080,1130,1250,1280,1230,1040,900,500,700,780,750,650,550],
    [650,760,880,970,1020,1050,1020,830,800,700,300,500,550,480,350],
    [510,620,730,800,850,870,850,780,720,650,500,200,300,350,320],
    [370,470,550,600,670,690,670,620,580,450,400,300,100,150,250]
])

x_ori = np.arange(0, 5601, 400)   
y_ori = np.arange(4800, -1, -400)
X_ori, Y_ori = np.meshgrid(x_ori, y_ori)

points = np.c_[X_ori.ravel(), Y_ori.ravel()]
vals = z_raw.ravel()

x_new = np.arange(0, 5601, 50) 
y_new = np.arange(0, 4801, 50)
X_new, Y_new = np.meshgrid(x_new, y_new)

Z_new = griddata(points, vals, (X_new, Y_new), method='linear')

flat_x = X_new.ravel()
flat_y = Y_new.ravel()
flat_z = Z_new.ravel()

df = pd.DataFrame({
    "x(m)": flat_x,
    "y(m)": flat_y,
    "高程z(m)": flat_z
})
df.to_csv("50m间隔_全高程数据.csv", index=False, encoding="utf-8-sig")
print("\n 全部50m网格高程已保存到：50m间隔_全高程数据.csv")

plt.figure(figsize=(12,7))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.contour(X_new, Y_new, Z_new, levels=20, cmap="terrain", linewidths=1)
plt.contourf(X_new, Y_new, Z_new, levels=20, cmap="terrain", alpha=0.7)
plt.colorbar(label="高程(m)")
plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.title("50m间隔地形等高线图")
plt.grid(alpha=0.3)
plt.show()