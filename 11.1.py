import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

data = {
    'x1': [0, 0, 0, 0, 0, 0, 0.17, 0.17, 0.17, 0.17, 0.21, 0],
    'x2': [0.23, 0.1, 0, 0.49, 0, 0.62, 0.27, 0.19, 0.21, 0.15, 0.36, 0],
    'x3': [0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.12, 0],
    'x4': [0, 0, 0.1, 0, 0.62, 0, 0.38, 0.38, 0.38, 0.38, 0.25, 0.55],
    'x5': [0, 0.12, 0.12, 0.12, 0.12, 0, 0, 0.02, 0, 0.02, 0, 0],
    'x6': [0.74, 0.74, 0.74, 0.37, 0.18, 0.37, 0, 0.06, 0.06, 0.1, 0, 0.37],
    'x7': [0.03, 0.04, 0.04, 0.02, 0.08, 0.01, 0.08, 0.08, 0.08, 0.08, 0.06, 0.08],
    'y':  [98.7, 97.8, 96.6, 92.0, 86.6, 91.2, 81.9, 83.1, 82.4, 83.2, 81.4, 88.1]
}
df = pd.DataFrame(data)
X = df[['x1','x2','x3','x4','x5','x6','x7']].values
y = df['y'].values.reshape(-1,1)

loo = LeaveOneOut()
print("=== 交叉验证结果 ===")
best_n = 1
best_r2 = -np.inf
max_components = min(5, X.shape[0]-1)   # 最多5个成分，且不超过样本数-1

for n in range(1, max_components+1):
    pls = PLSRegression(n_components=n)
    y_pred = cross_val_predict(pls, X, y, cv=loo)
    r2 = r2_score(y, y_pred)
    print(f"成分个数 = {n}，交叉验证 R² = {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_n = n

print(f"\n最佳主成分个数 = {best_n}\n")

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_std = scaler_x.fit_transform(X)
y_std = scaler_y.fit_transform(y).ravel()

pls_final = PLSRegression(n_components=best_n)
pls_final.fit(X_std, y_std)

coef_std = pls_final.coef_.ravel()

x_mean = scaler_x.mean_
x_scale = scaler_x.scale_  
y_mean = scaler_y.mean_[0]
y_scale = scaler_y.scale_[0]

raw_coef = y_scale * coef_std / x_scale
raw_intercept = y_mean - np.sum(raw_coef * x_mean)

print("=== 最终偏最小二乘回归方程 ===")
print(f"截距 β0 = {raw_intercept:.4f}")
for i, name in enumerate(['x1','x2','x3','x4','x5','x6','x7']):
    print(f"系数 β{i+1} ({name}) = {raw_coef[i]:.4f}")

# 计算训练集的拟合R²（原始尺度下）
y_pred_std = pls_final.predict(X_std)
y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1,1)).ravel()
train_r2 = r2_score(y, y_pred)
print(f"\n训练集拟合 R² = {train_r2:.4f}")

print(f"sklearn直接提供的训练R² = {pls_final.score(X_std, y_std):.4f}")