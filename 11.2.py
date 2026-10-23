import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

data = {
    'x1': [46,52,46,49,42,48,49,48,45,48,49,47,41,52,45,49,53,47,48,49,48,42,54,53,42,46,45,50,42,48,42,48,49,47,49,44,52,48],
    'x2': [55,55,69,50,55,61,60,63,55,64,52,62,51,55,52,57,65,57,60,55,69,57,64,63,71,66,56,60,51,50,52,67,74,55,74,52,66,68],
    'x3': [126,95,107,105,90,106,100,122,105,120,100,100,101,125,94,110,112,95,120,113,128,122,155,120,138,120,91,120,126,115,140,105,151,113,120,110,130,100],
    'x4': [51,42,38,48,46,43,49,52,48,38,42,34,53,43,50,47,47,47,47,41,42,46,51,42,44,45,29,42,50,41,48,39,49,40,53,37,47,45],
    'x5': [75.0,81.2,98.0,97.6,66.5,78.0,90.6,56.0,76.0,60.2,53.4,61.2,62.4,86.3,51.4,72.3,90.4,72.3,86.4,84.1,47.9,54.2,71.4,56.6,65.2,62.2,66.2,56.6,50.0,52.9,56.3,69.2,54.2,71.4,54.5,54.9,45.9,53.6],
    'x6': [25,18,18,16,2,25,15,17,15,20,6,10,5,5,20,19,15,9,12,15,20,15,19,8,17,22,18,8,13,6,15,23,20,19,22,14,14,23],
    'x7': [72,50,74,60,68,58,60,68,61,62,42,62,60,62,65,45,75,64,62,60,63,63,61,53,55,68,51,57,57,39,60,60,58,64,59,57,45,70],
    'y1': [6.8,7.2,6.8,6.8,7.2,7.0,7.0,7.0,6.8,7.0,7.4,7.2,8.0,6.8,7.6,7.0,6.6,6.6,6.8,7.0,7.0,7.2,6.9,7.5,7.0,7.4,7.9,6.8,7.7,7.4,6.9,7.6,7.0,7.6,6.9,7.5,6.8,7.2],
    'y2': [489,464,430,362,453,405,420,466,415,413,404,427,372,496,394,446,420,447,398,398,485,400,511,430,487,470,380,460,398,415,470,450,500,410,500,400,505,522],
    'y3': [27,30,32,26,23,29,21,28,24,28,23,25,25,30,24,30,30,25,28,27,30,28,33,29,29,28,26,32,27,28,27,28,30,29,33,29,28,28],
    'y4': [8,5,9,6,11,7,10,2,6,7,6,7,3,10,3,11,12,4,11,4,7,6,12,4,9,7,5,5,2,6,11,10,12,7,21,2,11,9],
    'y5': [360,348,386,331,391,389,379,362,386,398,400,407,409,350,399,337,357,447,381,387,350,388,298,353,370,360,358,348,383,314,348,326,330,331,348,421,355,352]
}

df = pd.DataFrame(data)
X = df[['x1','x2','x3','x4','x5','x6','x7']].values
Y = df[['y1','y2','y3','y4','y5']].values

loo = LeaveOneOut()
max_comp = min(6, X.shape[0]-1)
avg_r2_cv = []
for n in range(1, max_comp+1):
    pls = PLSRegression(n_components=n)
    y_pred = cross_val_predict(pls, X, Y, cv=loo)
    r2_list = [r2_score(Y[:, i], y_pred[:, i]) for i in range(Y.shape[1])]
    avg_r2 = np.mean(r2_list)
    avg_r2_cv.append(avg_r2)
    print(f"n={n}, mean_cv_R2={avg_r2:.4f}, each={np.round(r2_list,4)}")

best_n = np.argmax(avg_r2_cv) + 1
print(f"\n最佳成分数 = {best_n}\n")

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_std = scaler_x.fit_transform(X)
Y_std = scaler_y.fit_transform(Y)

pls_final = PLSRegression(n_components=best_n)
pls_final.fit(X_std, Y_std)

coef_std = pls_final.coef_       
x_mean = scaler_x.mean_
x_scale = scaler_x.scale_
y_mean = scaler_y.mean_
y_scale = scaler_y.scale_

raw_coef = np.zeros_like(coef_std)
for i in range(Y.shape[1]):
    raw_coef[i, :] = y_scale[i] * coef_std[i, :] / x_scale

raw_intercept = y_mean - np.sum(raw_coef * x_mean, axis=1)

Y_pred_std = pls_final.predict(X_std)
Y_pred = scaler_y.inverse_transform(Y_pred_std)
train_r2 = [r2_score(Y[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]

print("=== 最终回归方程 ===\n")
for i, name in enumerate(['y1','y2','y3','y4','y5']):
    eq = f"{name} = {raw_intercept[i]:.4f}"
    for j, var in enumerate(['x1','x2','x3','x4','x5','x6','x7']):
        eq += f" {raw_coef[i,j]:+.4f}*{var}"
    print(eq)
    print(f"    R²_train = {train_r2[i]:.4f}\n")
print(f"平均训练R² = {np.mean(train_r2):.4f}")