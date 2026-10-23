from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value

prob = LpProblem("Cargo_Plane_Loading", LpMaximize)

x = [[LpVariable(f"x{i+1}{j+1}", lowBound=0, cat='Continuous') for j in range(3)] for i in range(4)]

profit = [3100, 3800, 3500, 2850]
prob += lpSum(profit[i] * lpSum(x[i][j] for j in range(3)) for i in range(4)), "Total_Profit"

prob += lpSum(x[i][0] for i in range(4)) <= 10, "Front_Weight_Limit"
prob += lpSum(x[i][1] for i in range(4)) <= 16, "Middle_Weight_Limit"
prob += lpSum(x[i][2] for i in range(4)) <= 8, "Rear_Weight_Limit"

volume = [480, 650, 580, 390]
prob += lpSum(volume[i] * x[i][0] for i in range(4)) <= 6800, "Front_Volume_Limit"
prob += lpSum(volume[i] * x[i][1] for i in range(4)) <= 8700, "Middle_Volume_Limit"
prob += lpSum(volume[i] * x[i][2] for i in range(4)) <= 5300, "Rear_Volume_Limit"

max_cargo = [18, 15, 23, 12]
for i in range(4):
    prob += lpSum(x[i][j] for j in range(3)) <= max_cargo[i], f"Cargo{i+1}_Total_Limit"

front_weight = lpSum(x[i][0] for i in range(4))
middle_weight = lpSum(x[i][1] for i in range(4))
rear_weight = lpSum(x[i][2] for i in range(4))
prob += 16 * front_weight == 10 * middle_weight, "Balance_Front_Middle"
prob += 8 * middle_weight == 16 * rear_weight, "Balance_Middle_Rear"

prob.solve()

print(f"最大总利润：{value(prob.objective):.2f} 元")

print("\n各货物在各货舱的最优装载量：")
cargo_names = ["货物1", "货物2", "货物3", "货物4"]
hold_names = ["前舱", "中舱", "后舱"]
for i in range(4):
    for j in range(3):
        print(f"{cargo_names[i]}在{hold_names[j]}：{value(x[i][j]):.4f} t")
