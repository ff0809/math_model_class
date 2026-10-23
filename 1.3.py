from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value

prob = LpProblem("Factory_Production_Plan", LpMaximize)

x111 = LpVariable("产品I（A1B1）", lowBound=0, cat='Integer') 
x112 = LpVariable("产品I（A1B2）", lowBound=0, cat='Integer')  
x113 = LpVariable("产品I（A1B3）", lowBound=0, cat='Integer') 
x121 = LpVariable("产品I（A2B1）", lowBound=0, cat='Integer') 
x122 = LpVariable("产品I（A2B2）", lowBound=0, cat='Integer') 
x123 = LpVariable("产品I（A2B3）", lowBound=0, cat='Integer')
x211 = LpVariable("产品II（A1B1）", lowBound=0, cat='Integer') 
x221 = LpVariable("产品II（A2B1）", lowBound=0, cat='Integer') 
x322 = LpVariable("产品III（A2B2）", lowBound=0, cat='Integer')

prob += (
    0.375 * x111 + 0.3026 * x112 + 0.4 * x113 +
    0.4003 * x121 + 0.3279 * x122 + 0.4253 * x123 +
    0.65 * x211 + 0.8611 * x221 +
    0.6844 * x322
), "Total_Profit"

prob += (
    5 * (x111 + x112 + x113) + 10 * x211 <= 6000
), "A1_Available_Time"

prob += (
    7 * (x121 + x122 + x123) + 9 * x221 + 12 * x322 <= 10000
), "A2_Available_Time"

prob += (
    6 * (x111 + x121) + 8 * (x211 + x221) <= 4000
), "B1_Available_Time"

prob += (
    4 * (x112 + x122) + 11 * x322 <= 7000
), "B2_Available_Time"

prob += (
    7 * (x113 + x123) <= 4000
), "B3_Available_Time"

prob.solve()

print(f"最大总利润：{value(prob.objective)} 元")
print("\n各产品各加工路径的最优产量：")
for v in prob.variables():
    print(f"{v.name}: {v.varValue:.0f} 件")

prod1_total = x111.varValue + x112.varValue + x113.varValue + x121.varValue + x122.varValue + x123.varValue
prod2_total = x211.varValue + x221.varValue
prod3_total = x322.varValue

print("\n各产品的总产量："  )
print(f"产品Ⅰ总产量：{prod1_total:.0f} 件")
print(f"产品Ⅱ总产量：{prod2_total:.0f} 件")
print(f"产品Ⅲ总产量：{prod3_total:.0f} 件")