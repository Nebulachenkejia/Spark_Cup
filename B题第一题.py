import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp

def optimize_stations(df_features, total_max, profit, time_limit=300):
    df_features['predicted_demand'] = 0.55 * df_features['borrow_count'] + 0.44 * df_features['return_count'] + 220 * df_features['casual_ratio'] + 110 * df_features['POI_score']
    predicted_demand = df_features['predicted_demand'].values

    df_features['diff_i'] = np.abs(df_features['borrow_count'] - df_features['return_count'])
    diff_i = df_features['diff_i'].values

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None, None
    solver.SetTimeLimit(time_limit * 1000)

    n = len(df_features)
    big_m = 10000

    x_i = [solver.IntVar(0, 1, f"x_{i}") for i in range(n)]
    capacity = [solver.IntVar(0, big_m, f"capacity_{i}") for i in range(n)]
    total_loss = solver.NumVar(0, solver.infinity(), "total_loss")
    loss_terms = []
    for i in range(n):
        surplus = solver.NumVar(0, solver.infinity(), f"surplus_{i}")
        ratio = capacity[i] * 31 / predicted_demand[i]
        solver.Add(surplus >= 1 - ratio)
        solver.Add(surplus >= ratio - 1)
        solver.Add(surplus <= big_m * x_i[i])
        solver.Add(surplus >= 0)
        loss_terms.append(surplus / predicted_demand[i])
    solver.Add(total_loss == solver.Sum(loss_terms) / n)
    solver.Minimize(total_loss)

    # 站点与容量关联约束
    min_capacity = 1
    for i in range(n):
        solver.Add(capacity[i] >= min_capacity * x_i[i])  # 选中的站点容量至少为min_capacity
        solver.Add(capacity[i] <= big_m * x_i[i])  # 未选中的站点容量为0

    total_capacity = solver.Sum(capacity)
    solver.Add(total_capacity <= total_max)

    lamuda = 0.5 #设置权重
    vacancy_loss = solver.Sum([x_i[i] * diff_i[i]*diff_i[i] for i in range(n)])
    scheduling_cost = solver.Sum([x_i[i] * diff_i[i] for i in range(n)])
    solver.Add((lamuda * scheduling_cost + (1 - lamuda) * vacancy_loss) <= profit)

    solver.Add(solver.Sum(x_i) >= 0.8 * n)

    #求解
    status = solver.Solve()
    if status == 0:
        x_i_opt = [int(x.solution_value()) for x in x_i]
        capacity_opt = [int(c.solution_value()) for c in capacity]  # 转为整数
        print(f"找到最优解，状态码：{status}")
        print(x_i_opt, capacity_opt)
        print(np.sum(x_i_opt), np.sum(capacity_opt))
        return x_i_opt, capacity_opt
    else:
        print("未找到最优解，状态码：", status)
        if status in (1, 4) and solver.num_solutions() > 0:
            x_i_opt = [int(x.solution_value()) for x in x_i]
            capacity_opt = [int(c.solution_value()) for c in capacity]
            print("返回当前找到的最佳解")
            print(x_i_opt, capacity_opt)
            return x_i_opt, capacity_opt
        return None, None


if __name__ == "__main__":
    df_features = pd.read_csv("demand_features.csv")
    profit = 100000
    # 计算总现有容量和约束上限
    df_features['capacity'] = 15    #容量需重新计算
    total_old = np.sum(df_features['capacity'])
    total_max = 14000
    x_i_opt, capacity_opt = optimize_stations(df_features, total_max, profit)
    df_features['status'] = 'close'  # 默认值
    df_features['capacity'] = 0  # 默认值

    if x_i_opt is not None and capacity_opt is not None:
        df_features['status'] = np.where(x_i_opt == 1, 'keep', 'close')
        df_features['capacity'] = capacity_opt

    selected_columns = ['station_id', 'status', 'capacity']
    required_columns = ['station_id']
    for col in required_columns:
        if col not in df_features.columns:
            raise ValueError(f"数据中缺少必要的列: {col}")
    df1 = df_features[selected_columns].copy()
    print(df1)
    df1.to_csv('result_145.csv', index=False)



"""
数据清洗刷  
df_clean=df.dropna(subset=['start_station_id', 'end_station_id'], how='any')
print(f"删去数量: {len(df) - len(df_clean)}")
#筛去异常使用记录（骑行时长异常）
使用3σ法筛去两侧数据
df_clean['started_at']=pd.to_datetime(df_clean['started_at'])
df_clean['ended_at']=pd.to_datetime(df_clean['ended_at'])
df_clean['using_duration']=(df_clean['ended_at']-df_clean['started_at']).dt.total_seconds() / 60
mean=df_clean['using_duration'].mean()
std=df_clean['using_duration'].std()
low=max(0, mean - 3 * std)
high=mean + 3 * std
df1=df_clean[(df_clean['using_duration'] >= low)&(df_clean['using_duration'] <= high)]
df1.to_csv("202503-capitalbikeshare-tripdata_1.csv", index=False)
print(f"舍去数量: {len(df_clean) - len(df1)}")
"""
