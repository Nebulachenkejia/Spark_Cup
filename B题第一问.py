import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp

def optimize_stations(df_features, total_max, profit, time_limit=300):  #求解函数
    df_features['predicted_demand'] = 0.55 * df_features['borrow_count'] + 0.44 * df_features['return_count'] + 220 * df_features['casual_ratio'] + 110 * df_features['POI_score']
    predicted_demand = df_features['predicted_demand'].values
    df_features['diff_i'] = np.abs(df_features['borrow_count'] - df_features['return_count'])
    diff_i = df_features['diff_i'].values

    # 创建求解器并设置时间限制
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None, None
    solver.SetTimeLimit(time_limit * 1000)  # 转换为毫秒

    #预备参数
    n = len(df_features)
    big_m = 500
    # 决策变量
    x_i = [solver.IntVar(0, 1, f"x_{i}") for i in range(n)]
    # 容量变量
    capacity = [solver.IntVar(0, big_m, f"capacity_{i}") for i in range(n)]
    # 目标函数：最小化平均损失率
    total_loss = solver.NumVar(0, solver.infinity(), "total_loss")
    loss_terms = []
    for i in range(n):
        surplus = solver.NumVar(0, solver.infinity(), f"surplus_{i}")
        solver.Add(surplus >= predicted_demand[i] - capacity[i]*31)
        solver.Add(surplus <= predicted_demand[i] - capacity[i]*31 + 5000 * (1 - x_i[i]))
        solver.Add(surplus >= 0)
        if predicted_demand[i] > 0:
            loss_terms.append(surplus / predicted_demand[i])
        else:
            loss_terms.append(0)
    solver.Add(total_loss == solver.Sum(loss_terms) / n)
    solver.Minimize(total_loss)

    # 约束条件1：容量约束
    min_capacity = 3
    for i in range(n):
        solver.Add(capacity[i] >= min_capacity * x_i[i])
        solver.Add(capacity[i] <= big_m * x_i[i])

    # 约束条件2：总容量约束
    total_capacity = solver.Sum(capacity)
    solver.Add(total_capacity <= total_max)

    # 约束条件3：利润约束
    lamuda = 0.5
    scheduling_cost = solver.Sum([x_i[i] * diff_i[i] for i in range(n)])
    vacancy_loss = solver.Sum([x_i[i] * diff_i[i] * diff_i[i] for i in range(n)])
    solver.Add((lamuda * scheduling_cost + (1 - lamuda) * vacancy_loss) <= profit)

    # 约束条件4：至少保留80%的站点
    solver.Add(solver.Sum(x_i) >= 0.8 * n)

    # 求解
    status = solver.Solve()

    if status == 0:  # 最优解
        x_i_opt = [int(x.solution_value()) for x in x_i]
        capacity_opt = [int(c.solution_value()) for c in capacity]
        print("找到最优解，状态码：", status)
        print(x_i_opt, capacity_opt)
        print(np.sum(x_i_opt), np.sum(capacity_opt))
        return x_i_opt, capacity_opt
    else:
        print(f"未找到最优解，状态码：{status}")
        # 如果超时或找到可行解，返回当前找到的最佳解
        if status in (1, 4) and solver.num_solutions() > 0:
            x_i_opt = [int(x.solution_value()) for x in x_i]
            capacity_opt = [int(c.solution_value()) for c in capacity]
            print("返回当前找到的最佳解")
            print(x_i_opt, capacity_opt)
            return x_i_opt, capacity_opt
        return None, None


# 主程序
if __name__ == "__main__":
    # 读取数据
    df_features = pd.read_csv("demand_features.csv")

    # 给出总现有容量和约束上限
    total_capacity = 20000      #为统计表格1所得
    total_max = total_capacity*1.1
    profit = 800000     #为统计表格2所得

    # 运行优化，设置时间限制为300秒
    x_i_opt, capacity_opt = optimize_stations(df_features, total_max, profit, time_limit=300)

    # 处理结果
    df_features['status'] = 'close'  # 默认值
    df_features['capacity'] = 0  # 默认值
    if x_i_opt is not None and capacity_opt is not None:
        df_features['status'] = np.where(np.array(x_i_opt) == 1, 'keep', 'close')
        df_features['capacity'] = capacity_opt
        print("赋值成功")

    # 保存结果
    selected_columns = ['station_id', 'status', 'capacity']
    required_columns = ['station_id']
    for col in required_columns:
        if col not in df_features.columns:
            raise ValueError(f"数据中缺少必要的列: {col}")
    df1 = df_features[selected_columns].copy()
    df1.to_csv('result_1.csv', index=False)
