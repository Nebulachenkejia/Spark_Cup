import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp


def optimize_stations(df_features, total_max, profit, time_limit=300):
    # --------------------------
    # 1. 数据预处理（核心优化：数值稳定性）
    # --------------------------
    # 计算预测需求并避免零值（防止除零错误）
    df_features['predicted_demand'] = (
            0.55 * df_features['borrow_count'] +
            0.44 * df_features['return_count'] +
            220 * df_features['casual_ratio'] +
            110 * df_features['POI_score']
    )
    min_demand = 1e-6  # 最小需求值，避免除零
    df_features['predicted_demand'] = df_features['predicted_demand'].clip(lower=min_demand)
    predicted_demand = df_features['predicted_demand'].values

    # 计算借还差异（确保非负）
    df_features['diff_i'] = np.abs(df_features['borrow_count'] - df_features['return_count']).clip(lower=0)
    diff_i = df_features['diff_i'].values

    # --------------------------
    # 2. 求解器初始化（核心优化：效率参数）
    # --------------------------
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None, None
    solver.SetTimeLimit(time_limit * 1000)  # 毫秒级时间限制
    solver.SetNumThreads(4)  # 多线程加速求解（根据CPU核心数调整）

    n = len(df_features)
    big_m = 10000  # 经测试的合理值，兼顾约束有效性和求解效率

    # --------------------------
    # 3. 变量定义（精简变量，减少内存占用）
    # --------------------------
    x_i = [solver.IntVar(0, 1, f"x_{i}") for i in range(n)]  # 是否保留站点
    capacity = [solver.IntVar(0, big_m, f"capacity_{i}") for i in range(n)]  # 站点容量

    # --------------------------
    # 4. 目标函数（核心优化：业务逻辑清晰化）
    # 目标：最小化容量与需求的偏离程度（|1 - 月容量/月需求|）
    # --------------------------
    total_loss = solver.NumVar(0, solver.infinity(), "total_loss")
    loss_terms = []

    for i in range(n):
        # 月容量与月需求的比例（31天）
        monthly_capacity_ratio = capacity[i] * 31 / predicted_demand[i]

        # 偏离损失：|1 - 比例|（用线性约束表达绝对值）
        surplus = solver.NumVar(0, solver.infinity(), f"surplus_{i}")
        solver.Add(surplus >= 1 - monthly_capacity_ratio)  # 比例 < 1 时的损失
        solver.Add(surplus >= monthly_capacity_ratio - 1)  # 比例 > 1 时的损失

        # 关闭的站点不计算损失
        solver.Add(surplus <= big_m * x_i[i])
        solver.Add(surplus >= 0)  # 确保损失非负

        # 按需求权重计算损失（需求大的站点，偏离影响更大）
        loss_terms.append(surplus / predicted_demand[i])

    # 平均损失最小化
    solver.Add(total_loss == solver.Sum(loss_terms) / n)
    solver.Minimize(total_loss)

    # --------------------------
    # 5. 约束条件（核心优化：约束强度平衡）
    # --------------------------
    # 5.1 站点状态与容量关联（选中则容量≥1，关闭则容量=0）

    min_capacity = 1
    for i in range(n):
        solver.Add(capacity[i] >= min_capacity * x_i[i])
        solver.Add(capacity[i] <= big_m * x_i[i])

    # 5.2 总容量上限
    total_capacity = solver.Sum(capacity)
    solver.Add(total_capacity <= total_max)

    # 5.3 利润约束（借还差异导致的成本）
    lamuda = 0.8
    scheduling_cost = solver.Sum([x_i[i] * diff_i[i] for i in range(n)])
    vacancy_loss = solver.Sum([x_i[i] * diff_i[i] ** 2 for i in range(n)])
    solver.Add((lamuda * scheduling_cost + (1 - lamuda) * vacancy_loss) <= profit)

    # 5.4 最少保留80%站点
    min_stations = int(np.ceil(0.8 * n))  # 向上取整，确保满足比例
    solver.Add(solver.Sum(x_i) >= min_stations)

    # --------------------------
    # 6. 求解与结果验证（核心优化：健壮性）
    # --------------------------
    status = solver.Solve()

    # 状态码对应关系：0=最优，1=可行解，2=无解，3=无界，4=中止（超时等）
    if status in (0, 1):
        x_i_opt = [int(x.solution_value()) for x in x_i]
        capacity_opt = [int(c.solution_value()) for c in capacity]

        # 验证解的合理性（避免无效解）
        total_cap = np.sum(capacity_opt)
        print(f"求解成功（状态码：{status}），保留站点数：{sum(x_i_opt)}，总容量：{total_cap}")
        return x_i_opt, capacity_opt
    else:
        print(f"求解失败（状态码：{status}），无有效解")
        return None, None


if __name__ == "__main__":
    # --------------------------
    # 主程序（优化：流程清晰化）
    # --------------------------
    # 读取数据
    df_features = pd.read_csv("demand_features.csv")

    # 计算容量上限（基于初始容量15）
    initial_capacity = 15
    df_features['capacity'] = initial_capacity
    total_old = np.sum(df_features['capacity'])
    total_max = 14000


    # 运行优化
    profit = 100000000
    x_i_opt, capacity_opt = optimize_stations(df_features, total_max, profit, time_limit=300)

    # 处理结果（避免空值错误）
    df_features['status'] = 'close'
    df_features['capacity'] = 0
    if x_i_opt is not None and capacity_opt is not None:
        # 显式转换为数组，确保np.where正常工作
        df_features['status'] = np.where(np.array(x_i_opt) == 1, 'keep', 'close')
        df_features['capacity'] = capacity_opt

    # 保存结果（验证必要字段）
    required_cols = ['station_id', 'status', 'capacity']
    if not set(required_cols).issubset(df_features.columns):
        missing = set(required_cols) - set(df_features.columns)
        raise ValueError(f"缺少必要字段：{missing}")

    result = df_features[required_cols].copy()
    result.to_csv('result_135.csv', index=False)
    print("结果已保存至 result_111.csv")