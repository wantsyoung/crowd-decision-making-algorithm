import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel("Data_GDP.xlsx")

# 获取所有专家编号和预测目标
all_experts = df['专家编号'].unique()
all_targets = df['预测目标'].unique()

# 初始化专家权值累加器
expert_weights_accumulator = {expert: [] for expert in all_experts}

# 按预测目标分组处理
target_groups = df.groupby('预测目标')

for target_name, target_group in target_groups:
    # 获取该预测目标的所有专家数据
    target_data = target_group[['专家编号', '误差绝对值']].copy()

    # 按误差绝对值从大到小排序
    sorted_data = target_data.sort_values('误差绝对值', ascending=False)

    # 获取专家数量
    N = len(sorted_data)

    # 分配权值
    weights = []
    for rank in range(N):
        if N <= 2:
            if rank == 0:
                weight = 0
            else:
                weight = 1
        else:
            if rank == 0:
                weight = 0
            elif rank == N - 1:
                weight = 1
            else:
                weight = rank * (1.0 / (N - 2))
        weights.append(weight)

    # 将权值添加到专家的累加器中
    for i, row in sorted_data.iterrows():
        expert_weights_accumulator[row['专家编号']].append(weights.pop(0))

# 计算每个专家的平均权值
result_data = []
for expert, weights_list in expert_weights_accumulator.items():
    avg_weight = np.mean(weights_list) if weights_list else 0
    result_data.append({
        '专家编号': expert,
        '权值_OWA': avg_weight
    })

# 创建结果DataFrame并排序
result_df = pd.DataFrame(result_data)
result_df = result_df.sort_values('权值_OWA', ascending=False)

# 保存到Excel文件
result_df.to_excel("Wei_OWA_GDP.xlsx", index=False)

print("处理完成！结果已保存到 Wei_OWA_GDP.xlsx")