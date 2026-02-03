import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel("Data_GDP.xlsx")

# 按专家编号分组
expert_groups = df.groupby('专家编号')

# 计算每个专家的权值
result_data = []
for expert_id, group in expert_groups:
    # 获取该专家的所有误差值
    errors = group['误差值'].values.astype(float)

    # 计算平均偏差
    bias = np.mean(errors)

    # 计算方差
    variance = np.var(errors, ddof=1)

    # 计算均方误差
    mse = bias ** 2 + variance

    # 计算信噪比（SNR）
    snr = 1 / mse if mse != 0 else float('inf')

    result_data.append({
        '专家编号': expert_id,
        '权值_SNR': snr
    })

# 创建结果DataFrame
result_df = pd.DataFrame(result_data)

# 按权值从高到低排序
result_df = result_df.sort_values('权值_SNR', ascending=False)

# 保存到Excel文件
result_df.to_excel("Wei_SNR_GDP.xlsx", index=False)

print("处理完成！结果已保存到 Wei_SNR_GDP.xlsx")