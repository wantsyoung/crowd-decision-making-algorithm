import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 1. 读取数据文件
gdp_data = pd.read_excel('Data_GDP.xlsx')
weight_data = pd.read_excel('Weight\Wei_SNR_GDP.xlsx')

# 2. 合并数据
merged_data = pd.merge(gdp_data, weight_data, on='专家编号', how='inner')

# 按预测目标分组处理
results = []

for target, group in merged_data.groupby('预测目标'):
    # 获取数据
    predictions = group['预测值'].values
    actual_value = group['实际值'].iloc[0]
    weights = group['权值_SNR'].values

    # 3. 计算问题难度
    absolute_errors = np.abs(predictions - actual_value)
    x = np.mean(absolute_errors)
    y = np.var(absolute_errors)
    difficulty = x ** 2 + y

    # 4. CBF算法整合预测
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)
    range_width = max_pred - min_pred
    expanded_min = min_pred - 0.1 * range_width
    expanded_max = max_pred + 0.1 * range_width

    candidate_points = np.linspace(expanded_min, expanded_max, 100)

    min_corr = float('inf')
    best_candidates = []

    for candidate in candidate_points:
        errors = np.abs(predictions - candidate)
        corr, _ = spearmanr(weights, errors)

        if np.isnan(corr):
            continue

        if corr < min_corr:
            min_corr = corr
            best_candidates = [candidate]
        elif abs(corr - min_corr) < 1e-10:
            best_candidates.append(candidate)

    if best_candidates:
        d_cor_snr = np.mean(best_candidates)
    else:
        d_cor_snr = np.mean(predictions)

    # 5. 计算其他统计量
    num_experts = len(group)
    mean_prediction = np.mean(predictions)
    median_prediction = np.median(predictions)
    max_weight_idx = np.argmax(weights)
    max_weight_prediction = predictions[max_weight_idx]
    snr_weighted_mean = np.average(predictions, weights=weights)

    # 保存结果
    result = {
        '预测目标': target,
        '参与专家人数': num_experts,
        '实际值': actual_value,
        '问题难度': difficulty,
        'D_COR_SNR': d_cor_snr,
        '参与专家预测值的平均值': mean_prediction,
        '参与专家预测值的中位数': median_prediction,
        '权值最大专家的预测值': max_weight_prediction,
        'SNR加权平均值': snr_weighted_mean
    }

    results.append(result)

# 6. 创建结果DataFrame并保存
results_df = pd.DataFrame(results)

columns_order = [
    '预测目标', '参与专家人数', '实际值', '问题难度', 'D_COR_SNR',
    '参与专家预测值的平均值', '参与专家预测值的中位数',
    '权值最大专家的预测值', 'SNR加权平均值'
]
results_df = results_df[columns_order]

results_df.to_excel('GDP预测整合结果.xlsx', index=False)