import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 1. 读取数据文件
gdp_data = pd.read_excel('Data_GDP.xlsx')
weight_owa_df = pd.read_excel('Weight\Wei_OWA_GDP.xlsx')
weight_snr_df = pd.read_excel('Weight\Wei_SNR_GDP.xlsx')

# 2. 合并数据
merged_data = pd.merge(gdp_data, weight_owa_df, on='专家编号', how='inner')
merged_data = pd.merge(merged_data, weight_snr_df, on='专家编号', how='inner')

# 按预测目标分组处理
results = []

for target, group in merged_data.groupby('预测目标'):
    # 获取数据
    predictions = group['预测值'].values
    actual_value = group['实际值'].iloc[0]
    weights_owa = group['权值_OWA'].values
    weights_snr = group['权值_SNR'].values

    # 对预测值和实际值进行归一化处理
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)

    # 防止除零错误
    normalized_predictions = (predictions - min_pred) / (max_pred - min_pred)
    normalized_actual = (actual_value - min_pred) / (max_pred - min_pred)


    # 3. 计算问题难度（使用归一化后的数据）
    absolute_errors = np.abs(normalized_predictions - normalized_actual)
    x = np.mean(absolute_errors)
    y = np.var(absolute_errors)
    difficulty = x ** 2 + y

    # 4. CBF算法整合预测（使用归一化后的数据）
    min_norm_pred = np.min(normalized_predictions)
    max_norm_pred = np.max(normalized_predictions)
    range_width = max_norm_pred - min_norm_pred
    expanded_min = min_norm_pred - 0.1 * range_width
    expanded_max = max_norm_pred + 0.1 * range_width

    candidate_points = np.linspace(expanded_min, expanded_max, 100)

    min_corr = float('inf')
    best_candidates = []

    for candidate in candidate_points:
        errors = np.abs(normalized_predictions - candidate)
        corr, _ = spearmanr(weights_owa, errors)

        if np.isnan(corr):
            continue

        if corr < min_corr:
            min_corr = corr
            best_candidates = [candidate]
        elif abs(corr - min_corr) < 1e-10:
            best_candidates.append(candidate)

    if best_candidates:
        d_cor_owa = np.mean(best_candidates)  # 保持归一化结果
    else:
        d_cor_owa = np.mean(normalized_predictions)  # 保持归一化结果

    # 5. 计算其他统计量（使用归一化后的数据）
    num_experts = len(group)
    mean_prediction = np.mean(normalized_predictions)  # 归一化后的平均值
    median_prediction = np.median(normalized_predictions)  # 归一化后的中位数
    max_weight_idx = np.argmax(weights_owa)
    max_weight_prediction = normalized_predictions[max_weight_idx]  # 归一化后的权值最大预测
    owa_weighted_mean = np.average(normalized_predictions, weights=weights_owa)  # 归一化后的OWA加权
    snr_weighted_mean = np.average(normalized_predictions, weights=weights_snr)  # 归一化后的SNR加权

    # 保存结果（所有值都是归一化后的）
    result = {
        '预测目标': target,
        '专家人数': num_experts,
        '实际值': normalized_actual,  # 归一化后的实际值
        '问题难度': difficulty,
        'D_COR_OWA': d_cor_owa,
        '平均值': mean_prediction,
        '中位数': median_prediction,
        '权值最大': max_weight_prediction,
        'OWA加权': owa_weighted_mean,
        'SNR加权': snr_weighted_mean
    }

    results.append(result)

# 6. 创建结果DataFrame并保存
results_df = pd.DataFrame(results)

columns_order = [
    '预测目标', '专家人数', '实际值', '问题难度', 'D_COR_OWA',
    '平均值', '中位数',
    '权值最大', 'OWA加权', 'SNR加权'
]
results_df = results_df[columns_order]

results_df.to_excel('result_GDP_OWA_normalized.xlsx', index=False)