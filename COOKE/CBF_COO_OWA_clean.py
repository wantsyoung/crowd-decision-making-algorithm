#3西格玛去除异常值
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr

# 读取权重文件
weight_df = pd.read_excel('Weight/Wei_OWA_Cooke.xlsx', index_col=0, engine='openpyxl')
weight_snr_df = pd.read_excel('Weight/Wei_SNR_Cooke.xlsx', index_col=0, engine='openpyxl')
# 存储所有结果
all_results = []

# 遍历DATA_COOKE文件夹中的所有Excel文件
data_folder = 'DATA_COOKE'
for filename in os.listdir(data_folder):
    if not (filename.endswith('.xlsx') or filename.endswith('.xls')):
        continue

    file_path = os.path.join(data_folder, filename)
    dataset_name = os.path.splitext(filename)[0]

    # 尝试读取数据文件，如果失败则跳过
    try:
        data_df = pd.read_excel(file_path, index_col=0, engine='openpyxl')
    except:
        try:
            data_df = pd.read_excel(file_path, index_col=0, engine='xlrd')
        except:
            continue

    # 检查数据集是否在权重文件中
    if dataset_name not in weight_df.columns:
        continue
    if dataset_name not in weight_snr_df.columns:
        continue

    dataset_weights = weight_df[dataset_name]
    dataset_weights_snr = weight_snr_df[dataset_name]

    # 处理每个问题
    for col in data_df.columns:
        if not col.startswith('QUE.'):
            continue

        # 获取该列数据
        col_data = data_df[col].dropna()

        if len(col_data) < 2:
            continue

        # 分离专家预测值和实际值
        expert_predictions = col_data.iloc[:-1].values
        actual_value = col_data.iloc[-1]

        # 1. 预处理：去除异常值（3西格玛原则）
        mean = np.mean(expert_predictions)
        std = np.std(expert_predictions)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        # 过滤异常值
        filtered_indices = (expert_predictions >= lower_bound) & (expert_predictions <= upper_bound)
        filtered_predictions = expert_predictions[filtered_indices]

        # 如果所有值都被过滤掉了，使用原始值
        if len(filtered_predictions) == 0:
            filtered_predictions = expert_predictions

        # 归一化
        min_val = min(np.min(filtered_predictions), actual_value)
        max_val = max(np.max(filtered_predictions), actual_value)

        if max_val != min_val:
            normalized_predictions = (filtered_predictions - min_val) / (max_val - min_val)
            normalized_actual = (actual_value - min_val) / (max_val - min_val)
        else:
            normalized_predictions = filtered_predictions * 0
            normalized_actual = 0

        # 2. 计算问题难度
        absolute_errors = np.abs(normalized_predictions - normalized_actual)
        x = np.mean(absolute_errors)
        y = np.var(absolute_errors)
        difficulty = x ** 2 + y

        # 3. CBF算法整合预测
        candidate_points = np.arange(-0.1, 1.11, 0.01)

        min_corr = float('inf')
        best_candidates = []

        # 获取对应专家的权重
        expert_names = col_data.index[:-1]
        filtered_expert_names = [name for i, name in enumerate(expert_names) if filtered_indices[i]]
        weights = [dataset_weights.get(name, 1.0) for name in filtered_expert_names]
        weights_snr = [dataset_weights_snr.get(name, 1.0) for name in filtered_expert_names]

        for candidate in candidate_points:
            errors = np.abs(normalized_predictions - candidate)

            corr, _ = spearmanr(weights, errors)

            if np.isnan(corr):
                continue

            if corr < min_corr:
                min_corr = corr
                best_candidates = [candidate]
            elif abs(corr - min_corr) < 1e-10:
                best_candidates.append(candidate)

        if best_candidates:
            d_cor_owa = np.mean(best_candidates)
        else:
            d_cor_owa = np.mean(normalized_predictions)

        # 计算其他统计量（使用归一化后的值）
        num_experts = len(normalized_predictions)
        mean_prediction = np.mean(normalized_predictions)
        median_prediction = np.median(normalized_predictions)

        # 权值最大专家的预测值
        max_weight_idx = np.argmax(weights)
        max_weight_prediction = normalized_predictions[max_weight_idx]

        # OWA加权平均值
        owa_weighted_mean = np.average(normalized_predictions, weights=weights)
        # SNR加权平均值
        snr_weighted_mean = np.average(normalized_predictions, weights=weights_snr)
        # 保存结果
        result = {
            '预测目标': f'{dataset_name}_{col}',
            '专家人数': num_experts,
            '实际值': normalized_actual,
            '问题难度': difficulty,
            'D_COR_OWA': d_cor_owa,
            '平均值': mean_prediction,
            '中位数': median_prediction,
            '权值最大': max_weight_prediction,
            'OWA加权': owa_weighted_mean,
            'SNR加权': snr_weighted_mean
        }

        all_results.append(result)

# 创建结果DataFrame并保存
results_df = pd.DataFrame(all_results)

columns_order = [
    '预测目标', '专家人数', '实际值', '问题难度', 'D_COR_OWA',
    '平均值', '中位数', '权值最大', 'OWA加权', 'SNR加权'
]
results_df = results_df[columns_order]

results_df.to_excel('result_Cooke_OWA_clean.xlsx', index=False, engine='openpyxl')