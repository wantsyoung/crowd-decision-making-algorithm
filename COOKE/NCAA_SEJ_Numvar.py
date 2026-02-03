import pandas as pd
import numpy as np
import os
import random
from scipy.stats import spearmanr

# 设置随机种子以确保结果可重复
random.seed(42)
np.random.seed(42)

# 读取权重文件
weight_df = pd.read_excel('Weight/Wei_OWA_Cooke.xlsx', index_col=0, engine='openpyxl')
weight_snr_df = pd.read_excel('Weight/Wei_SNR_Cooke.xlsx', index_col=0, engine='openpyxl')

# 存储所有结果
all_results = []

# 定义决策者数量范围 - 从4到30
n_range = range(4, 31, 1)

# 设置抽样次数 - 每个n值固定抽样10次
num_samples = 10

# 遍历DATA_COOKE文件夹中的所有Excel文件
data_folder = 'DATA_COOKE'
data_files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx') or f.endswith('.xls')]

# 收集所有问题数据
all_problems = []

for filename in data_files:
    file_path = os.path.join(data_folder, filename)
    dataset_name = os.path.splitext(filename)[0]

    # 检查数据集是否在权重文件中
    if dataset_name not in weight_df.columns or dataset_name not in weight_snr_df.columns:
        continue

    # 尝试读取数据文件，如果失败则跳过
    try:
        data_df = pd.read_excel(file_path, index_col=0, engine='openpyxl')
    except:
        try:
            data_df = pd.read_excel(file_path, index_col=0, engine='xlrd')
        except:
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
        expert_names = col_data.index[:-1]

        # 获取对应专家的权重
        weights = [dataset_weights.get(name, 1.0) for name in expert_names]
        weights_snr = [dataset_weights_snr.get(name, 1.0) for name in expert_names]

        # 存储问题数据
        problem_data = {
            'dataset_name': dataset_name,
            'problem_name': col,
            'expert_predictions': expert_predictions,
            'actual_value': actual_value,
            'expert_names': expert_names,
            'weights': weights,
            'weights_snr': weights_snr,
            'total_experts': len(expert_predictions)  # 添加总专家数记录
        }

        all_problems.append(problem_data)

# 对每个决策者数量n进行抽样和计算
for n in n_range:
    print(f"正在处理决策者数量: {n}")

    # 筛选出专家数量至少为n的问题
    valid_problems = [p for p in all_problems if p['total_experts'] >= n]

    if not valid_problems:
        print(f"  没有找到专家数量至少为{n}的问题，跳过")
        continue

    print(f"  找到{len(valid_problems)}个有效问题")

    for sample_idx in range(num_samples):
        print(f"    正在进行第{sample_idx + 1}次抽样")

        # 对每个有效问题随机选择n个决策者
        for problem in valid_problems:
            # 随机选择n个决策者
            indices = random.sample(range(len(problem['expert_predictions'])), n)

            # 获取抽样后的数据
            sampled_predictions = [problem['expert_predictions'][i] for i in indices]
            sampled_weights = [problem['weights'][i] for i in indices]
            sampled_weights_snr = [problem['weights_snr'][i] for i in indices]

            # 1. 预处理：去除异常值（3西格玛原则）
            mean = np.mean(sampled_predictions)
            std = np.std(sampled_predictions)
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            # 过滤异常值
            filtered_indices = [(pred >= lower_bound) and (pred <= upper_bound) for pred in sampled_predictions]
            filtered_predictions = [pred for i, pred in enumerate(sampled_predictions) if filtered_indices[i]]
            filtered_weights = [weight for i, weight in enumerate(sampled_weights) if filtered_indices[i]]
            filtered_weights_snr = [weight for i, weight in enumerate(sampled_weights_snr) if filtered_indices[i]]

            # 如果所有值都被过滤掉了，使用原始值
            if len(filtered_predictions) == 0:
                filtered_predictions = sampled_predictions
                filtered_weights = sampled_weights
                filtered_weights_snr = sampled_weights_snr

            # 归一化
            min_val = min(np.min(filtered_predictions), problem['actual_value'])
            max_val = max(np.max(filtered_predictions), problem['actual_value'])

            if max_val != min_val:
                normalized_predictions = [(pred - min_val) / (max_val - min_val) for pred in filtered_predictions]
                normalized_actual = (problem['actual_value'] - min_val) / (max_val - min_val)
            else:
                normalized_predictions = [0 for _ in filtered_predictions]
                normalized_actual = 0

            # 2. 计算问题难度
            absolute_errors = np.abs(np.array(normalized_predictions) - normalized_actual)
            x = np.mean(absolute_errors)
            y = np.var(absolute_errors)
            difficulty = x ** 2 + y

            # 3. CBF算法整合预测
            candidate_points = np.arange(-0.1, 1.11, 0.01)

            min_corr = float('inf')
            best_candidates = []

            for candidate in candidate_points:
                errors = np.abs(np.array(normalized_predictions) - candidate)

                corr, _ = spearmanr(filtered_weights, errors)

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
            mean_prediction = np.mean(normalized_predictions)
            median_prediction = np.median(normalized_predictions)

            # 权值最大专家的预测值
            max_weight_idx = np.argmax(filtered_weights)
            max_weight_prediction = normalized_predictions[max_weight_idx]

            # OWA加权平均值
            owa_weighted_mean = np.average(normalized_predictions, weights=filtered_weights)
            # SNR加权平均值
            snr_weighted_mean = np.average(normalized_predictions, weights=filtered_weights_snr)

            # 保存结果
            result = {
                '预测目标': f"{problem['dataset_name']}_{problem['problem_name']}",
                '专家人数': n,  # 存储对应的n，而不是原有人数
                '抽样编号': sample_idx + 1,
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
    '预测目标', '专家人数', '抽样编号', '实际值', '问题难度', 'D_COR_OWA',
    '平均值', '中位数', '权值最大', 'OWA加权', 'SNR加权'
]
results_df = results_df[columns_order]

results_df.to_excel('result_Cooke_OWA_varying_n.xlsx', index=False, engine='openpyxl')

print("程序执行完成！结果已保存到 result_Cooke_OWA_varying_n.xlsx")

# 可选：生成汇总统计结果（按决策者数量）
summary_results = []

for n in n_range:
    n_data = results_df[results_df['专家人数'] == n]

    if len(n_data) == 0:
        continue

    # 计算各种方法的误差
    n_data = n_data.copy()
    n_data['D_COR_OWA误差'] = np.abs(n_data['D_COR_OWA'] - n_data['实际值'])
    n_data['平均值误差'] = np.abs(n_data['平均值'] - n_data['实际值'])
    n_data['中位数误差'] = np.abs(n_data['中位数'] - n_data['实际值'])
    n_data['权值最大误差'] = np.abs(n_data['权值最大'] - n_data['实际值'])
    n_data['OWA加权误差'] = np.abs(n_data['OWA加权'] - n_data['实际值'])
    n_data['SNR加权误差'] = np.abs(n_data['SNR加权'] - n_data['实际值'])

    summary = {
        '决策者数量': n,
        '样本数量': len(n_data),
        'D_COR_OWA平均误差': n_data['D_COR_OWA误差'].mean(),
        '平均值平均误差': n_data['平均值误差'].mean(),
        '中位数平均误差': n_data['中位数误差'].mean(),
        '权值最大平均误差': n_data['权值最大误差'].mean(),
        'OWA加权平均误差': n_data['OWA加权误差'].mean(),
        'SNR加权平均误差': n_data['SNR加权误差'].mean(),
        'D_COR_OWA误差标准差': n_data['D_COR_OWA误差'].std(),
        '平均值误差标准差': n_data['平均值误差'].std(),
        '中位数误差标准差': n_data['中位数误差'].std()
    }

    summary_results.append(summary)

# 保存汇总结果
summary_df = pd.DataFrame(summary_results)
summary_df.to_excel('summary_Cooke_OWA_varying_n.xlsx', index=False, engine='openpyxl')

print("汇总结果已保存到 summary_Cooke_OWA_varying_n.xlsx")