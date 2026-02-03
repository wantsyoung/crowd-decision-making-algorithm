import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr
import warnings

# 读取权重文件
weight_owa_df = pd.read_excel('Weight/Wei_OWA_Cooke.xlsx', index_col=0, engine='openpyxl')
weight_snr_df = pd.read_excel('Weight/Wei_SNR_Cooke.xlsx', index_col=0, engine='openpyxl')

# 存储所有结果
all_results = []
# 存储有问题的数据集信息
problematic_datasets = []

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
            print(f"无法读取文件: {filename}")
            continue

    # 检查数据集是否在权重文件中
    if dataset_name not in weight_owa_df.columns:
        print(f"数据集 {dataset_name} 不在权重文件 Wei_OWA_Cooke.xlsx 中")
        continue
    if dataset_name not in weight_snr_df.columns:
        print(f"数据集 {dataset_name} 不在权重文件 Wei_SNR_Cooke.xlsx 中")
        continue

    dataset_weights_owa = weight_owa_df[dataset_name]
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

        # 1. 归一化
        min_val = min(np.min(expert_predictions), actual_value)
        max_val = max(np.max(expert_predictions), actual_value)

        if max_val != min_val:
            normalized_predictions = (expert_predictions - min_val) / (max_val - min_val)
            normalized_actual = (actual_value - min_val) / (max_val - min_val)
        else:
            normalized_predictions = expert_predictions * 0
            normalized_actual = 0

        # 2. 计算问题难度
        absolute_errors = np.abs(normalized_predictions - normalized_actual)
        x = np.mean(absolute_errors)
        y = np.var(absolute_errors)
        difficulty = x ** 2 + y

        # 3. CBF算法整合预测
        candidate_points = np.arange(0, 1, 0.01)

        min_corr = float('inf')
        best_candidates = []

        # 获取对应专家的权重
        expert_names = col_data.index[:-1]
        weights_OWA = [dataset_weights_owa.get(name, 1.0) for name in expert_names]
        weights_snr = [dataset_weights_snr.get(name, 1.0) for name in expert_names]

        # 检查权重和错误数据
        current_problem = f"{dataset_name}_{col}"

        # 检查是否为常数数组
        if np.all(weights_OWA == weights_OWA[0]):
            print(f"警告: 数据集 {current_problem} 的 weights_OWA 是常数数组 (所有值都为 {weights_OWA[0]})")
            problematic_datasets.append({
                'dataset': current_problem,
                'issue': '常数权重数组',
                'weights_OWA': weights_OWA,
                'weights_snr': weights_snr
            })

        if np.all(normalized_predictions == normalized_predictions[0]):
            print(
                f"警告: 数据集 {current_problem} 的 normalized_predictions 是常数数组 (所有值都为 {normalized_predictions[0]})")
            problematic_datasets.append({
                'dataset': current_problem,
                'issue': '常数预测值数组',
                'predictions': normalized_predictions.tolist(),
                'actual_value': normalized_actual
            })

        for candidate in candidate_points:
            errors = np.abs(normalized_predictions - candidate)

            # 检查错误数组是否为常数
            if np.all(errors == errors[0]):
                # 如果是常数数组，跳过这个候选点或使用默认值
                continue

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    corr, _ = spearmanr(weights_OWA, errors)

                    # 检查是否有警告
                    if w and any("ConstantInputWarning" in str(warning.category) for warning in w):
                        print(f"警告: 数据集 {current_problem} 候选点 {candidate:.2f} 出现常数输入")
                        print(f"  - weights_OWA: {weights_OWA}")
                        print(f"  - errors: {errors}")
                        continue

                    if np.isnan(corr):
                        print(f"警告: 数据集 {current_problem} 候选点 {candidate:.2f} 相关系数为 NaN")
                        print(f"  - weights_OWA: {weights_OWA}")
                        print(f"  - errors: {errors}")
                        continue

                    if corr < min_corr:
                        min_corr = corr
                        best_candidates = [candidate]
                    elif abs(corr - min_corr) < 1e-10:
                        best_candidates.append(candidate)

            except Exception as e:
                print(f"错误: 数据集 {current_problem} 候选点 {candidate:.2f} 计算相关系数时出错: {e}")
                continue

        if best_candidates:
            d_cor_owa = np.mean(best_candidates)
        else:
            print(f"警告: 数据集 {current_problem} 没有有效的候选点，使用平均值作为备选")
            d_cor_owa = np.mean(normalized_predictions)

        # 计算其他统计量（使用归一化后的值）
        num_experts = len(normalized_predictions)
        mean_prediction = np.mean(normalized_predictions)
        median_prediction = np.median(normalized_predictions)

        # 权值最大专家的预测值
        max_weight_idx = np.argmax(weights_OWA)
        max_weight_prediction = normalized_predictions[max_weight_idx]

        # OWA加权平均值
        owa_weighted_mean = np.average(normalized_predictions, weights=weights_OWA)
        # SNR加权平均值
        snr_weighted_mean = np.average(normalized_predictions, weights=weights_snr)

        # 保存结果
        result = {
            '预测目标': current_problem,
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

if len(all_results) > 0:
    columns_order = [
        '预测目标', '专家人数', '实际值', '问题难度', 'D_COR_OWA',
        '平均值', '中位数', '权值最大', 'OWA加权', 'SNR加权'
    ]
    results_df = results_df[columns_order]
    results_df.to_excel('result_Cooke_OWA.xlsx', index=False, engine='openpyxl')
    print(f"成功处理 {len(all_results)} 个问题")
else:
    print("没有成功处理任何问题")

# 保存有问题的数据集信息
if problematic_datasets:
    problem_df = pd.DataFrame(problematic_datasets)
    problem_df.to_excel('problematic_datasets_Cooke.xlsx', index=False, engine='openpyxl')
    print(f"发现 {len(problematic_datasets)} 个有问题的数据集，已保存到 problematic_datasets_Cooke.xlsx")

# 打印总结信息
print("\n=== 处理总结 ===")
print(f"总处理问题数: {len(all_results)}")
print(f"发现问题数据集数: {len(problematic_datasets)}")

if problematic_datasets:
    print("\n有问题的数据集详情:")
    for problem in problematic_datasets:
        print(f"- {problem['dataset']}: {problem['issue']}")