import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import random

# 设置matplotlib后端为Agg，避免显示问题
plt.switch_backend('Agg')

# 设置字体为Times New Roman并增大字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也设为Times New Roman风格
plt.rcParams['font.size'] = 14  # 增大全局字体大小

# 设置随机种子以确保结果可重复
random.seed(42)
np.random.seed(42)

# 1. 读取数据文件
gdp_data = pd.read_excel('Data_GDP.xlsx')
weight_owa_df = pd.read_excel('Weight\Wei_OWA_GDP.xlsx')
weight_snr_df = pd.read_excel('Weight\Wei_SNR_GDP.xlsx')

# 2. 合并数据
merged_data = pd.merge(gdp_data, weight_owa_df, on='专家编号', how='inner')
merged_data = pd.merge(merged_data, weight_snr_df, on='专家编号', how='inner')

# 3. 定义决策者数量范围
n_range = range(4, 55, 1)  # 从4到40

# 4. 设置抽样次数（为了结果的稳定性）
num_samples = 10  # 每个n值抽样2次

all_results = []

# 按预测目标分组
grouped_data = merged_data.groupby('预测目标')

for n in n_range:
    print(f"正在处理决策者数量: {n}")

    for sample_idx in range(num_samples):
        # 对每个预测目标随机选择n个决策者
        sampled_groups = []

        for target, group in grouped_data:
            # 新增：判断决策者数量是否小于n，如果小于则跳过
            if len(group) < n:
                continue

            # 随机选择n个决策者
            sampled_group = group.sample(n=n, random_state=sample_idx)
            sampled_groups.append(sampled_group)

        # 如果没有足够的数据，跳过这个抽样
        if not sampled_groups:
            continue

        # 合并所有抽样数据
        sampled_data = pd.concat(sampled_groups, ignore_index=True)

        # 按预测目标处理抽样数据
        for target, group in sampled_data.groupby('预测目标'):
            # 获取数据
            predictions = group['预测值'].values
            actual_value = group['实际值'].iloc[0]
            weights_owa = group['权值_OWA'].values
            weights_snr = group['权值_SNR'].values

            # 计算问题难度
            absolute_errors = np.abs(predictions - actual_value)
            x = np.mean(absolute_errors)
            y = np.var(absolute_errors)
            difficulty = x ** 2 + y

            # CBF算法整合预测
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
                corr, _ = spearmanr(weights_owa, errors)

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
                d_cor_owa = np.mean(predictions)

            # 计算其他统计量
            mean_prediction = np.mean(predictions)
            median_prediction = np.median(predictions)
            max_weight_idx = np.argmax(weights_owa)
            max_weight_prediction = predictions[max_weight_idx]
            owa_weighted_mean = np.average(predictions, weights=weights_owa)
            snr_weighted_mean = np.average(predictions, weights=weights_snr)

            # 计算各种方法的误差
            d_cor_owa_error = np.abs(d_cor_owa - actual_value)
            mean_error = np.abs(mean_prediction - actual_value)
            median_error = np.abs(median_prediction - actual_value)
            max_weight_error = np.abs(max_weight_prediction - actual_value)
            owa_weighted_error = np.abs(owa_weighted_mean - actual_value)
            snr_weighted_error = np.abs(snr_weighted_mean - actual_value)

            # 保存结果
            result = {
                '决策者数量': n,
                '抽样编号': sample_idx + 1,
                '预测目标': target,
                '实际值': actual_value,
                '问题难度': difficulty,
                'D_COR_OWA': d_cor_owa,
                'D_COR_OWA误差': d_cor_owa_error,
                '平均值': mean_prediction,
                '平均值误差': mean_error,
                '中位数': median_prediction,
                '中位数误差': median_error,
                '权值最大': max_weight_prediction,
                '权值最大误差': max_weight_error,
                'OWA加权': owa_weighted_mean,
                'OWA加权误差': owa_weighted_error,
                'SNR加权': snr_weighted_mean,
                'SNR加权误差': snr_weighted_error
            }

            all_results.append(result)

# 5. 创建结果DataFrame并保存
results_df = pd.DataFrame(all_results)

# 重新排列列的顺序
columns_order = [
    '决策者数量', '抽样编号', '预测目标', '实际值', '问题难度',
    'D_COR_OWA', 'D_COR_OWA误差',
    '平均值', '平均值误差',
    '中位数', '中位数误差',
    '权值最大', '权值最大误差',
    'OWA加权', 'OWA加权误差',
    'SNR加权', 'SNR加权误差'
]
results_df = results_df[columns_order]

# 保存结果
results_df.to_excel('result_GDP_OWA_varying_n.xlsx', index=False)

print("程序执行完成！结果已保存到 result_GDP_OWA_varying_n.xlsx")

# 6. 生成汇总统计结果（按决策者数量）
summary_results = []

# 定义方法名称和对应的误差列
methods = {
    'NCAA': 'D_COR_OWA误差',
    'Averaging': '平均值误差',
    'Median': '中位数误差',
    'Dictator': '权值最大误差',
    'OWA': 'OWA加权误差',
    'SNR': 'SNR加权误差'
}

for n in n_range:
    n_data = results_df[results_df['决策者数量'] == n]

    if len(n_data) == 0:
        continue

    summary = {'Number of decision makers': n, 'Sample size': len(n_data)}

    # 计算每个方法的平均误差、标准差和MSE
    for method_name, error_col in methods.items():
        errors = n_data[error_col]
        mean_error = errors.mean()
        std_error = errors.std()
        mse = mean_error ** 2 + std_error ** 2

        summary[f'{method_name}_mean_error'] = mean_error
        summary[f'{method_name}_std'] = std_error
        summary[f'{method_name}_MSE'] = mse

    summary_results.append(summary)

# 保存汇总结果
summary_df = pd.DataFrame(summary_results)
summary_df.to_excel('summary_GDP_OWA_varying_n.xlsx', index=False)

print("汇总结果已保存到 summary_GDP_OWA_varying_n.xlsx")

# 7. 绘制三张并列图表
# 创建图表 - 三张子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# 定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 第一张图：平均误差随决策者数量变化
x = summary_df['Number of decision makers']
for i, (method_name, error_col) in enumerate(methods.items()):
    ax1.plot(x, summary_df[f'{method_name}_mean_error'],
             marker='o', linewidth=2, label=method_name, color=colors[i])

ax1.set_xlabel('Number of decision makers', fontsize=16)
ax1.set_ylabel('Mean of error', fontsize=16)
ax1.set_title('Mean of error vs number of decision makers', fontsize=16)
ax1.legend(fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# 设置x轴和y轴标签字体
for label in ax1.get_xticklabels():
    label.set_fontsize(14)
for label in ax1.get_yticklabels():
    label.set_fontsize(14)

# 第二张图：标准差随决策者数量变化
for i, (method_name, error_col) in enumerate(methods.items()):
    ax2.plot(x, summary_df[f'{method_name}_std'],
             marker='s', linewidth=2, label=method_name, color=colors[i])

ax2.set_xlabel('Number of decision makers', fontsize=16)
ax2.set_ylabel('STD of error', fontsize=16)
ax2.set_title('STD of error vs number of decision makers', fontsize=16)
ax2.legend(fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

# 设置x轴和y轴标签字体
for label in ax2.get_xticklabels():
    label.set_fontsize(14)
for label in ax2.get_yticklabels():
    label.set_fontsize(14)

# 第三张图：MSE随决策者数量变化
for i, (method_name, error_col) in enumerate(methods.items()):
    ax3.plot(x, summary_df[f'{method_name}_MSE'],
             marker='^', linewidth=2, label=method_name, color=colors[i])

ax3.set_xlabel('Number of decision makers', fontsize=16)
ax3.set_ylabel('MSE', fontsize=16)
ax3.set_title('MSE vs number of decision makers', fontsize=16)
ax3.legend(fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.7)

# 设置x轴和y轴标签字体
for label in ax3.get_xticklabels():
    label.set_fontsize(14)
for label in ax3.get_yticklabels():
    label.set_fontsize(14)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('GDP_OWA_varying_n_analysis.png', dpi=300, bbox_inches='tight')
print("图表已保存为 GDP_OWA_varying_n_analysis.png")

# 打印统计结果
print("\n各预测方法在不同决策者数量下的性能统计：")
print(summary_df.head())