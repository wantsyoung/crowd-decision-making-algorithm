import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
df = pd.read_excel('result_GDP_OWA_normalized.xlsx')
# 只过滤掉实际值为负数的行，但保留NaN
df = df[df["实际值"] >= 0]

# 定义难度档位
# difficulty_levels = [0, 0.2, 0.4, 0.6, 0.8]
difficulty_levels = [0, 0.1, 0.2, 0.3, 0.4]
# difficulty_levels = [0, 0.04, 0.08, 0.17, 0.35]
# 创建难度区间
difficulty_ranges = [(difficulty_levels[i], difficulty_levels[i + 1]) for i in range(len(difficulty_levels) - 1)]
# 添加最后一个区间 (0.5, 无穷大)
difficulty_ranges.append((0.4, 1))

# 定义预测方法列名
prediction_columns = ['D_COR_OWA', '平均值', '中位数', '权值最大', 'OWA加权', 'SNR加权']

# 存储结果的字典
mean_results = {col: [] for col in prediction_columns}
variance_results = {col: [] for col in prediction_columns}
mse_results = {col: [] for col in prediction_columns}  # 存储均值的平方+方差
count_results = []  # 存储每个区间的数据点数量

# 对每个难度区间进行计算
for i, (low, high) in enumerate(difficulty_ranges):
    # 筛选问题难度在当前区间的数据
    if high == float('inf'):
        filtered_df = df[df['问题难度'] > low]
        range_label = f'({low}, ∞)'
    else:
        filtered_df = df[(df['问题难度'] > low) & (df['问题难度'] <= high)]
        range_label = f'({low}, {high}]'

    count_results.append(len(filtered_df))
    print(f"难度区间 {range_label}: {len(filtered_df)} 个数据点")

    # 对每个预测方法计算与实际值的绝对差值
    for col in prediction_columns:
        differences = abs(filtered_df[col] - filtered_df['实际值'])
        # 计算均值和方差
        mean_val = differences.mean()
        variance_val = differences.std()

        # 计算均值的平方+方差
        mse_val = mean_val ** 2 + variance_val ** 2

        # 存储结果
        mean_results[col].append(mean_val)
        variance_results[col].append(variance_val)
        mse_results[col].append(mse_val)

# 设置字体为Times New Roman并增大字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也设为Times New Roman风格
plt.rcParams['font.size'] = 14  # 增大全局字体大小

# 创建图表 - 现在有三张子图
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
fig, (ax3) = plt.subplots(1, 1, figsize=(7, 6))
# 定义方法名称的英文映射
method_names = {
    'D_COR_OWA': 'NCAA',
    '平均值': 'Averaging',
    '中位数': 'Median',
    '权值最大': 'Dictator',
    'OWA加权': 'OWA',
    'SNR加权': 'SNR'
}

# 为每个方法分配固定的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
method_colors = dict(zip(prediction_columns, colors))

# 第一个图：均值随难度变化
x = np.arange(len(difficulty_ranges))
width = 0.12  # 稍微调整柱状图宽度以适应三张图

# # 为每个预测方法绘制柱状图
# for i, col in enumerate(prediction_columns):
#     ax1.bar(x + i * width, mean_results[col], width, color=method_colors[col], label=method_names[col])
#
# ax1.set_xlabel('Problem Difficulty', fontsize=16)  # 增大坐标轴标签字体
# ax1.set_ylabel('MAE', fontsize=16)
# ax1.set_title('Variation of MAE with Problem Difficulty in SPF', fontsize=16)
# ax1.set_xticks(x + width * 3)  # 调整x轴刻度位置
# # 创建区间标签
range_labels = []
# for low, high in difficulty_ranges:
#     if high == float('inf'):
#         range_labels.append(f'({low}, ∞)')
#     else:
#         range_labels.append(f'({low}, {high}]')
# ax1.set_xticklabels(range_labels, rotation=45, fontsize=14)
# ax1.legend(fontsize=14)  # 增大图例字体
# # 添加网格线便于阅读
# ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
#
# # 设置y轴标签字体
# for label in ax1.get_yticklabels():
#     label.set_fontsize(14)
#
# # 第二个图：标准差随难度变化
# for i, col in enumerate(prediction_columns):
#     # 处理NaN值，用0代替以便绘图
#     variance_data = [0 if np.isnan(x) else x for x in variance_results[col]]
#     ax2.bar(x + i * width, variance_data, width, color=method_colors[col], label=method_names[col])
#
# ax2.set_xlabel('Problem Difficulty', fontsize=16)
# ax2.set_ylabel('SDE', fontsize=16)
# ax2.set_title('Variation of SDE with Problem Difficulty in SPF', fontsize=16)
# ax2.set_xticks(x + width * 3)
# ax2.set_xticklabels(range_labels, rotation=45, fontsize=14)
# ax2.legend(fontsize=14)
# ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
#
# # 设置y轴标签字体
# for label in ax2.get_yticklabels():
#     label.set_fontsize(14)

# 第三个图：MSE随难度变化
for i, col in enumerate(prediction_columns):
    # 处理NaN值，用0代替以便绘图
    mse_data = [0 if np.isnan(x) else x for x in mse_results[col]]
    ax3.bar(x + i * width, mse_data, width, color=method_colors[col], label=method_names[col])

ax3.set_xlabel('Problem Difficulty', fontsize=16)
ax3.set_ylabel('MSE', fontsize=16)
ax3.set_title('Variation of MSE with Problem Difficulty in SPF', fontsize=16)
ax3.set_xticks(x + width * 3)
ax3.set_xticklabels(range_labels, rotation=45, fontsize=14)
ax3.legend(fontsize=14)
ax3.grid(True, axis='y', linestyle='--', alpha=0.7)

# 设置y轴标签字体
for label in ax3.get_yticklabels():
    label.set_fontsize(14)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('NCAA_difficulty_SPF.png', dpi=300, bbox_inches='tight')
print("图表已保存为 difficulty_range_analysis.png")

# 打印详细结果
print("\n各预测方法在不同难度区间下的绝对差值均值:")
mean_df = pd.DataFrame(mean_results, index=range_labels)
print(mean_df)

print("\n各预测方法在不同难度区间下的绝对差值标准差:")
variance_df = pd.DataFrame(variance_results, index=range_labels)
print(variance_df)

print("\n各预测方法在不同难度区间下的MSE:")
mse_df = pd.DataFrame(mse_results, index=range_labels)
print(mse_df)

print("\n各难度区间数据点数量:")
count_df = pd.DataFrame({'数据点数量': count_results}, index=range_labels)
print(count_df)