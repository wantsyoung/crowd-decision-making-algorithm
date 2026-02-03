import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置matplotlib后端为Agg，避免显示问题
plt.switch_backend('Agg')

# 设置字体为Times New Roman并增大字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也设为Times New Roman风格
plt.rcParams['font.size'] = 14  # 增大全局字体大小

# 1. 读取之前生成的结果文件
results_df = pd.read_excel('result_GDP_OWA_varying_n.xlsx')

# 2. 过滤数据，只保留决策者数量在4到50之间的数据
results_df = results_df[(results_df['决策者数量'] >= 4) & (results_df['决策者数量'] <= 55)]

# 3. 计算各种方法的误差（如果结果文件中还没有计算误差）
if 'D_COR_OWA误差' not in results_df.columns:
    results_df['D_COR_OWA误差'] = np.abs(results_df['D_COR_OWA'] - results_df['实际值'])
    results_df['平均值误差'] = np.abs(results_df['平均值'] - results_df['实际值'])
    results_df['中位数误差'] = np.abs(results_df['中位数'] - results_df['实际值'])
    results_df['权值最大误差'] = np.abs(results_df['权值最大'] - results_df['实际值'])
    results_df['OWA加权误差'] = np.abs(results_df['OWA加权'] - results_df['实际值'])
    results_df['SNR加权误差'] = np.abs(results_df['SNR加权'] - results_df['实际值'])

# 4. 生成汇总统计结果（按决策者数量）
n_range = sorted(results_df['决策者数量'].unique())
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
print(f"决策者数量范围: {min(n_range)} - {max(n_range)}")
print(f"有效的决策者数量值: {n_range}")

# 5. 绘制三张并列图表
# 创建图表 - 三张子图
fig, (ax3) = plt.subplots(1, 1, figsize=(6, 6))
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
# 定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 第一张图：平均误差随决策者数量变化
x = summary_df['Number of decision makers']
# for i, (method_name, error_col) in enumerate(methods.items()):
#     ax1.plot(x, summary_df[f'{method_name}_mean_error'],
#              marker='o', linewidth=2, label=method_name, color=colors[i])
#
# ax1.set_xlabel('Number of decision makers', fontsize=16)
# ax1.set_ylabel('Mean of error', fontsize=16)
# ax1.set_title('Mean of error vs number of decision makers', fontsize=16)
# ax1.legend(fontsize=14)
# ax1.grid(True, linestyle='--', alpha=0.7)
# # 设置x轴范围
# ax1.set_xlim(4, 55)
#
# # 设置x轴和y轴标签字体
# for label in ax1.get_xticklabels():
#     label.set_fontsize(14)
# for label in ax1.get_yticklabels():
#     label.set_fontsize(14)
#
# # 第二张图：标准差随决策者数量变化
# for i, (method_name, error_col) in enumerate(methods.items()):
#     ax2.plot(x, summary_df[f'{method_name}_std'],
#              marker='s', linewidth=2, label=method_name, color=colors[i])
#
# ax2.set_xlabel('Number of decision makers', fontsize=16)
# ax2.set_ylabel('STD of error', fontsize=16)
# ax2.set_title('STD of error vs number of decision makers', fontsize=16)
# ax2.legend(fontsize=14)
# ax2.grid(True, linestyle='--', alpha=0.7)
# # 设置x轴范围
# ax2.set_xlim(4, 55)
#
# # 设置x轴和y轴标签字体
# for label in ax2.get_xticklabels():
#     label.set_fontsize(14)
# for label in ax2.get_yticklabels():
#     label.set_fontsize(14)

# 第三张图：MSE随决策者数量变化
for i, (method_name, error_col) in enumerate(methods.items()):
    ax3.plot(x, summary_df[f'{method_name}_MSE'],
             marker='^', linewidth=2, label=method_name, color=colors[i])

ax3.set_xlabel('Number of Decision Makers', fontsize=16)
ax3.set_ylabel('MSE', fontsize=16)
ax3.set_title('Variation of MSE with Number of Decision Makers in SPF', fontsize=16)
ax3.legend(fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.7)
# 设置x轴范围
ax3.set_xlim(4, 55)

# 设置x轴和y轴标签字体
for label in ax3.get_xticklabels():
    label.set_fontsize(14)
for label in ax3.get_yticklabels():
    label.set_fontsize(14)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('NCAA_numvaring_SPF.png', dpi=300, bbox_inches='tight')
print("图表已保存为 numvaring_GDP.png")
print(f"图表中显示的决策者数量范围: 4-55")