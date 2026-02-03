# 衡量算法有效性时，考察绝对误差，非相对误差
# 画图时，画出各算法的均方误差，非均值或方差
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib后端为Agg，避免显示问题
plt.switch_backend('Agg')

# 读取Excel文件
df = pd.read_excel('result_Cooke_OWA_clean.xlsx')

# 过滤掉实际值小于0.01的数据行
df = df[df['实际值'] >= 0]

# 定义需要计算差值的列名
prediction_columns = [
    'D_COR_OWA',
    '平均值',
    '中位数',
    '权值最大',
    'OWA加权', 'SNR加权'
]

# 计算绝对差值
differences = {}
for col in prediction_columns:
    differences[col] = abs(df[col] - df['实际值'])

# 转换为DataFrame便于分析
diff_df = pd.DataFrame(differences)

# 计算平均绝对误差、标准差和均方误差
mae_values = diff_df.mean()  # 平均绝对误差
std_values = diff_df.std()   # 标准差
mse_values = (diff_df ** 2).mean()  # 均方误差

# 设置字体为Times New Roman并增大字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也设为Times New Roman风格
plt.rcParams['font.size'] = 14  # 增大全局字体大小

# 创建图表 - 三张子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
# fig, (ax3) = plt.subplots(1, 1, figsize=(6, 6))
# 定义方法名称的英文映射
method_names = {
    'D_COR_OWA': 'NCAA',
    '平均值': 'Averaging',
    '中位数': 'Median',
    '权值最大': 'Dictator',
    'OWA加权': 'OWA',
    'SNR加权': 'SNR'
}

# 定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
x_pos = np.arange(len(mae_values))

# 第一张图：平均绝对误差柱状图

bars1 = ax1.bar(x_pos, mae_values.values, color=colors, alpha=0.7)
ax1.set_title('Mean Absolute Error for All Methods in SEJ', fontsize=16)
# ax1.set_xlabel('Aggregating Methods', fontsize=16)
ax1.set_ylabel('Mean Absolute Error', fontsize=16)
ax1.set_xticks(x_pos)
ax1.set_ylim(0.2, 0.3)
ax1.set_xticklabels([method_names[col] for col in mae_values.index], rotation=45, ha='right', fontsize=14)
# 添加网格线
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# 设置y轴标签字体
for label in ax1.get_yticklabels():
    label.set_fontsize(14)

# 在柱子上添加数值标签
for bar, value in zip(bars1, mae_values.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontsize=12)

# 第二张图：标准差柱状图
bars2 = ax2.bar(x_pos, std_values.values, color=colors, alpha=0.7)
ax2.set_title('Standard Deviation of Error for All Methods in SEJ', fontsize=16)
ax2.set_xlabel('Aggregating Methods', fontsize=16)
ax2.set_ylabel('Standard Deviation of Error', fontsize=16)
ax2.set_xticks(x_pos)
ax2.set_ylim(0.2, 0.3)
ax2.set_xticklabels([method_names[col] for col in std_values.index], rotation=45, ha='right', fontsize=14)
# 添加网格线
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

# 设置y轴标签字体
for label in ax2.get_yticklabels():
    label.set_fontsize(14)

# 在柱子上添加数值标签
for bar, value in zip(bars2, std_values.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontsize=12)

# 第三张图：均方误差柱状图
bars3 = ax3.bar(x_pos, mse_values.values, color=colors, alpha=0.7)
ax3.set_title('Mean Squared Error for All Methods in SEJ', fontsize=16)
# ax3.set_xlabel('Aggregating Methods', fontsize=16)
ax3.set_ylabel('Mean Squared Error', fontsize=16)
ax3.set_xticks(x_pos)
ax3.set_ylim(0.1, 0.15)
ax3.set_xticklabels([method_names[col] for col in mse_values.index], rotation=45, ha='right', fontsize=14)
# 添加网格线
ax3.grid(True, axis='y', linestyle='--', alpha=0.7)

# 设置y轴标签字体
for label in ax3.get_yticklabels():
    label.set_fontsize(14)

# 在柱子上添加数值标签
for bar, value in zip(bars3, mse_values.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontsize=12)

# 调整布局
plt.tight_layout()

# 保存图表而不是显示
plt.savefig('NCAA_overall_SEJ.png', dpi=300, bbox_inches='tight')
print("图表已保存为 total_analysis.png")

# 打印统计结果
print(f"\n过滤后剩余数据行数: {len(df)}")
print("\n各预测方法与实际值的绝对误差统计：")
print("\n平均绝对误差：")
print(mae_values)
print("\n标准差：")
print(std_values)
print("\n均方误差：")
print(mse_values)