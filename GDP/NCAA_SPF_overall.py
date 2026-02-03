import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib后端为Agg，避免显示问题
plt.switch_backend('Agg')

# 读取Excel文件
df = pd.read_excel('result_GDP_OWA_normalized.xlsx')

# 定义需要计算差值的列名
prediction_columns = ['D_COR_OWA', '平均值', '中位数', '权值最大', 'OWA加权', 'SNR加权']

# 计算绝对差值
differences = {}
for col in prediction_columns:
    differences[col] = abs(df[col] - df['实际值'])

# 转换为DataFrame便于分析
diff_df = pd.DataFrame(differences)

# 计算均值和方差
means = diff_df.mean()
variances = diff_df.var()
# 计算均值的平方+方差（MSE）
mse_values = means ** 2 + variances

# 设置字体为Times New Roman并增大字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也设为Times New Roman风格
plt.rcParams['font.size'] = 14  # 增大全局字体大小

# 创建图表 - 现在有两张子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 定义方法名称的英文映射
method_names = {
    'D_COR_OWA': 'NCAA',
    '平均值': 'Averaging',
    '中位数': 'Median',
    '权值最大': 'Dictator',
    'OWA加权': 'OWA',
    'SNR加权': 'SNR'
}

# 第一张图：绘制均值柱状图（暂时注释掉）
x_pos = np.arange(len(means))
bars1 = ax1.bar(x_pos, means.values, color=colors, alpha=0.7)
ax1.set_title('Mean Absolute Error for All in SPF', fontsize=16)  # 增大标题字体
# ax1.set_xlabel('Aggregating Methods', fontsize=16)  # 增大坐标轴标签字体
ax1.set_ylabel('Mean Absolute Error', fontsize=16)
ax1.set_xticks(x_pos)
ax1.set_ylim(0.3, 0.6)
# 使用英文标签并增大字体
ax1.set_xticklabels([method_names[col] for col in means.index], rotation=45, ha='right', fontsize=14)
# 添加网格线
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# 设置y轴标签字体
for label in ax1.get_yticklabels():
    label.set_fontsize(14)

# 在柱子上添加数值标签并增大字体
for bar, value in zip(bars1, means.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontsize=12)

# 第二张图：绘制方差柱状图（暂时注释掉）
bars2 = ax2.bar(x_pos, variances.values, color=colors, alpha=0.7)
ax2.set_title('Standard Deviation of Error for All Methods in SPF', fontsize=16)
ax2.set_xlabel('Aggregating Methods', fontsize=16)
ax2.set_ylabel('Standard Deviation of Error', fontsize=16)
ax2.set_xticks(x_pos)
# 使用英文标签并增大字体
ax2.set_xticklabels([method_names[col] for col in variances.index], rotation=45, ha='right', fontsize=14)
# 添加网格线
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
ax2.set_ylim(0.2, 0.3)
# 设置y轴标签字体
for label in ax2.get_yticklabels():
    label.set_fontsize(14)

# 在柱子上添加数值标签并增大字体
for bar, value in zip(bars2, variances.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontsize=12)

# 第三张图：绘制MSE（均值的平方+方差）柱状图
x_pos = np.arange(len(mse_values))
bars3 = ax3.bar(x_pos, mse_values.values, color=colors, alpha=0.7)
ax3.set_title('Mean Squared Error for All Methods in SPF', fontsize=16)
# ax3.set_xlabel('Aggregating Methods', fontsize=16)
ax3.set_ylabel('Mean Squared Error', fontsize=16)
ax3.set_xticks(x_pos)
# 使用英文标签并增大字体
ax3.set_xticklabels([method_names[col] for col in mse_values.index], rotation=45, ha='right', fontsize=14)
# 添加网格线
ax3.grid(True, axis='y', linestyle='--', alpha=0.7)

# 设置第三张图的纵坐标范围为[0.2, 0.8]
ax3.set_ylim(0.2, 0.7)

# 设置y轴标签字体
for label in ax3.get_yticklabels():
    label.set_fontsize(14)

# 在柱子上添加数值标签并增大字体
for bar, value in zip(bars3, mse_values.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontsize=12)

# # 第四张图：以D_COR_OWA为基准，计算其他方法的百分比变化
# baseline_method = 'D_COR_OWA'
# baseline_mse = mse_values[baseline_method]
#
# # 计算其他方法相对于基准的百分比变化
# comparison_methods = [col for col in mse_values.index if col != baseline_method]
# percentage_changes = []
#
# for method in comparison_methods:
#     change = ((mse_values[method] - baseline_mse) / baseline_mse) * 100
#     percentage_changes.append(change)
#
# # 绘制百分比变化柱状图
# x_pos_comp = np.arange(len(comparison_methods))


# 调整布局
plt.tight_layout()

# 保存图表而不是显示
plt.savefig('NCAA_overall_SPF.png', dpi=300, bbox_inches='tight')
print("图表已保存为 overall_GDP.png")

# 打印统计结果
print("\n各预测方法与实际值的绝对差值统计：")
print("\n均值：")
print(means)
print("\n方差：")
print(variances)
print("\nMSE：")
print(mse_values)