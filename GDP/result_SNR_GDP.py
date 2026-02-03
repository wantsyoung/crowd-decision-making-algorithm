import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib后端为Agg，避免显示问题
plt.switch_backend('Agg')

# 读取Excel文件
df = pd.read_excel('result_SNR_GDP.xlsx')

# 定义需要计算差值的列名
prediction_columns = [
    'D_COR_SNR',
    '平均值',
    '中位数',
    '权值最大',
    'SNR加权'
]

# 计算绝对差值
differences = {}
for col in prediction_columns:
    differences[col] = abs(df[col] - df['实际值'])/df['实际值']

# 转换为DataFrame便于分析
diff_df = pd.DataFrame(differences)

# 计算均值和方差
means = diff_df.mean()
variances = diff_df.var()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 绘制均值柱状图
x_pos = np.arange(len(means))
bars1 = ax1.bar(x_pos, means.values, color='skyblue', alpha=0.7)
ax1.set_title('各预测方法与实际值的绝对差值均值')
ax1.set_xlabel('预测方法')
ax1.set_ylabel('绝对差值均值')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(means.index, rotation=45, ha='right')

# 在柱子上添加数值标签
for bar, value in zip(bars1, means.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{value:.4f}', ha='center', va='bottom')

# 绘制方差柱状图
bars2 = ax2.bar(x_pos, variances.values, color='lightcoral', alpha=0.7)
ax2.set_title('各预测方法与实际值的绝对差值的方差')
ax2.set_xlabel('预测方法')
ax2.set_ylabel('方差')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(variances.index, rotation=45, ha='right')

# 在柱子上添加数值标签
for bar, value in zip(bars2, variances.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{value:.4f}', ha='center', va='bottom')

# 调整布局
plt.tight_layout()

# 保存图表而不是显示
plt.savefig('difference_analysis.png', dpi=300, bbox_inches='tight')
print("图表已保存为 difference_analysis.png")

# 打印统计结果
print("\n各预测方法与实际值的绝对差值统计：")
print("\n均值：")
print(means)
print("\n方差：")
print(variances)