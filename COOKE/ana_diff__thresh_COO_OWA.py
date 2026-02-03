import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
df = pd.read_excel('result_Cooke_OWA.xlsx')
df = df[df["实际值"]>=0.1]
# 定义难度档位
difficulty_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 定义预测方法列名
prediction_columns = ['D_COR_OWA', '平均值', '中位数', '权值最大', 'OWA加权', 'SNR加权']

# 存储结果的字典
mean_results = {col: [] for col in prediction_columns}
variance_results = {col: [] for col in prediction_columns}

# 对每个难度档位进行计算
for level in difficulty_levels:
    # 筛选问题难度大于当前档位的数据
    filtered_df = df[df['问题难度'] > level]

    # 对每个预测方法计算与实际值的绝对差值
    for col in prediction_columns:
        differences = abs(filtered_df[col] - filtered_df['实际值'])/filtered_df['实际值']

        # 计算均值和方差
        mean_val = differences.mean()
        variance_val = differences.var()

        # 存储结果
        mean_results[col].append(mean_val + np.sqrt(variance_val))
        variance_results[col].append(variance_val)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 第一个图：均值随难度变化
x = np.arange(len(difficulty_levels))
width = 0.15  # 柱状图宽度

# 为每个预测方法绘制柱状图
for i, col in enumerate(prediction_columns):
    ax1.bar(x + i * width, mean_results[col], width, label=col)

ax1.set_xlabel('难度档位')
ax1.set_ylabel('绝对差值均值')
ax1.set_title('各预测方法绝对差值均值随问题难度变化')
ax1.set_xticks(x + width * 2)
ax1.set_xticklabels([f'>{level}' for level in difficulty_levels])
ax1.legend()

# 第二个图：方差随难度变化
for i, col in enumerate(prediction_columns):
    ax2.bar(x + i * width, variance_results[col], width, label=col)

ax2.set_xlabel('难度档位')
ax2.set_ylabel('方差')
ax2.set_title('各预测方法绝对差值方差随问题难度变化')
ax2.set_xticks(x + width * 2)
ax2.set_xticklabels([f'>{level}' for level in difficulty_levels])
ax2.legend()

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('difficulty_thresh_analysis.png', dpi=300, bbox_inches='tight')
print("图表已保存为 difficulty_analysis.png")

# 打印详细结果
print("\n各预测方法在不同难度档位下的绝对差值均值:")
mean_df = pd.DataFrame(mean_results, index=[f'>{level}' for level in difficulty_levels])
print(mean_df)

print("\n各预测方法在不同难度档位下的绝对差值方差:")
variance_df = pd.DataFrame(variance_results, index=[f'>{level}' for level in difficulty_levels])
print(variance_df)