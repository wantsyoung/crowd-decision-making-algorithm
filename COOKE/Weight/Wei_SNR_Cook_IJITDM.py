import pandas as pd
import numpy as np
import os

# 设置文件夹路径
folder_path = "../DATA_COOKE"

# 获取文件夹中所有Excel文件，排除临时文件
excel_files = [f for f in os.listdir(folder_path)
               if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]

# 创建一个空的DataFrame来存储所有结果
all_results = pd.DataFrame()


# 归一化函数
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.full_like(data, 0.5)
    return (data - min_val) / (max_val - min_val)


for file_name in excel_files:
    file_path = os.path.join(folder_path, file_name)

    # 读取Excel文件
    df = pd.read_excel(file_path, index_col=0, engine='openpyxl')

    # 分离正确答案和专家答案
    correct_answers = df.iloc[-1]  # 最后一行是正确答案
    expert_answers = df.iloc[:-1]  # 除了最后一行都是专家答案

    # 对每个问题进行归一化
    normalized_expert_answers = pd.DataFrame(index=expert_answers.index, columns=expert_answers.columns)
    normalized_correct_answers = pd.Series(index=expert_answers.columns)

    for problem in expert_answers.columns:
        # 获取该问题的所有专家答案和正确答案
        problem_data = expert_answers[problem].values.astype(float)
        correct_answer = float(correct_answers[problem])

        # 合并专家答案和正确答案进行归一化
        all_data = np.append(problem_data, correct_answer)
        normalized_data = normalize_data(all_data)

        # 分离归一化后的数据
        normalized_expert_answers[problem] = normalized_data[:-1]  # 专家答案
        normalized_correct_answers[problem] = normalized_data[-1]  # 正确答案

    # 计算每个专家的指标
    expert_snrs = {}

    for expert in normalized_expert_answers.index:
        # 获取专家的归一化答案向量
        expert_vector = normalized_expert_answers.loc[expert].values.astype(float)
        correct_vector = normalized_correct_answers.values.astype(float)

        # 计算平均偏差
        bias = np.mean(expert_vector - correct_vector)

        # 计算方差
        errors = (expert_vector - correct_vector) - bias
        variance = np.mean(errors ** 2)

        # 计算均方误差
        mse = bias ** 2 + variance

        # 计算信噪比
        snr = 1.0 / mse if mse != 0 else float('inf')

        expert_snrs[expert] = snr

    # 对SNR进行归一化得到权重
    total_snr = sum(expert_snrs.values())
    expert_weights = {expert: snr / total_snr for expert, snr in expert_snrs.items()}

    # 将当前文件的结果添加到总结果中
    file_results_df = pd.DataFrame.from_dict(expert_weights, orient='index', columns=[file_name])
    if all_results.empty:
        all_results = file_results_df
    else:
        all_results = all_results.join(file_results_df, how='outer')


# 按专家编号排序（按数字顺序）
def extract_expert_number(expert_name):
    return int(expert_name.replace('EXPERT', ''))


sorted_index = sorted(all_results.index, key=extract_expert_number)
all_results = all_results.loc[sorted_index]

# 保存到Excel文件
all_results.to_excel("Wei_SNR_Cooke.xlsx")

print("处理完成！结果已保存到 Wei_SNR_Cooke.xlsx")