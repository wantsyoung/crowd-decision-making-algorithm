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

for file_name in excel_files:
    file_path = os.path.join(folder_path, file_name)

    # 读取Excel文件
    df = pd.read_excel(file_path, index_col=0, engine='openpyxl')

    # 分离正确答案和专家答案
    correct_answers = df.iloc[-1]  # 最后一行是正确答案
    expert_answers = df.iloc[:-1]  # 除了最后一行都是专家答案

    # 初始化专家权值累加器
    expert_weights_accumulator = {expert: [] for expert in expert_answers.index}

    # 遍历每个问题
    for problem in expert_answers.columns:
        # 获取该问题的正确答案和专家答案
        correct_answer = correct_answers[problem]
        problem_data = expert_answers[problem]

        # 计算每个专家的偏差
        deviations = {}
        for expert in expert_answers.index:
            expert_answer = problem_data[expert]
            deviation = abs(expert_answer - correct_answer)
            deviations[expert] = deviation

        # 按偏差从大到小排序
        sorted_deviations = sorted(deviations.items(), key=lambda x: x[1], reverse=True)

        # 分配权值
        N = len(sorted_deviations)
        for rank, (expert, deviation) in enumerate(sorted_deviations):
            if N <= 2:
                if rank == 0:
                    weight = 0
                else:
                    weight = 1
            else:
                if rank == 0:
                    weight = 0
                elif rank == N - 1:
                    weight = 1
                else:
                    weight = rank * (1.0 / (N - 2))

            expert_weights_accumulator[expert].append(weight)

    # 计算每个专家的平均权值
    expert_weights = {}
    for expert, weights_list in expert_weights_accumulator.items():
        avg_weight = np.mean(weights_list)
        expert_weights[expert] = avg_weight

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
all_results.to_excel("Wei_OWA_Cooke.xlsx")

print("处理完成！结果已保存到 Wei_OWA_Cooke.xlsx")