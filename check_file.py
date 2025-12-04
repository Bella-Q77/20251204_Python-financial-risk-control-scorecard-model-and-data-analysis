import os
import pandas as pd

# 检查文件是否存在
file_path = '20251204_Python-financial-risk-control-scorecard-model-and-data-analysis/german_credit.xlsx'
print('文件路径:', file_path)
print('文件是否存在:', os.path.exists(file_path))

# 检查目录内容
print('\n目录内容:')
for file in os.listdir('20251204_Python-financial-risk-control-scorecard-model-and-data-analysis'):
    print(file)

# 尝试读取数据
try:
    df = pd.read_excel(file_path)
    print('\n数据读取成功!')
    print('样本数量:', len(df))
    print('特征数量:', len(df.columns))
    print('列名:', df.columns.tolist())
except Exception as e:
    print('\n数据读取失败:', str(e))
