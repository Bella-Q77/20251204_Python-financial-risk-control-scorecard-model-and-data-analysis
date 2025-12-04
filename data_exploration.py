import pandas as pd

# 读取数据集
df = pd.read_excel('20251204_Python-financial-risk-control-scorecard-model-and-data-analysis/german_credit.xlsx')

# 输出基本信息
print('样本总量:', len(df))
print('特征数量:', len(df.columns)-1)
print('目标变量:', df.columns[-1])
print('\n特征数据类型:\n', df.dtypes)
print('\n前5行数据:\n', df.head())
print('\n目标变量分布:\n', df[df.columns[-1]].value_counts())
