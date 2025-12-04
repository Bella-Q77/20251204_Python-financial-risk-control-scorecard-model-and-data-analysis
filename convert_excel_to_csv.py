import pandas as pd

# 读取Excel文件
df = pd.read_excel('20251204_Python-financial-risk-control-scorecard-model-and-data-analysis/german_credit.xlsx')

# 将数据保存为CSV文件
df.to_csv('20251204_Python-financial-risk-control-scorecard-model-and-data-analysis/german_credit.csv', index=False)

print('Excel文件已成功转换为CSV文件')
