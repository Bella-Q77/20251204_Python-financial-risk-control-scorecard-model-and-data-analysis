import pandas as pd
import numpy as np

# 读取数据集
df = pd.read_excel('german_credit.xlsx')

# 输出原始数据集的基本信息
print('原始数据集基本信息:')
print('样本数量:', len(df))
print('特征数量:', len(df.columns))
print('特征类型:')
print(df.dtypes)
print('各特征的统计描述:')
print(df.describe())
print('目标变量分布情况:')
print(df['target'].value_counts())

# 检查缺失值
print('\n缺失值检查:')
print(df.isnull().sum())

# 检查异常值
print('\n异常值检查:')
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        # 使用IQR法检测异常值
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f'{col}的异常值数量: {len(outliers)}')

# 正负样本均衡性判断
print('\n正负样本均衡性判断:')
target_counts = df['target'].value_counts()
positive_ratio = target_counts[1] / len(df)
negative_ratio = target_counts[0] / len(df)
print(f'正样本比例: {positive_ratio:.2%}')
print(f'负样本比例: {negative_ratio:.2%}')
print(f'不平衡系数: {max(positive_ratio, negative_ratio) / min(positive_ratio, negative_ratio):.2f}')

# 缺失值处理：使用中位数插补（与原项目不同）
print('\n缺失值处理:')
print('处理方法: 使用中位数插补所有数值型特征的缺失值')
print('原因: 中位数不受极端值影响，比均值更稳健')
for col in df.columns:
    if df[col].dtype in ['int64', 'float64'] and df[col].isnull().sum() > 0:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
        print(f'{col}缺失值已用中位数{median:.2f}插补')

# 异常值处理：使用Winsorization截断（与原项目不同）
print('\n异常值处理:')
print('处理方法: 使用Winsorization截断法将异常值替换为上下限')
print('原因: 截断法可以保留数据分布，同时消除极端值的影响')
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        print(f'{col}异常值已用Winsorization处理')

# 保存处理后的数据集
df.to_excel('processed_german_credit.xlsx', index=False)
print('\n处理后的数据集已保存为processed_german_credit.xlsx')

# 输出处理后的数据集基本信息
print('\n处理后的数据集基本信息:')
print('样本数量:', len(df))
print('特征数量:', len(df.columns))
print('各特征的统计描述:')
print(df.describe())
print('目标变量分布情况:')
print(df['target'].value_counts())