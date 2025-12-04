# -*- coding: utf-8 -*-
"""
数据质量评估与预处理脚本
对german_credit.xlsx数据集进行全面的数据质量评估和预处理
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据基本信息统计
print("="*50)
print("1. 数据基本信息统计")
print("="*50)

# 读取数据集
df = pd.read_excel('german_credit.xlsx')
print(f"数据集样本总量: {df.shape[0]}")
print(f"数据集特征数量: {df.shape[1]}")
print("\n各特征的数据类型:")
print(df.dtypes)

# 特征含义说明（根据UCI德国信用数据集文档）
feature_descriptions = {
    'Account Balance': '账户余额',
    'Duration of Credit (month)': '信贷期限（月）',
    'Payment Status of Previous Credit': '之前信贷的支付状态',
    'Purpose': '信贷目的',
    'Credit Amount': '信贷金额',
    'Value Savings/Stocks': '储蓄/股票价值',
    'Length of current employment': '当前就业年限',
    'Instalment per cent': '分期付款百分比',
    'Sex & Marital Status': '性别与婚姻状况',
    'Guarantors': '担保人',
    'Duration in Current address': '当前地址居住年限',
    'Most valuable available asset': '最有价值的可用资产',
    'Age (years)': '年龄（岁）',
    'Concurrent Credits': '同时期的信贷数量',
    'Type of apartment': '公寓类型',
    'No of Credits at this Bank': '在该银行的信贷数量',
    'Occupation': '职业',
    'No of dependents': '受抚养人数',
    'Telephone': '是否有电话',
    'Foreign Worker': '是否为外籍工人',
    'Creditability': '信用度（目标变量，1=好，0=坏）'
}

print("\n特征含义说明:")
for feature, desc in feature_descriptions.items():
    print(f"  {feature}: {desc}")

# 目标变量分布
print("\n目标变量分布:")
print(df['Creditability'].value_counts())
print(df['Creditability'].value_counts(normalize=True))

# 2. 数据质量问题探查
print("\n" + "="*50)
print("2. 数据质量问题探查")
print("="*50)

# 缺失值分析
print("\n缺失值分析:")
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'缺失值数量': missing_values, '缺失比例(%)': missing_percentage})
missing_df = missing_df[missing_df['缺失值数量'] > 0].sort_values('缺失比例(%)', ascending=False)
print(missing_df)

# 可视化缺失值分布
msno.matrix(df)
plt.title('缺失值分布矩阵')
plt.savefig('missing_values_matrix.png')
plt.close()

msno.bar(df)
plt.title('缺失值比例柱状图')
plt.savefig('missing_values_bar.png')
plt.close()

# 类别不平衡评估
print("\n类别不平衡评估:")
class_counts = df['Creditability'].value_counts()
class_ratio = class_counts[1] / class_counts[0] if class_counts[0] != 0 else np.inf
print(f"好信用样本数量: {class_counts[1]}, 占比: {class_counts[1]/len(df)*100:.2f}%")
print(f"坏信用样本数量: {class_counts[0]}, 占比: {class_counts[0]/len(df)*100:.2f}%")
print(f"类别比例(好/坏): {class_ratio:.2f}:1")

# 异常值检测（数值型特征）
print("\n异常值检测:")
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('Creditability')  # 移除目标变量

# 使用孤立森林算法检测异常值
scaler = StandardScaler()
numeric_data = scaler.fit_transform(df[numeric_features])

iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
anomaly_scores = iso_forest.fit_predict(numeric_data)

# 统计异常值数量
anomaly_count = sum(anomaly_scores == -1)
normal_count = sum(anomaly_scores == 1)
print(f"孤立森林检测到的异常值数量: {anomaly_count}, 占比: {anomaly_count/len(df)*100:.2f}%")
print(f"正常样本数量: {normal_count}, 占比: {normal_count/len(df)*100:.2f}%")

# 可视化异常值分布
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_features])
plt.title('数值型特征箱线图（显示异常值）')
plt.xticks(rotation=45)
plt.savefig('numeric_features_boxplot.png')
plt.close()

# 3. 数据清洗处理
print("\n" + "="*50)
print("3. 数据清洗处理")
print("="*50)

# 保存原始数据副本
df_original = df.copy()

# 缺失值处理：使用KNN填充（与项目现有均值填充不同）
print("\n缺失值处理 - KNN填充:")
# 分离数值型和分类型特征
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

# 对数值型特征进行KNN填充
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_features] = knn_imputer.fit_transform(df[numeric_features])

# 验证缺失值处理结果
print("缺失值处理后各特征缺失值数量:")
print(df.isnull().sum())

# 异常值处理：使用孤立森林检测并剔除异常值（与项目现有方法不同）
print("\n异常值处理 - 孤立森林检测并剔除:")
# 重新训练孤立森林模型
numeric_data_clean = scaler.fit_transform(df[numeric_features])
anomaly_scores_clean = iso_forest.fit_predict(numeric_data_clean)

# 剔除异常值
df_clean = df[anomaly_scores_clean == 1].reset_index(drop=True)
print(f"剔除异常值后样本数量: {len(df_clean)}")
print(f"剔除的异常值数量: {len(df) - len(df_clean)}")

# 4. 结果输出与方法说明
print("\n" + "="*50)
print("4. 结果输出与方法说明")
print("="*50)

# 数据清洗前后对比
print("\n数据清洗前后对比:")
comparison = pd.DataFrame({
    '指标': ['样本数量', '特征数量', '缺失值总数', '异常值比例'],
    '清洗前': [len(df_original), len(df_original.columns), df_original.isnull().sum().sum(), f"{anomaly_count/len(df_original)*100:.2f}%"],
    '清洗后': [len(df_clean), len(df_clean.columns), df_clean.isnull().sum().sum(), "0%"]
})
print(comparison)

# 方法说明
print("\n方法说明:")
print("1. 缺失值处理方法 - KNN填充:")
print("   - 选择原因：KNN填充利用特征之间的相似性进行填充，比简单的均值/中位数填充更准确")
print("   - 优势：考虑了特征之间的相关性，适用于各种类型的缺失数据")
print("   - 参数设置：n_neighbors=5（使用5个最近邻样本进行填充）")

print("\n2. 异常值检测算法 - 孤立森林:")
print("   - 选择原因：孤立森林是一种高效的异常值检测算法，适用于高维数据")
print("   - 优势：不需要假设数据分布，检测速度快，对异常值敏感")
print("   - 参数设置：n_estimators=100（使用100棵树），contamination='auto'（自动检测异常值比例）")

print("\n3. 类别不平衡评估指标:")
print("   - 样本数量和占比：直接展示各类别的样本分布")
print("   - 类别比例：好信用与坏信用样本的比例，评估不平衡程度")

# 保存清洗后的数据
df_clean.to_excel('german_credit_cleaned.xlsx', index=False)
print("\n数据清洗完成！清洗后的数据已保存为'german_credit_cleaned.xlsx'")

# 保存评估报告
with open('data_quality_report.txt', 'w', encoding='utf-8') as f:
    f.write("数据质量评估报告\n")
    f.write("="*50 + "\n")
    f.write("1. 数据基本信息统计\n")
    f.write(f"数据集样本总量: {df.shape[0]}\n")
    f.write(f"数据集特征数量: {df.shape[1]}\n")
    f.write("\n各特征的数据类型:\n")
    f.write(str(df.dtypes) + "\n")
    f.write("\n2. 数据质量问题探查\n")
    f.write("\n缺失值分析:\n")
    f.write(str(missing_df) + "\n")
    f.write("\n类别不平衡评估:\n")
    f.write(f"好信用样本数量: {class_counts[1]}, 占比: {class_counts[1]/len(df)*100:.2f}%\n")
    f.write(f"坏信用样本数量: {class_counts[0]}, 占比: {class_counts[0]/len(df)*100:.2f}%\n")
    f.write(f"类别比例(好/坏): {class_ratio:.2f}:1\n")
    f.write("\n异常值检测:\n")
    f.write(f"孤立森林检测到的异常值数量: {anomaly_count}, 占比: {anomaly_count/len(df)*100:.2f}%\n")
    f.write("\n3. 数据清洗处理\n")
    f.write("\n数据清洗前后对比:\n")
    f.write(str(comparison) + "\n")
    f.write("\n4. 方法说明\n")
    f.write("缺失值处理方法 - KNN填充:\n")
    f.write("   - 选择原因：KNN填充利用特征之间的相似性进行填充，比简单的均值/中位数填充更准确\n")
    f.write("   - 优势：考虑了特征之间的相关性，适用于各种类型的缺失数据\n")
    f.write("   - 参数设置：n_neighbors=5（使用5个最近邻样本进行填充）\n")
    f.write("\n异常值检测算法 - 孤立森林:\n")
    f.write("   - 选择原因：孤立森林是一种高效的异常值检测算法，适用于高维数据\n")
    f.write("   - 优势：不需要假设数据分布，检测速度快，对异常值敏感\n")
    f.write("   - 参数设置：n_estimators=100（使用100棵树），contamination='auto'（自动检测异常值比例）\n")

print("数据质量评估报告已保存为'data_quality_report.txt'")