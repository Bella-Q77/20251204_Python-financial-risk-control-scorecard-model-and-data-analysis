# -*- coding: utf-8 -*-
"""
数据质量评估与预处理脚本
执行全面的数据质量评估与预处理操作，包括：
1. 数据基本信息统计
2. 数据质量问题探查（缺失值、类别不平衡、异常值）
3. 数据清洗处理（缺失值、异常值）
4. 结果输出与方法说明

采用的技术方案与项目现有代码存在明显差异：
- 缺失值处理：使用KNN填充和基于树模型的预测填充
- 异常值检测：使用孤立森林算法和IQR扩展法
- 数据质量评估：提供更全面的元数据信息和可视化分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataQualityAssessment:
    def __init__(self, file_path):
        """初始化数据质量评估对象"""
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        self.missing_info = None
        self.outlier_info = None
        
    def load_data(self):
        """加载数据集"""
        print("正在加载数据集...")
        self.df = pd.read_excel(self.file_path)
        print(f"数据集加载完成，包含 {self.df.shape[0]} 个样本和 {self.df.shape[1]} 个特征")
        
    def basic_info_statistics(self):
        """数据基本信息统计"""
        print("\n=== 数据基本信息统计 ===")
        
        # 样本总量和特征数量
        print(f"- 样本总量: {self.df.shape[0]}")
        print(f"- 特征数量: {self.df.shape[1]}")
        
        # 各特征的数据类型和特征含义
        print("\n- 特征数据类型和含义:")
        feature_info = {
            'Account Balance': '账户余额（分类型）',
            'Duration of Credit (month)': '信贷期限（月，数值型）',
            'Payment Status of Previous Credit': ' previous credit的支付状态（分类型）',
            'Purpose': '信贷目的（分类型）',
            'Credit Amount': '信贷金额（数值型）',
            'Value Savings/Stocks': '储蓄/股票价值（分类型）',
            'Length of current employment': '当前就业年限（分类型）',
            'Instalment per cent': '分期付款百分比（分类型）',
            'Sex & Marital Status': '性别与婚姻状况（分类型）',
            'Guarantors': '担保人（分类型）',
            'Duration in Current address': '当前地址居住年限（分类型）',
            'Most valuable available asset': '最有价值的可用资产（分类型）',
            'Age (years)': '年龄（岁，数值型）',
            'Concurrent Credits': '并发信贷（分类型）',
            'Type of apartment': '公寓类型（分类型）',
            'No of Credits at this Bank': '该银行的信贷数量（分类型）',
            'Occupation': '职业（分类型）',
            'No of dependents': '受抚养人数（分类型）',
            'Telephone': '电话（分类型）',
            'Foreign Worker': '外国工人（分类型）',
            'Creditability': '信贷能力（目标变量，0=坏客户，1=好客户）'
        }
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            含义 = feature_info.get(col, '未知')
            print(f"  - {col}: {dtype}，{含义}")
        
        # 目标变量的分布情况
        print("\n- 目标变量分布情况:")
        target_dist = self.df['Creditability'].value_counts(normalize=True)
        print(f"  - 好客户（1）: {target_dist[1]:.2%} ({self.df['Creditability'].value_counts()[1]} 个样本)")
        print(f"  - 坏客户（0）: {target_dist[0]:.2%} ({self.df['Creditability'].value_counts()[0]} 个样本)")
        
        # 数据采集时间范围
        print("\n- 数据采集时间范围: 未知")
        
        # 可视化目标变量分布
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Creditability', data=self.df)
        plt.title('目标变量分布')
        plt.xlabel('信贷能力')
        plt.ylabel('样本数量')
        plt.xticks([0, 1], ['坏客户', '好客户'])
        plt.show()
        
    def missing_value_analysis(self):
        """缺失值分析"""
        print("\n=== 缺失值分析 ===")
        
        # 统计每个特征的缺失值数量及缺失比例
        missing_count = self.df.isnull().sum()
        missing_ratio = self.df.isnull().sum() / self.df.shape[0]
        
        self.missing_info = pd.DataFrame({'缺失值数量': missing_count, '缺失比例': missing_ratio})
        self.missing_info = self.missing_info[self.missing_info['缺失值数量'] > 0]
        
        if self.missing_info.empty:
            print("- 未发现缺失值")
        else:
            print("- 缺失值统计:")
            print(self.missing_info)
            
            # 识别缺失值分布模式
            print("\n- 缺失值分布模式识别:")
            print("  由于样本量有限，无法确定具体的缺失值分布模式（随机缺失、完全随机缺失或系统性缺失）")
            
            # 可视化缺失值分布
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
            plt.title('缺失值分布热力图')
            plt.show()
        
    def class_imbalance_assessment(self):
        """类别不平衡评估"""
        print("\n=== 类别不平衡评估 ===")
        
        # 计算目标变量各类别的样本数量、占比及类别比例差异
        target_counts = self.df['Creditability'].value_counts()
        target_ratio = self.df['Creditability'].value_counts(normalize=True)
        imbalance_ratio = target_ratio[1] / target_ratio[0] if target_ratio[0] != 0 else np.inf
        
        print(f"- 好客户（1）样本数量: {target_counts[1]}")
        print(f"- 坏客户（0）样本数量: {target_counts[0]}")
        print(f"- 好客户占比: {target_ratio[1]:.2%}")
        print(f"- 坏客户占比: {target_ratio[0]:.2%}")
        print(f"- 类别比例差异: {imbalance_ratio:.2f}:1 (好客户:坏客户)")
        
        # 评估是否存在类别不平衡问题
        if imbalance_ratio > 2 or imbalance_ratio < 0.5:
            print("\n- 评估结果: 存在明显的类别不平衡问题")
        else:
            print("\n- 评估结果: 类别分布相对平衡，不存在明显的类别不平衡问题")
        
    def outlier_detection(self):
        """异常值检测"""
        print("\n=== 异常值检测 ===")
        
        # 选择数值型特征
        numeric_features = ['Duration of Credit (month)', 'Credit Amount', 'Age (years)']
        
        self.outlier_info = {}  
        for feature in numeric_features:
            print(f"\n- 特征: {feature}")
            
            # 统计方法检测异常值（IQR扩展法）
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            # 扩展IQR范围，减少误判
            lower_bound = Q1 - 2 * IQR
            upper_bound = Q3 + 2 * IQR
            
            outliers_iqr = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
            
            # 孤立森林算法检测异常值
            iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
            iso_forest.fit(self.df[[feature]])
            
            outlier_scores = iso_forest.decision_function(self.df[[feature]])
            outlier_labels = iso_forest.predict(self.df[[feature]])
            
            outliers_iso = self.df[outlier_labels == -1]
            
            # 合并两种方法的检测结果
            outliers_combined = pd.concat([outliers_iqr, outliers_iso]).drop_duplicates()
            
            self.outlier_info[feature] = {
                'IQR检测异常值数量': len(outliers_iqr),
                '孤立森林检测异常值数量': len(outliers_iso),
                '合并检测异常值数量': len(outliers_combined),
                '异常值样本': outliers_combined
            }
            
            print(f"  - IQR扩展法检测异常值数量: {len(outliers_iqr)}")
            print(f"  - 孤立森林算法检测异常值数量: {len(outliers_iso)}")
            print(f"  - 合并检测异常值数量: {len(outliers_combined)}")
            
            # 可视化异常值分布
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.boxplot(x=self.df[feature])
            plt.title(f'{feature} 箱线图（IQR扩展法）')
            plt.axhline(y=lower_bound, color='r', linestyle='--')
            plt.axhline(y=upper_bound, color='r', linestyle='--')
            
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=self.df.index, y=self.df[feature], hue=outlier_labels, palette={1: 'blue', -1: 'red'})
            plt.title(f'{feature} 散点图（孤立森林算法）')
            plt.legend(title='异常值标签', loc='upper right', labels=['正常', '异常'])
            
            plt.tight_layout()
            plt.show()
            
            # 分析异常值的分布特征及潜在产生原因
            if len(outliers_combined) > 0:
                print(f"  - 异常值分布特征: {feature} 大于 {upper_bound:.2f} 或小于 {lower_bound:.2f}")
                print(f"  - 潜在产生原因: 可能是真实的极端值，也可能是数据录入错误")
        
    def missing_value_handling(self):
        """缺失值处理"""
        print("\n=== 缺失值处理 ===")
        
        # 检查是否存在缺失值
        if self.missing_info is None or self.missing_info.empty:
            print("- 未发现缺失值，无需进行缺失值处理")
            self.df_cleaned = self.df.copy()
            return
        
        # 选择缺失值处理方法
        print("- 采用的缺失值处理方法:")
        print("  1. KNN填充：对于数值型特征，使用KNN算法进行填充")
        print("  2. 基于树模型的预测填充：对于分类型特征，使用决策树分类器进行预测填充")
        
        self.df_cleaned = self.df.copy()
        
        # 处理数值型特征的缺失值（KNN填充）
        numeric_features = self.df.select_dtypes(include=[np.number]).columns
        numeric_features_with_missing = [col for col in numeric_features if col in self.missing_info.index]
        
        if numeric_features_with_missing:
            print(f"\n- 处理数值型特征的缺失值: {numeric_features_with_missing}")
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            
            # 保存非数值型特征
            non_numeric_features = self.df.select_dtypes(exclude=[np.number]).columns
            non_numeric_data = self.df[non_numeric_features]
            
            # 处理数值型特征
            numeric_data = self.df[numeric_features]
            numeric_data_imputed = pd.DataFrame(knn_imputer.fit_transform(numeric_data), columns=numeric_features)
            
            # 合并数据
            self.df_cleaned = pd.concat([numeric_data_imputed, non_numeric_data], axis=1)
            
            # 恢复原始列顺序
            self.df_cleaned = self.df_cleaned[self.df.columns]
            
            print("  - 数值型特征缺失值处理完成")
        
        # 处理分类型特征的缺失值（基于树模型的预测填充）
        categorical_features = self.df.select_dtypes(exclude=[np.number]).columns
        categorical_features_with_missing = [col for col in categorical_features if col in self.missing_info.index]
        
        if categorical_features_with_missing:
            print(f"\n- 处理分类型特征的缺失值: {categorical_features_with_missing}")
            
            for feature in categorical_features_with_missing:
                # 分离包含缺失值的样本和不包含缺失值的样本
                df_missing = self.df_cleaned[self.df_cleaned[feature].isnull()]
                df_not_missing = self.df_cleaned[self.df_cleaned[feature].notnull()]
                
                # 准备特征和目标变量
                X_train = df_not_missing.drop(columns=[feature])
                y_train = df_not_missing[feature]
                X_test = df_missing.drop(columns=[feature])
                
                # 处理X_train和X_test中的缺失值（如果有的话）
                X_train = X_train.fillna(X_train.median())
                X_test = X_test.fillna(X_train.median())
                
                # 训练决策树分类器
                dt_classifier = DecisionTreeClassifier(random_state=42)
                dt_classifier.fit(X_train, y_train)
                
                # 预测缺失值
                y_pred = dt_classifier.predict(X_test)
                
                # 填充缺失值
                self.df_cleaned.loc[self.df_cleaned[feature].isnull(), feature] = y_pred
                
                print(f"  - {feature} 缺失值处理完成")
        
        # 验证缺失值处理结果
        print(f"\n- 缺失值处理结果验证: 剩余缺失值数量为 {self.df_cleaned.isnull().sum().sum()}")
        
    def outlier_handling(self):
        """异常值处理"""
        print("\n=== 异常值处理 ===")
        
        # 检查是否存在异常值
        if self.outlier_info is None or not self.outlier_info:
            print("- 未发现异常值，无需进行异常值处理")
            return
        
        # 选择异常值处理方法
        print("- 采用的异常值处理方法:")
        print("  1. 截断法：对于数值型特征，将异常值截断到合理的范围内")
        print("  2. 不处理：对于可能是真实极端值的异常值，选择不处理")
        
        # 处理数值型特征的异常值
        numeric_features = ['Duration of Credit (month)', 'Credit Amount', 'Age (years)']
        
        for feature in numeric_features:
            if feature not in self.outlier_info:
                continue
            
            print(f"\n- 处理特征: {feature}")
            
            # 计算合理的范围（IQR扩展法）
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 2 * IQR
            upper_bound = Q3 + 2 * IQR
            
            # 截断异常值
            self.df_cleaned[feature] = np.where(self.df_cleaned[feature] < lower_bound, lower_bound, self.df_cleaned[feature])
            self.df_cleaned[feature] = np.where(self.df_cleaned[feature] > upper_bound, upper_bound, self.df_cleaned[feature])
            
            print(f"  - 异常值处理完成，将值截断到 [{lower_bound:.2f}, {upper_bound:.2f}] 范围内")
            
            # 可视化处理后的结果
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.df_cleaned[feature])
            plt.title(f'{feature} 处理后箱线图')
            plt.show()
        
    def compare_before_after(self):
        """比较数据清洗前后的基本信息"""
        print("\n=== 数据清洗前后基本信息对比 ===")
        
        # 样本量变化
        print(f"- 样本量变化: 清洗前 {self.df.shape[0]} 个样本，清洗后 {self.df_cleaned.shape[0]} 个样本")
        
        # 特征完整性
        print(f"- 特征完整性: 清洗前缺失值数量 {self.df.isnull().sum().sum()}，清洗后缺失值数量 {self.df_cleaned.isnull().sum().sum()}")
        
        # 目标变量分布
        print("\n- 目标变量分布对比:")
        target_dist_before = self.df['Creditability'].value_counts(normalize=True)
        target_dist_after = self.df_cleaned['Creditability'].value_counts(normalize=True)
        
        print(f"  - 好客户（1）占比: 清洗前 {target_dist_before[1]:.2%}，清洗后 {target_dist_after[1]:.2%}")
        print(f"  - 坏客户（0）占比: 清洗前 {target_dist_before[0]:.2%}，清洗后 {target_dist_after[0]:.2%}")
        
        # 数值型特征统计分布对比
        numeric_features = ['Duration of Credit (month)', 'Credit Amount', 'Age (years)']
        print("\n- 数值型特征统计分布对比:")
        
        for feature in numeric_features:
            print(f"\n  - 特征: {feature}")
            print(f"    - 均值: 清洗前 {self.df[feature].mean():.2f}，清洗后 {self.df_cleaned[feature].mean():.2f}")
            print(f"    - 中位数: 清洗前 {self.df[feature].median():.2f}，清洗后 {self.df_cleaned[feature].median():.2f}")
            print(f"    - 标准差: 清洗前 {self.df[feature].std():.2f}，清洗后 {self.df_cleaned[feature].std():.2f}")
        
    def save_cleaned_data(self, output_file_path):
        """保存清洗后的数据"""
        print(f"\n- 保存清洗后的数据到: {output_file_path}")
        self.df_cleaned.to_excel(output_file_path, index=False)
        print("  - 数据保存完成")
        
    def run_full_assessment(self):
        """运行完整的数据质量评估与预处理流程"""
        print("=== 开始数据质量评估与预处理流程 ===")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据基本信息统计
        self.basic_info_statistics()
        
        # 3. 数据质量问题探查
        self.missing_value_analysis()
        self.class_imbalance_assessment()
        self.outlier_detection()
        
        # 4. 数据清洗处理
        self.missing_value_handling()
        self.outlier_handling()
        
        # 5. 结果对比
        self.compare_before_after()
        
        # 6. 保存清洗后的数据
        output_file_path = "German_credit_cleaned.xlsx"
        self.save_cleaned_data(output_file_path)
        
        print("\n=== 数据质量评估与预处理流程完成 ===")


# 运行数据质量评估与预处理流程
if __name__ == "__main__":
    file_path = "German_credit.xlsx"
    dqa = DataQualityAssessment(file_path)
    dqa.run_full_assessment()
    
    print("数据质量评估与预处理流程完成。")