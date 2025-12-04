import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest



# 读取数据集
df = pd.read_excel('20251204_Python-financial-risk-control-scorecard-model-and-data-analysis/german_credit.xlsx')

# 1. Data Basic Information Statistics
print('='*50)
print('1. Data Basic Information Statistics')
print('='*50)
print(f'Total samples: {len(df)}')
print(f'Number of features: {len(df.columns)-1}')
print(f'Target variable: {df.columns[-1]}')
print('\nFeature data types:')
print(df.dtypes)
print('\nTarget variable distribution:')
print(df[df.columns[-1]].value_counts())
print('\nTarget variable proportion:')
print(df[df.columns[-1]].value_counts(normalize=True))

# 2. Data Quality Issue Detection
print('\n' + '='*50)
print('2. Data Quality Issue Detection')
print('='*50)

# 2.1 Missing Value Analysis
print('\n2.1 Missing Value Analysis')
missing_count = df.isnull().sum()
missing_ratio = df.isnull().sum() / len(df)
missing_df = pd.DataFrame({'Missing count': missing_count, 'Missing ratio': missing_ratio})
missing_df = missing_df[missing_df['Missing count'] > 0]
print('Features with missing values:')
print(missing_df)

# 2.2 Class Imbalance Assessment
print('\n2.2 Class Imbalance Assessment')
target_counts = df[df.columns[-1]].value_counts()
print(f'Number of classes: {len(target_counts)}')
print(f'Minimum class samples: {target_counts.min()}')
print(f'Maximum class samples: {target_counts.max()}')
print(f'Class ratio difference: {target_counts.max() / target_counts.min():.2f}')

# 2.3 Outlier Detection (using Isolation Forest algorithm)
print('\n2.3 Outlier Detection')
# Select numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns
# Exclude target variable
numeric_features = numeric_features[numeric_features != df.columns[-1]]

# Train Isolation Forest model
ios = IsolationForest(contamination=0.1, random_state=42)
ios.fit(df[numeric_features])
# Predict outliers
outliers = ios.predict(df[numeric_features])
# Mark outliers
print(f'Number of outliers: {len(outliers[outliers == -1])}')
print(f'Outlier ratio: {len(outliers[outliers == -1]) / len(df):.2f}')

# 3. Data Cleaning and Processing
print('\n' + '='*50)
print('3. Data Cleaning and Processing')
print('='*50)

# 3.1 Missing Value Handling (using KNN imputation)
print('\n3.1 Missing Value Handling (KNN imputation)')
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
# Check if there are still missing values after imputation
print(f'Number of missing values after imputation: {df_imputed.isnull().sum().sum()}')

# 3.2 Outlier Handling (using IQR extension method)
print('\n3.2 Outlier Handling (IQR extension method)')
for feature in numeric_features:
    Q1 = df_imputed[feature].quantile(0.25)
    Q3 = df_imputed[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Replace outliers with boundary values
    df_imputed[feature] = np.where(df_imputed[feature] < lower_bound, lower_bound, df_imputed[feature])
    df_imputed[feature] = np.where(df_imputed[feature] > upper_bound, upper_bound, df_imputed[feature])

# 4. Result Output and Method Explanation
print('\n' + '='*50)
print('4. Result Output and Method Explanation')
print('='*50)

# 4.1 Comparison before and after data cleaning
print('\n4.1 Comparison before and after data cleaning')
print(f'Sample size before cleaning: {len(df)}')
print(f'Sample size after cleaning: {len(df_imputed)}')
print(f'Total missing values before cleaning: {df.isnull().sum().sum()}')
print(f'Total missing values after cleaning: {df_imputed.isnull().sum().sum()}')

# 4.2 Method Explanation
print('\n4.2 Method Explanation')
print('Missing value handling method: KNN imputation (K=5)')
print('Reason for selection: KNN imputation can use information from similar samples to fill missing values, which is more accurate than simple mean/median imputation')
print('Outlier detection algorithm: Isolation Forest algorithm')
print('Reason for selection: Isolation Forest algorithm is suitable for outlier detection in high-dimensional data and does not require assumption of data distribution')
print('Outlier handling method: IQR extension method (1.5 times IQR)')
print('Reason for selection: IQR extension method is a statistical distribution-based outlier handling method that is simple and effective')
print('Class imbalance evaluation metric: Class ratio difference')
print('Reason for selection: Class ratio difference can intuitively reflect the degree of class imbalance')

# Save cleaned data
df_imputed.to_csv('cleaned_german_credit.csv', index=False)
print('\nCleaned data has been saved as cleaned_german_credit.csv')
