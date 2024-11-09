import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('path/to/your/heart_failure_clinical_records_dataset.csv')

# Step 1: Examine the Data
print("First few rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())

# Step 2: Check for Missing Values
print("\nMissing values in each column:")
print(data.isnull().sum())  # Summarize missing values in each column

# Step 3: Remove Duplicates (if any)
data = data.drop_duplicates()

# Step 4: Handle Outliers
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Capping outliers at the 1st and 99th percentiles for numeric columns
for col in numeric_cols:
    lower_limit = data[col].quantile(0.01)
    upper_limit = data[col].quantile(0.99)
    data[col] = data[col].clip(lower=lower_limit, upper=upper_limit)

# Step 5: Encode Categorical Variables (if any)
print("\nCategorical columns:")
print(data.select_dtypes(include=['object']).columns)

# Step 6: Feature Scaling
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Step 7: Split Data into Training and Test Sets
# Assuming the target variable is 'DEATH_EVENT' in this dataset
X = data.drop('DEATH_EVENT', axis=1)  # Features
y = data['DEATH_EVENT']               # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining and testing sets created successfully.")
print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

