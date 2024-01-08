import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
import pickle


# load dataset
url = 'lending_club_loan_two.csv'
df_before = pd.read_csv(url)

# fix date data
# Create a new DataFrame for 'issue_d'
df_before_issue = df_before['issue_d'].str.split('-', expand=True)
df_before_issue.columns = ['issue_month', 'issue_year']

# Create a new DataFrame for 'earliest_cr_line'
df_before_cr_line = df_before['earliest_cr_line'].str.split('-', expand=True)
df_before_cr_line.columns = ['cr_line_month', 'cr_line_year']

# Concatenate the new DataFrames with the original DataFrame
df_before = pd.concat([df_before, df_before_issue, df_before_cr_line], axis=1)

# drop data
columns_to_drop = ['emp_title', 'address', 'title', 'earliest_cr_line', 'issue_d', 'grade', 'sub_grade', 'purpose','initial_list_status']
df_before = df_before.drop(columns=columns_to_drop)

# label encoder 
le = LabelEncoder() 
df_before['term'] = le.fit_transform(df_before['term'])
df_before['emp_length'] = le.fit_transform(df_before['emp_length'])
df_before['home_ownership'] = le.fit_transform(df_before['home_ownership'])
df_before['verification_status'] = le.fit_transform(df_before['verification_status'])
df_before['loan_status'] = le.fit_transform(df_before['loan_status'])
df_before['application_type'] = le.fit_transform(df_before['application_type'])
df_before['cr_line_month'] = le.fit_transform(df_before['cr_line_month'])
df_before['issue_month'] = le.fit_transform(df_before['issue_month'])

# impute data
df_before['emp_length'].fillna(df_before['emp_length'].median(), inplace=True)
df_before['pub_rec_bankruptcies'].fillna(df_before['pub_rec_bankruptcies'].median(), inplace=True)
df_before['mort_acc'].fillna(df_before['mort_acc'].median(), inplace=True)
df_before['revol_util'].fillna(df_before['revol_util'].mean(), inplace=True)

# remove outlier
z_score_vars = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc']
no_outlier = df_before.copy()
outlier_info = {}
for col in z_score_vars:
    lower_limit = df_before[col].quantile(0.01)
    upper_limit = df_before[col].quantile(0.99)
    outlier_count = ((df_before[col] < lower_limit) | (df_before[col] > upper_limit)).sum()
    total_count = len(df_before[col])
    outlier_percentage = outlier_count / total_count * 100
    outlier_info[col] = {'count': outlier_count, 'percentage': outlier_percentage}

# Drop rows with outliers
for col in z_score_vars:
    lower_limit = df_before[col].quantile(0.01)
    upper_limit = df_before[col].quantile(0.99)
    no_outlier = no_outlier[(no_outlier[col] >= lower_limit) & (no_outlier[col] <= upper_limit)]

# split data
target_column = 'loan_status'
feature_columns = no_outlier.columns[no_outlier.columns != target_column]
X = no_outlier[feature_columns]
y = no_outlier[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# oversampling & undersampling
X_train_resampled, y_train_resampled = SMOTEENN(sampling_strategy='all', enn=EditedNearestNeighbours(sampling_strategy='all')).fit_resample(X_train, y_train)

# train data
clf = RandomForestClassifier(n_estimators=83, bootstrap=True, criterion='gini', max_depth=None,  random_state=0)
clf.fit(X_train_resampled, y_train_resampled)

# rb to read, wb to write
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(clf, file)