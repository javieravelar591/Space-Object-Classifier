import numpy as np
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make sure processed data folder exists
os.makedirs("data/processed", exist_ok=True)

# cols that aren't important for classification
cols_to_drop = [
    'objid', 'specobjid', 'fiberid', 'field',
    'run', 'rerun', 'camcol', 'plate', 'mjd'
]

dataset = pd.read_csv('data/raw/SDsS_DR18.csv')
dataset = dataset.drop(columns=cols_to_drop)
x = dataset.drop(columns=['class'])
print(x[0:1])
y = dataset['class']

# Common way is to just ignore the missing data
# We will be replacing the missing values with the average of the columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Only getting the columns with numerical values
original_cols = x.columns
x = imputer.fit_transform(x) # the columns we want to replace missing values in, returns new matrix so replace the columns with missing vlaues
le = LabelEncoder()
y = le.fit_transform(y) # last column from above, don't need to convert to numpy array { 0: Galaxy, 1: QSO, 2: Star}

# will get 4 matrices, 1 train/1 test set for X matrix
# 1 train/1 test set for dependent variables Y
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# for some of the machine learning models, want to avoid some features being dominated by other features
sc = StandardScaler()

# only want to apply feature scaling to numerical values, not categorical like the country IE the ones we converted to numerical values
# fit computs the mean and std deviation of the featuer columns
# transform will apply the standardize formula
X_train = sc.fit_transform(X_train)
# only call transform on the training set since we want to use the same fitted scaler
X_test = sc.transform(X_test)

dataset = pd.DataFrame(dataset)
X_train_df = pd.DataFrame(X_train, columns=original_cols)
print(X_train_df[0:10])
X_test_df = pd.DataFrame(X_test, columns=original_cols)
Y_train_df = pd.DataFrame(Y_train, columns=["class"])
Y_test_df = pd.DataFrame(Y_test, columns=["class"])

X_train_df.to_pickle("data/processed/X_train.pkl")
X_test_df.to_pickle("data/processed/X_test.pkl")
Y_train_df.to_pickle("data/processed/Y_train.pkl")
Y_test_df.to_pickle("data/processed/Y_test.pkl")

# for data visualization
dataset.to_pickle("data/processed/df_train.pkl")