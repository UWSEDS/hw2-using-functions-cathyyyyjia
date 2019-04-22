import pandas as pd


# 1

# Find an online data source
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

# Read the online file and create a data frame
df = pd.read_csv(url, sep=';', header=0)

# Red wine quality data from UCI machine learning repo
# 1599 rows (observations) * 12 columns (attributes)
# Columns               Data Type
# fixed acidity         'float64'
# volatile acidity      'float64'
# citric acid           'float64'
# residual sugar        'float64'
# chlorides             'float64'
# free sulfur dioxide   'float64'
# total sulfur dioxide  'float64'
# density               'float64'
# pH                    'float64'
# sulphates             'float64'
# alcohol               'float64'
# quality               'int64'


# 2

def test_create_dataframe(df):
    """
    Input - a pandas DataFrame
    Output - return True if the following conditions hold:
             i.   The DataFrame contains only the columns that you specified in #1.
             ii.  The columns contain the correct data type
             iii. There are at least 10 rows in the DataFrame
             else return Falsse
    """
    
    # i.   The DataFrame contains only the columns that you specified in #1
    # ii.  The columns contain the correct data type
    cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    for col in df.columns:
        # Check with cols (columns specified in #1)
        if col not in cols:
            return False
        # All columns should be 'float64' except for 'quality' which is 'int64'
        if col != 'quality':
            if df[col].dtypes != 'float64':
                return False
        else:
            if df[col].dtypes != 'int64':
                return False
    
    # iii. There are at least 10 rows in the DataFrame
    if len(df) < 10:
        return False
    
    # All three conditions satisfied
    return True

# Test case 1: Original data frame
# Return True
test1 = df.copy()
print('Test case 1: Original data frame')
print(test_create_dataframe(test1))

# Test case 2: Six selected columns and first 15 rows from the original data frame
# Return True
test2 = df.copy()[['fixed acidity', 'volatile acidity', 'citric acid',
                   'residual sugar','chlorides', 'free sulfur dioxide']][0:15]
print('\nTest case 2: Extracted 15 rows * 6 columns data frame')
print(test_create_dataframe(test2))

# Test case 3: Five selected columns from the original data frame and one additional column 'dummy'
# Return False: Fail in condition 1
test3 = df.copy()
test3['dummy'] = test3['quality']
print("\nTest case 3: Extracted 5 columns with extra 'dummy' column")
print(test_create_dataframe(test3))

# Test case 4: Wrong data type for column 'pH'
# Return False: Fail in condition 2
test4 = df.copy()
test4['pH'] = test4['pH'].astype('int64')
print("\nTest case 4: Column 'pH' has data type 'int64'")
print(test_create_dataframe(test4))

# Test case 5: Six selected columns and first 9 rows from the original data frame
# Return False: Fail in condition 3
test5 = df.copy()[['fixed acidity', 'volatile acidity', 'citric acid',
                   'residual sugar','chlorides', 'free sulfur dioxide']][0:9]
print('\nTest case 5: Extracted 9 rows * 6 columns data frame')
print(test_create_dataframe(test5))
