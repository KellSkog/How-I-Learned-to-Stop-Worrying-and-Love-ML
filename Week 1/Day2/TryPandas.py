import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting._matplotlib import scatter_matrix
import seaborn as sns # Heatmap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

def histo(dataFrame):
    dataFrame.hist(sharex=False, sharey=False,xlabelsize=15, ylabelsize=15, color='orange', figsize=(15,15))
    plt.suptitle("Histograms", y=1.0, fontweight='bold', fontsize=30)
    plt.show()
def density(dataFrame):
    dataFrame.plot(kind='density', subplots=True, layout=(7,2), sharex=False, fontsize=12, figsize=(15,15))
    plt.suptitle("Probability Desity Function", y=1.0, fontweight='bold', fontsize=30)
    plt.show()
def box(dataFrame):
    dataFrame.plot(kind='box', subplots=False, layout=(4,4), sharex=False, sharey=False, fontsize=12, figsize=(8,8))
    plt.suptitle("Box and Whisker", y=1.0, fontweight='bold', fontsize=30)
    plt.show()
def scatter(dataFrame):
    axes = scatter_matrix(dataFrame, figsize=(10,10))
    plt.suptitle("Scatter Matrix", y=1.0, fontweight='bold', fontsize=20)
    plt.rcParams['axes.labelsize'] = 10
    [plt.setp(item.yaxis.get_majorticklabels(), 'size', 10) for item in axes.ravel()]
    [plt.setp(item.xaxis.get_majorticklabels(), 'size', 10) for item in axes.ravel()]
    plt.show()
def heatmap(dataFrame):
    plt.figure(figsize = (8,8))
    plt.style.use('default')
    sns.heatmap(dataFrame.corr(), annot = True)
    plt.show()

def modelSelection():
    x,y = np.arange(10).reshape((5,2)), range(5)
    # print(f'X: {x}\n Y: {list(y)}')
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=42)
    print(f'xTrain {xTrain}')
    print(f'xTest {xTest}')
    print(f'yTrain {yTrain}')
    print(f'yTest {yTest}')
def kFold():
    dataset = range(16)
    KFCrossValidator = KFold(n_splits=4, shuffle=False)
    KFdataset = KFCrossValidator.split(dataset)
    print('{} {:^61} {}'.format('Round', 'Training set', 'Test set'))
    for iteration, data in enumerate(KFdataset, start=1):
        print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))

def main():
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('future.no_silent_downcasting', True)

    names = ['X', 'Y', 'month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
    dataFrame = pd.read_csv('Week 1/Day 1/forestfires.csv', names=names)
    # print(f'Shape of forestfires.csv: {dataFrame.shape}')
    # print(f'Data types in dataFrame: \n{dataFrame.dtypes}')
    # print(f'First five rows:\n{dataFrame.head()}')
    # print(f'Describe dataset \n{dataFrame.describe()}')
    # print(f"Pearson\n{dataFrame.corr(method='pearson')}")
    '''FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0.
    This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

    For example, when doing 'df[col].method(value, inplace=True)', 
    try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, 
    to perform the operation inplace on the original object.'''
    # Fixed!
    df = dataFrame.copy()
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month'] = df['month'].replace(month_mapping)

    day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
    df['day'] = df['day'].replace(day_mapping)

    # print(f'First five rows:\n{df.head()}')
    # print(f"Pearson\n{df.corr(method='pearson')}")
    # histo(df)
    # density(df)
    # box(df)
    # scatter(df)
    # heatmap(df)
    # modelSelection()
    kFold()

if __name__ == '__main__':
    main()