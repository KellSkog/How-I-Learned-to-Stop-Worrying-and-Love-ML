import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting._matplotlib import scatter_matrix
import seaborn as sns # Heatmap

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

    df = dataFrame
    df.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'), (1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
    df.day.replace(('mon','tue','wed','thu','fri','sat','sun'), (1,2,3,4,5,6,7), inplace=True)
    # print(f'First five rows:\n{df.head()}')
    # print(f"Pearson\n{df.corr(method='pearson')}")
    # histo(df)
    # density(df)
    # box(df)
    # scatter(df)
    heatmap(df)

if __name__ == '__main__':
    main()