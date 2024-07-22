import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

#SVM
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

# Solving Regression problems
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.impute import KNNImputer # Replace NaN with K-Nearest Neighbor

# Implementing ALS for movie recommendation
import implicit 
from scipy.sparse import coo_matrix, csr_matrix
from threadpoolctl import threadpool_limits
import heapq

# Implementing K-Means Clustering
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')
parser_bayes = subparsers.add_parser('bayes', help='Run the naive Bayes example')
parser_svm = subparsers.add_parser('svm', help='Run the Support Vector Machine example')
parser_plot = subparsers.add_parser('plot', help='Plot the graph of Ri-Rf, Rm-Rf')
parser_google = subparsers.add_parser('google', help='Linear regress Google stock')
parser_movie = subparsers.add_parser('movie', help='ALS for movie recommendation')
parser_kmeans = subparsers.add_parser('kmeans', help='K-means sentiment analysis')

def naiveBayes():
    with open("HowToThinkAboutMachineLearningAlgorithms/src/imdb_labelled.txt", 'r') as text_file:
        lines = text_file.read().split('\n')
    with open("HowToThinkAboutMachineLearningAlgorithms/src/yelp_labelled.txt", 'r') as text_file:
        lines += text_file.read().split('\n')
    with open("HowToThinkAboutMachineLearningAlgorithms/src/amazon_cells_labelled.txt", 'r') as text_file:
        lines += text_file.read().split('\n')
    lines = [line.split('\t') for line in lines if len(line.split('\t')) == 2 and line.split('\t')[1] != '']

    trainComments = [line[0] for line in lines]
    trainLabels = [int(line[1]) for line in lines]

    countVec = CountVectorizer(binary=True)
    trainComments = countVec.fit_transform(trainComments)
    
    classifier = BernoulliNB().fit(trainComments, trainLabels)

    print(classifier.predict(countVec.transform(['this is the best movie']))[0] == 1)
    print(classifier.predict(countVec.transform(["this is not the worst movie"]))[0] == 0)
    
def toNum(text):
    try:
        return float(text)
    except:
        return np.nan
def seriesToNum(series):
    return series.apply(toNum)
def toLabel(str):
        if(str == 'ad.'):
            return 1
        return 0

def supportVectorMachine():
    ''' A Support Vector Machine is used to find the bundary plane between sets of points.
        The points represent objects in an N-dimensiona hypercube.
        '''
    dataFile = 'HowToThinkAboutMachineLearningAlgorithms/src/ad.data'
    # Read the csv as comma separated without header
    data = pd.read_csv(dataFile, sep=',', header=None, low_memory=False)
    # Last column contain label, exclude it
    trainData = data.iloc[0:,0:-1].apply(seriesToNum)
    trainData=trainData.dropna()
    # Add label as int
    trainLabels = data.iloc[trainData.index, -1].apply(toLabel)

    # Train the model
    clf = LinearSVC()
    clf.fit(trainData[100:2300], trainLabels[100:2300])

    # Test
    print(clf.predict(trainData.iloc[12].values.reshape(1, -1))[0])
    print(clf.predict(trainData.iloc[-1].values.reshape(1, -1))[0])

def plot():
    plt.plot([1,2,3,4,5,6,7,8], [2,3,4,3,5,8,6,8], 'bo')
    plt.plot([1,8], [2,8], 'g')
    plt.title("Linear relationships", fontsize = 20)
    plt.xlabel("Rm - Rf" , fontsize = 20)
    plt.ylabel("Ri - Rf", fontsize = 20)
    plt.show()

def readFile(filename, tbond=False):
    data = pd.read_csv(filename, sep=',', usecols=[0,6], names=['Date', 'Price'], header=0)
    if not tbond: # Regular stock
        returns = np.array(data['Price'][:-1], float) / np.array(data['Price'][1:], float) - 1
        data['Returns'] = np.append(returns, np.nan)
    else: # Treasury yields
        data['Returns'] = data['Price'] / 100.0
    data.index = data['Date']
    data = data['Returns'][0:-1]
    return data

def google():
    googleData = readFile('HowToThinkAboutMachineLearningAlgorithms/src/GOOG.csv')
    nasdaqData = readFile('HowToThinkAboutMachineLearningAlgorithms/src/nasdaq.csv')
    tbondData = readFile('HowToThinkAboutMachineLearningAlgorithms/src/tbond.csv', tbond=True)

    # Handle NaN values
    nasdaq_minus_tbond = nasdaqData - tbondData
    google_minus_tbond = googleData - tbondData
    
    # Fill NaN values with the mean of the column
    nasdaq_minus_tbond = nasdaq_minus_tbond.fillna(nasdaq_minus_tbond.mean())
    google_minus_tbond = google_minus_tbond.fillna(google_minus_tbond.mean())
    
    reshaped = nasdaq_minus_tbond.values.reshape(-1, 1)
    regressor = SGDRegressor(eta0=0.1, max_iter=100000, fit_intercept=False)
    regressor.fit(reshaped, google_minus_tbond.values)
    print(f'beta {regressor.coef_}')

def movie():
    threadpool_limits(1, "blas")
    # check_blas_config()
    filename = 'HowToThinkAboutMachineLearningAlgorithms/src/u.data'
    data = pd.read_csv(filename, sep='\t', header=None, usecols=[0,1,2], names=['userId', 'itemId', 'rating'])
    data['userId'] = data['userId'].astype('category')
    data['itemId'] = data['itemId'].astype('category')
    
    ratingMatrix = coo_matrix((data['rating'].astype(float),
        (data['itemId'].cat.codes.copy(),
        data['userId'].cat.codes.copy())))
    csrMatrix = ratingMatrix.tocsr()
    '''RuntimeWarning: OpenBLAS is configured to use 16 threads. 
    It is highly recommended to disable its internal threadpool by setting the environment variable 'OPENBLAS_NUM_THREADS=1' 
    or by calling 'threadpoolctl.threadpool_limits(1, "blas")'. 
    Having OpenBLAS use a threadpool can lead to severe performance issues here.'''
    model = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.01)
    model.fit(csrMatrix)
    # userFactors, itemFactors = implicit.als.AlternatingLeastSquares(ratingMatrix, factors=10, regularization=0.01)
    userFactors = model.user_factors
    itemFactors = model.item_factors

    user196 = itemFactors.dot(userFactors[196])
    sugestions = heapq.nlargest(3, range(len(user196)), user196.take)
    print(f'Hot movies {sugestions}')

def kmeans():
    with open('HowToThinkAboutMachineLearningAlgorithms/src/imdb_labelled.txt', 'r') as textFile:
              lines = textFile.read().split('\n')
    lines = [line.split('\t') for line in lines if len(line.split('\t')) == 2 and line.split('\t')[1] != '']

    trainDocuments = [line[0] for line in lines]
    tfidfVectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english') # stop_words='english' removes words like a, an, the, is
    trainDocuments = tfidfVectorizer.fit_transform(trainDocuments)
    km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=True)
    km.fit(trainDocuments)
    # for count in range(3):
    #     for i in range(len(lines)):
    #         if km.labels_[i] == 0:
    #             print(lines[i])
    cluster = 2
    count = 0
    for i in range(len(lines)):
        if count > 3:
            break
        if km.labels_[i] == cluster:
            print(lines[i])
            count += 1

def main():
    # naiveBayes()
    # supportVectorMachine()
    args = parser.parse_args()
    if args.command == 'bayes':
        naiveBayes()
    elif args.command == 'svm':
        supportVectorMachine()
    elif args.command == 'plot':
        plot()
    elif args.command == 'google':
        google()
    elif args.command == 'movie':
        movie()
    elif args.command == 'kmeans':
        kmeans()
    else:
        parser.print_help()

if(__name__ == '__main__'):
    main()