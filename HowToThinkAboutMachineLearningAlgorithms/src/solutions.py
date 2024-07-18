from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

#SVM
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

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

def main():
    supportVectorMachine()

if(__name__ == '__main__'):
    main()