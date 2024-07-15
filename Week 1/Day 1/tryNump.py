import numpy
import scipy
import matplotlib.pyplot as plt
import pandas as pd

def demoArrays():
    pythonArray = [[1, 2, 3], [10, 20, 30]] #pythonArray[0][1] == 2
    print(f'pythonArray[0][1] == 2 {pythonArray[0][1] == 2}')
    npArray = numpy.array(pythonArray)
    print(f'shape rows, cols: {npArray.shape}')
    print(f'First row of npArray[0] {npArray[0]}')
    print(f'First column of npArray {npArray[:,0]}')

def plotIt():
    plt.plot([1,2,3,4], [5, 10, 15, 20])
    plt.title("Linear relationships", fontsize = 20)
    plt.xlabel("X axis" , fontsize = 20)
    plt.ylabel("Y Axis", fontsize = 20)
    plt.show()

'''Pandas core components are Series and DataFrame'''
def demoPandas():
    dictionary = {
        'cars' : [5,4,1,7],
        'boats' : [2,6,0,2]
    }

    vehicles = pd.DataFrame(dictionary, index = ['Peter', 'Sara', 'Ali', 'John'])
    print(f'vehicles.info: {vehicles.info()}')
    print(f"Ali: {vehicles.loc['Ali']}")
    print(f'vehicles.head {vehicles.head()}')

def loadForestFires(filename, names):
    df = pd.read_csv(filename, names=names)
    # isnull finds all rows in each column and return true if null
    # sum sums all rows for each column and returns in this case the count of nullvalues
    # in each column.
    print(f'Sum:\n{pd.isnull(df).sum()}') # Prints number of null values in each 

def main():
    # demoArrays()
    # plotIt()
    # demoPandas()
    names = ['X', 'Y', 'month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
    loadForestFires('Week 1/Day 1/forestfires.csv', names)

if __name__ == '__main__':
    main()