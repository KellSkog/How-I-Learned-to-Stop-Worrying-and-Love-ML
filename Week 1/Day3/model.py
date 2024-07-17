import warnings
import pandas
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pickle # Serialize ML models
from flask import Flask, request
from flask_ngrok import run_with_ngrok
import requests
import numpy as np

def testModels(x, y):
    maxErrorScoring = 'max_error'
    negMeanAbsErrScoring = 'neg_mean_absolute_error'
    r2_scoring = 'r2'
    negMeanSqrErrScoring = 'neg_mean_squared_error'

    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('Ridge', Ridge()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    models.append(('SVR', SVR()))

    results = []
    names = []
    for name, model in models:
        # ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.
        # kfold = KFold(n_splits=10, random_state=7)
        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring=maxErrorScoring)
        cv_results2 = cross_val_score(model, x, y, cv=kfold, scoring=negMeanAbsErrScoring)
        cv_results3 = cross_val_score(model, x, y, cv=kfold, scoring=r2_scoring)
        cv_results4 = cross_val_score(model, x, y, cv=kfold, scoring=negMeanSqrErrScoring)
        msg = "%s: max error: %f , mean absolute error: %f, r2: %f, mean square error: %f" % (name, cv_results.mean(), -cv_results2.mean(), cv_results3.mean(), -cv_results4.mean())
        print(msg)

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def home():
    X = int(request.args.get('X', ''))
    Y = int(request.args.get('Y', ''))
    month = int(request.args.get('month', ''))
    day = int(request.args.get('day', ''))
    FFMC = float(request.args.get('FFMC', ''))
    DMC = float(request.args.get('DMC', ''))
    DC = float(request.args.get('DC', ''))
    ISI = float(request.args.get('ISI', ''))
    temp = float(request.args.get('temp', ''))
    RH = float(request.args.get('RH', ''))
    wind = float(request.args.get('wind', ''))
    rain = float(request.args.get('rain', ''))
    prediction = lasso_model.predict([[X, Y, month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain]])
    print('*******************************')
    print(prediction)
    return 'Prediction is ' + str(prediction[0])

def lasso(x, y):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)
    lassoModel = Lasso()
    lassoModel.fit(xTrain, yTrain)
    predictions = lassoModel.predict(xTest)
    # print(predictions)

    # Serializing the model allows sharing and reusing the trained model (algo + training = model)
    # pickle.dump(lassoModel, open('model.pkl', 'wb'))
    model = pickle.load(open('model.pkl', 'rb'))

    # run_with_ngrok(app)
    # app.run()


def main():
    pandas.set_option('future.no_silent_downcasting', True)
    names = ['X', 'Y', 'month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
    df = pandas.read_csv('Week 1/Day 1/forestfires.csv', names=names)

    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month'] = df['month'].replace(month_mapping)

    day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
    df['day'] = df['day'].replace(day_mapping)

    array = df.values
    x = array[:,0:12]
    y = array[:,12]
    # testModels(x, y)
    lasso(x, y)


if __name__ == '__main__':
    main()