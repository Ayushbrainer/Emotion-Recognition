import numpy as np;
from sklearn.utils import shuffle


def cost_sigmoid(T,Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def error_rate(actual,predict):
    return np.mean(actual!=predict)


def getData(balance_ones=False,NTest=1000):
    X = []
    Y = []
    first = True

    for row in open('./my_face_recog/fer2013/fer2013/fer2013.csv'):
        if(first):
            first = False
            continue
        else:
            r = row.split(',')
            X.append([int(p) for p in r[1].split()])
            Y.append(int(r[0]))
    
    X,Y = np.array(X)/255,np.array(Y)

    X,Y = shuffle(X,Y)

    Xtrain,Ytrain = X[:-NTest],Y[:-NTest]
    XTest,YTest = X[-NTest:],Y[-NTest:]


    return Xtrain,Ytrain,XTest,YTest

def sigmoid(z):
    return 1/(1+np.exp(-z))


def getBinaryData(balance_ones=False,NTest=1000):
    X = []
    Y = []
    first = True

    for row in open('./my_face_recog/fer2013/fer2013/fer2013.csv'):
        if(first):
            first = False
            continue
        else:
            y = int(row[0])
            if(y==0 or y==1):
                r = row.split(',')
                X.append([int(p) for p in r[1].split()])
                Y.append(int(r[0]))
    
    X,Y = np.array(X)/255,np.array(Y)

    X,Y = shuffle(X,Y)

    # Xtrain,Ytrain = X[:-NTest],Y[:-NTest]
    # XTest,YTest = X[-NTest:],Y[-NTest:]


    return X,Y
