import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData,error_rate,sigmoid,cost_sigmoid

class LogisticModel(object):
    def __init__(self):
        pass

    def forward(self,X):
        return sigmoid(X.dot(self.w)+self.b)
    
    def predict(self,X):
        pY = self.forward(X)
        return np.round(pY)
    
    def score(self,X,Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y,prediction)

    def fit(self,X,Y,epochs=120000,show_fig=True,learning_rate = 0.01*10e-6,reg=0.):
        X,Y = shuffle(X,Y)
        Xvalid = X[-1000:]
        Yvalid = Y[-1000:]

        X,Y = X[:-1000],Y[:-1000]

        N,D = X.shape
        self.w = np.random.rand(D)/np.sqrt(D)
        self.b=0


        costs = []
        best_learning_rate = 0.01

        for i in range(epochs):
            pY = self.forward(X)
            self.w  -= learning_rate*(X.T.dot(pY-Y) + reg*self.w)
            self.b  -= learning_rate*((pY-Y).sum() + reg*self.b)
            
    
            if i % 100 == 0:
                pYvalid = self.forward(Xvalid)
                c = cost_sigmoid(Yvalid,pYvalid)
                costs.append(c)
                error = error_rate(Yvalid,np.round(pYvalid))

                if i%10000 == 0:
                    print("i:" ,i," error: ",error," cost: ",c)

                if(error < best_learning_rate):
                    best_learning_rate = error

        print("best_learning_rate: " , (best_learning_rate))

        if(show_fig):
            plt.plot(costs)
            plt.show()




def main():
    X,Y = getBinaryData()

    X0 = X[Y==0]
    X1 = X[Y==1]
    X0Test = X0[-10:]
    X1Test = X1[-10:]
    X1 = X1[:-10]
    X1 = np.repeat(X1,9,axis=0)
    X0 = X0[:-10]
    


    model = LogisticModel()
    
    XTest = np.vstack((X0Test,X1Test))
    YTest = np.array([0]*len(X0Test) + [1]*len(X1Test))

    X = np.vstack((X0,X1))
    Y = np.array([0]*len(X0) + [1]*len(X1))

    model.fit(X,Y,show_fig=True)

    print("FITTING SCORE: ",model.score(X,Y))

    print("TESTING SCORE: ", model.score(XTest,YTest))
    


if __name__=='__main__':
    main()