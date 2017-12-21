import numpy as np
import pylab as pl
import pickle
import os

from PIL import Image  
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

import feature as NPD
import ensemble as ADB
import ensemble as CLA


def exfuture(m,n,file):
    X = np.zeros((m+n,165600))
    i = 0
    output = open(file,"wb")
    for file in os.listdir("./datasets/original/face"):
        
        im = np.array(Image.open("./datasets/original/face/"+file).convert('L').resize((24,24),Image.BILINEAR),'i')
        x = NPD.NPDFeature(im)
        X[i] = x.extract()
        i += 1
    for file in os.listdir("./datasets/original/nonface"):
        
        im = np.array(Image.open("./datasets/original/nonface/"+file).convert('L').resize((24,24),Image.BILINEAR),'i')
        x = NPD.NPDFeature(im)
        X[i] = x.extract()
        i += 1
    pickle.dump(X,output)
    y = np.hstack((np.ones((m,)).T,np.zeros((n,)).T))
    pickle.dump(y,output)
    output.close()

if __name__ == "__main__":
    # write your code here
    
    #exfuture(500,500,"traindata.mat")
    
    M = 5
    data = open("traindata.mat","rb")
    X = pickle.load(data)
    y = pickle.load(data)
    W = np.ones((800,)) * 1/800
    data.close()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    '''
    G = ADB.AdaBoostClassifier(tree.DecisionTreeClassifier,5)
    G.fit(X_train,y_train)
    output = open('model.m','wb')
    pickle.dump(G,output)
    output.close()
    '''
    with open('model.m', "rb") as f:
        G = pickle.load(f)
    predict = G.predict(X_test,0.05)
    with open('report.txt', "w") as f:
        f.write(classification_report(predict, y_test, target_names = ['nonface', 'face']))

        