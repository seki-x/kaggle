# -*- coding: utf-8 -*-
"""
original author: @ssdf93
Source: www.zybuluo.com/ssdf93/note/54643 ### little changes
This is a temporary script file.
"""
import pandas as pd

train_raw = pd.read_csv('data\\train1002.csv', header=0)
test_raw = pd.read_csv('data\\test502.csv', header=0)

train = train_raw.values
test = test_raw.values

'''
sklearning process
-->
    clf = SomeAlgorithmClassifier()
    clf.fit(train_X,train_y)
    out=clf.predict(test)
-->
'''

def kNN():
    '''
    kNN algo in sklearn
    '''
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    print 'Start training'
    knn.fit(train[0::,1::], train[0::, 0])
    print 'Start predicting'
    out = knn.predict(test)
    print 'Start writing'
    n, m = test.shape
    ids = range(1, n+1)
    predictions_file = open('ssdf93_out'+'.csv', 'wb')
    import csv
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerows(["ImageId","Label"])
    open_file_object.writerows(zip(ids, out))
    predictions_file.close()
    print 'kNN is done'

def PCAkNN():
    '''
    kNN plus pca
    '''
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.decomposition import RandomizedPCA
    knn = KNeighborsClassifier()
    print 'Start PCA to 100'
    train_x=train[0::,1::]
    pca = RandomizedPCA(n_components=100, whiten=True).fit(train_x)
    train_x_pca = pca.transform(train_x)
    test_x_pca = pca.transform(test)
    print 'Start training'
    knn.fit(train_x_pca, train[0::,0])
    print 'Start predicting'
    out = knn.predict(test_x_pca)
    print 'Start writing!'
    n,m=test_x_pca.shape
    ids = range(1, n+1)
    predictions_file = open("ssdf93_out_pca_100"+".csv", "wb")
    import csv
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId", "Label"])
    open_file_object.writerows(zip(ids, out))
    predictions_file.close()
    print 'kNN plus pca is done'
    
    t = pca.explained_variance_ratio_
    print sum(t[:30])
    print sum(t[:50])
    print sum(t[:70])
    print sum(t)

def PCASVM():
    '''
    svm plus pca
    '''
    from sklearn.decomposition import RandomizedPCA
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn import svm
    import csv
    print 'Start PCA to 50'
    train_x=train[0::,1::]
    train_label=train[::,0]
    pca = RandomizedPCA(n_components=50, whiten=True).fit(train_x)
    train_x_pca=pca.transform(train_x)
    test_x_pca=pca.transform(test)
    
    a_train, b_train, a_label, b_label = train_test_split(train_x_pca, train_label\
    , test_size=0.33, random_state=23323)
    
    print a_train.shape
    print a_label.shape
    print 'Start training'
    rbf_svc = svm.SVC(kernel='rbf')
    rbf_svc.fit(a_train,a_label)
    print 'Start predicting'
    b_predict=rbf_svc.predict(b_train)
    score=accuracy_score(b_label,b_predict)
    print "The accruacy socre is ", score
    print 'Start writing!'
    out=rbf_svc.predict(test_x_pca)
    n,m=test_x_pca.shape
    ids = range(1,n+1)
    predictions_file = open("ssdf93_out_pca_svm"+".csv","wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId","Label"])
    open_file_object.writerows(zip(ids,out))
    predictions_file.close()
    print 'svm plus pca is done'
