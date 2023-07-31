# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 23:00:26 2023

@author: Sarah
"""

import csv, random, re, nltk
import pandas as pd
import time
# import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import download
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import multiprocessing as mp

# from tensorflow import keras
# from tensorflow.keras import layers, models
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# Press the green button in the gutter to run the script.


# def round_of_rating(numbers):
#     return list(map(lambda v: (v * 2) / 2, numbers))
start_time = time.time()

if __name__ == '__main__':

    # You need to run this only once, in order to download the stopwords list
    try:
        stop_words_list = stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    # Load the stopwords list
    stop_words_list = stopwords.words('english')

    booktextlist = []
    y = []

    percentage = 100
    upperlimit = 0.5+(0.5*(percentage/100))
    lowerlimit = 0.5-(0.5*(percentage/100))
    noofwords = 50000000


    def read_data(row):
        y.append(float(row['rating']))
        with open(row['textpath'], encoding="utf8") as f:
            contents = f.read()
            contents = re.sub('[^A-Za-z ]+', '', contents)
            contents = contents.lower()
            #contents = contents[int(len(contents) * lowerlimit):int(len(contents) * upperlimit)]
            words = list(map(str, contents.split()))
            #contents = " ".join(random.sample(words, noofwords))
            " ".join(contents.split())
            booktextlist.append(contents)
        return [y, booktextlist]
    
    def get_x_y(y, booktextlist):
        lab = preprocessing.LabelEncoder()
        y = lab.fit_transform(y)

        # Instantiate CountVectorizer
        cv = CountVectorizer(stop_words=stop_words_list)

        # Fit and transform
        cv_fit = cv.fit_transform(booktextlist)
        word_list = cv.get_feature_names_out()
        count_list = cv_fit.toarray()

        # Create a dataframe with words and their respective frequency
        # Each row represents a document starting from document1
        X = pd.DataFrame(data=count_list, columns=word_list)
        # y = round_of_rating(y)

        # Print out the df
        # print(X)

        # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        # clf.fit(X.to_numpy(), y)
        # Pipeline(steps=[('standardscaler', StandardScaler()),
        #                 ('svc', SVC(gamma='auto'))])
        # print(clf.predict([[X.to_numpy()[0]]]))

        # model = Sequential()
        # model.add(Dense(50, input_shape=(noofwords,), activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # # compile network
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # scores = cross_val_score(model, X, y, cv=5)
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        # text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha = 1e-3, random_state = 42,max_iter = 5, tol = None))])
        # text_clf.fit(X, y)
        # predicted = text_clf.predict(docs_test)
        # np.mean(predicted == twenty_test.target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
        #y_pred = clf.predict(X_test)
        #print(accuracy_score(y_test,y_pred))
        #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        return X_train, X_test, y_train, y_test
    
    def train_model(data):
        
        if data == 'svm':
            svm_clf = svm.SVC(kernel='linear', C=1, random_state=42).fit(X_train, y_train)
            svm_pred = svm_clf.predict(X_test, y_test)
            print("Using Support Vector Machine, accuracy is", svm_clf.score(svm_pred, y_test))
            score = svm_clf.score(X_test, y_test)
        if data == 'lin_reg':
            lin_clf = linear_model.LogisticRegression().fit(X_train, y_train)
            print("Using Logistic Regression Model, accuracy is", lin_clf.score(X_test, y_test))
            score = lin_clf.score(X_test, y_test)
        if data == 'gauss_nb':
            gauss_clf = GaussianNB().fit(X_train, y_train)
            print("Using Gaussian Naive Bayes Model, accuracy is", gauss_clf.score(X_test, y_test))
            score = gauss_clf.score(X_test, y_test)
        if data == 'dnn':
            dnn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (15,), random_state = 1).fit(X_train, y_train)
            print("Using Deep Neural Network, accuracy is", dnn_clf.score(X_test, y_test))
            score = dnn_clf.score(X_test, y_test)
        print(str(data) + " model accuracy: " + str(score))
        return ([data, score])
        
    rows = []
    with open('dataset.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
            
    pool = mp.Pool()
    y = []
    bookcontents = []
    for ys, bookcontent in pool.map(read_data, rows):
        y.append(ys)
        bookcontents.append(bookcontent)
    y = [item for sublist in y for item in sublist]
    bookcontents = [item for sublist in bookcontents for item in sublist]
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = get_x_y(y, bookcontents)
    model_types = ["svm", "lin_reg", "gauss_nb", "dnn"]
    model_info = []
    for model in model_types:
        model_info.append(model)

    pool = mp.Pool()
    for acc in pool.map(train_model, model_info):
        print(str(acc[0]) + " model accuracy: " + str(acc[1]))
    
    end_time = time.time()
    print("Total Run Time: " + str(end_time - start_time) + " seconds")