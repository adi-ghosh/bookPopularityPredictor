import csv, random, re, nltk
import pandas as pd
# import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import download
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# from tensorflow import keras
# from tensorflow.keras import layers, models
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# Press the green button in the gutter to run the script.


# def round_of_rating(numbers):
#     return list(map(lambda v: (v * 2) / 2, numbers))


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

    percentage = 10
    upperlimit = 0.5+(0.5*(percentage/100))
    lowerlimit = 0.5-(0.5*(percentage/100))
    noofwords = 50000

    with open('dataset.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
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

        # y_pred = clf.predict(X_test)
        # print(accuracy_score(y_test,y_pred))
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


        clf = svm.SVC(kernel='linear', C=1, random_state=42).fit(X_train, y_train)
        print("Using Support Vector Machine, accuracy is", clf.score(X_test, y_test))

        clf = linear_model.LogisticRegression().fit(X_train, y_train)
        print("Using Logistic Regression Model, accuracy is", clf.score(X_test, y_test))

        clf = GaussianNB().fit(X_train, y_train)
        print("Using Gaussian Naive Bayes Model, accuracy is", clf.score(X_test, y_test))

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (15,), random_state = 1).fit(X_train, y_train)
        print("Using Deep Neural Network, accuracy is", clf.score(X_test, y_test))
