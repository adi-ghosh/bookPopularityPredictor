import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # You need to run this only once, in order to download the stopwords list
    # nltk.download('stopwords')

    # Load the stopwords list
    stop_words_list = stopwords.words('english')

    booktextlist = []
    y = []

    with open('dataset.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            y.append(row['rating'])
            with open(row['textpath'], encoding="utf8") as f:
                contents = f.read()
                #print(contents)
                booktextlist.append(contents)

        # Instantiate CountVectorizer
        cv = CountVectorizer(stop_words=stop_words_list)

        # Fit and transform
        cv_fit = cv.fit_transform(booktextlist)
        word_list = cv.get_feature_names_out()
        count_list = cv_fit.toarray()

        # Create a dataframe with words and their respective frequency
        # Each row represents a document starting from document1
        X = pd.DataFrame(data=count_list, columns=word_list)

        # Print out the df
        # print(X)

        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X.to_numpy(), y)
        Pipeline(steps=[('standardscaler', StandardScaler()),
                        ('svc', SVC(gamma='auto'))])
        print(clf.predict([[X.to_numpy()[0]]]))



