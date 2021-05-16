import warnings
warnings.filterwarnings("ignore")
import argparse
from data_preparation import data_pre, tokenize, tfidfvect, countvect, evaluation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
def main(datafile):
    # import the dataframe using data_pre function
    df = data_pre(datafile)

    # import the cleaned text data (review_content in the dataset)
    totalvocab = []
    corpus = list(df['review_content'])
    for word in corpus:
        allwords_stemmed = tokenize(word)
        totalvocab.extend(allwords_stemmed)
    vocab_frame = pd.DataFrame({'words': totalvocab})

    # TFIDF Vectorizer
    tf_df_df = tfidfvect(totalvocab, corpus)

    # Count Vectorizer
    tf_df_cv = countvect(totalvocab, corpus)

    # Create new dataframe
    newdf = df[['Age', 'Department Name', 'Division Name', 'Sentiment']].reset_index(drop=True)
    data_tf = pd.concat([tf_df_df, newdf], axis=1).reindex(newdf.index)
    data_cv = pd.concat([tf_df_cv, newdf], axis=1).reindex(newdf.index)

    # split data
    train_data_tf, test_data_tf = train_test_split(data_tf, train_size=0.75, random_state=0)
    train_data_cv, test_data_cv = train_test_split(data_cv, train_size=0.75, random_state=0)
    # select the columns and prepare data for the models
    x_train_tf = train_data_tf.iloc[:, 0:53]
    y_train_tf = train_data_tf['Sentiment']
    x_train_cv = train_data_cv.iloc[:, 0:53]
    y_train_cv = train_data_cv['Sentiment']
    x_test_tf = test_data_tf.iloc[:, 0:53]
    y_test_tf = test_data_tf['Sentiment']
    x_test_cv = test_data_cv.iloc[:, 0:53]
    y_test_cv = test_data_cv['Sentiment']
    # print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

    # Building the models
    lr = LogisticRegression(max_iter=10000)
    lr_tf = lr.fit(x_train_tf, y_train_tf)
    lr_cv = lr.fit(x_train_cv, y_train_cv)
    nb = MultinomialNB()
    nb_tf = nb.fit(x_train_tf, y_train_tf)
    nb_cv = nb.fit(x_train_cv, y_train_cv)
    svm = SVC()
    svm_tf = svm.fit(x_train_tf, y_train_tf)
    svm_cv = svm.fit(x_train_cv, y_train_cv)

    # Prediction
    df2 = test_data_tf.copy()
    df2['Logistic Regression TF'] = lr_tf.predict(x_test_tf)
    df2['Naive Bayes TF'] = nb_tf.predict(x_test_tf)
    df2['SVM TF'] = svm_tf.predict(x_test_tf)
    df2['Logistic Regression CV'] = lr_cv.predict(x_test_cv)
    df2['Naive Bayes CV'] = nb_cv.predict(x_test_cv)
    df2['SVM CV'] = svm_cv.predict(x_test_cv)

    # Evaluation
    prediction = {"TfidfVectorizer Logistic Regression Model": df2['Logistic Regression TF'],
                  "TfidfVectorizer Naive Bayes Model": df2['Naive Bayes TF'],
                  "TfidfVectorizer SVM Model": df2['SVM TF'],
                  "CountVectorizer Logistic Regression Model": df2['Logistic Regression CV'],
                  "CountVectorizer Naive Bayes Model": df2['Naive Bayes CV'],
                  "CountVectorizer SVM Model": df2['SVM CV']}

    for cond, pred in prediction.items():
        print(f"{cond}")
        evl = evaluation(prediction=pred,label=df2['Sentiment'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final Project Models Without Topic Modeling')
    parser.add_argument('--path', type=str, default="Womens Clothing E-Commerce Reviews.csv",
                        help='path to womens clothing dataset')
    args = parser.parse_args()

    main(args.path)

'''Expected Result:
TfidfVectorizer Logistic Regression Model
Accuracy Score: 0.77
Precision Score: 0.994
Recall Score: 0.77
F1 Score: 0.867

TfidfVectorizer Naive Bayes Model
Accuracy Score: 0.771
Precision Score: 0.977
Recall Score: 0.771
F1 Score: 0.859

TfidfVectorizer SVM Model
Accuracy Score: 0.77
Precision Score: 1.0
Recall Score: 0.77
F1 Score: 0.87

CountVectorizer Logistic Regression Model
Accuracy Score: 0.776
Precision Score: 0.914
Recall Score: 0.776
F1 Score: 0.833

CountVectorizer Naive Bayes Model
Accuracy Score: 0.763
Precision Score: 0.846
Recall Score: 0.763
F1 Score: 0.798

CountVectorizer SVM Model
Accuracy Score: 0.77
Precision Score: 1.0
Recall Score: 0.77
F1 Score: 0.87
'''