import warnings
warnings.filterwarnings("ignore")
import argparse
from data_preparation import data_pre, tokenize, countvect, evaluation
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def main(datafile):
    df = data_pre(datafile)
    # import the cleaned text data (review_content in the dataset)
    totalvocab = []
    corpus = list(df['review_content'])
    for word in corpus:
        allwords_stemmed = tokenize(word)
        totalvocab.extend(allwords_stemmed)
    vocab_frame = pd.DataFrame({'words': totalvocab})

    # Count Vectorizer
    tf_df_cv = countvect(totalvocab, corpus)
    LDA = LatentDirichletAllocation(n_components=7, random_state=40)
    LDA.fit(tf_df_cv)
    for i, topic in enumerate(LDA.components_):
        print(f'Top 10 words for topic {i}:')
        print([tf_df_cv.columns[i] for i in topic.argsort()[-10:]])
        print('\n')

    topic_num = LDA.transform(tf_df_cv)
    df['Topic'] = topic_num.argmax(axis=1)

    # Create the new dataframe
    newdf = df[['Age', 'Department Name', 'Division Name', 'Topic', 'Sentiment']].reset_index(drop=True)
    data_lda = pd.concat([tf_df_cv, newdf], axis=1).reindex(newdf.index)

    # split data
    train_data_lda, test_data_lda = train_test_split(data_lda, train_size=0.75, random_state=0)
    # select the columns and prepare data for the models
    x_train_lda = train_data_lda.iloc[:, 0:54]
    y_train_lda = train_data_lda['Sentiment']
    x_test_lda = test_data_lda.iloc[:, 0:54]
    y_test_lda = test_data_lda['Sentiment']

    # Model
    lr = LogisticRegression(max_iter=10000)
    lr_lda = lr.fit(x_train_lda, y_train_lda)
    nb = MultinomialNB()
    nb_lda = nb.fit(x_train_lda, y_train_lda)
    svm = SVC()
    svm_lda = svm.fit(x_train_lda, y_train_lda)

    # Prediction
    df2 = test_data_lda.copy()
    df2['Logistic Regression LDA'] = lr_lda.predict(x_test_lda)
    df2['Naive Bayes LDA'] = nb_lda.predict(x_test_lda)
    df2['SVM LDA'] = svm_lda.predict(x_test_lda)

    # Evaluation
    prediction = {"After LDA CountVectorizer Logistic Regression Model": df2['Logistic Regression LDA'],
                  "After LDA CountVectorizer Naive Bayes Model": df2['Naive Bayes LDA'],
                  "After LDA CountVectorizer SVM Model": df2['SVM LDA']}

    for cond, pred in prediction.items():
        print(f"{cond}")
        evl = evaluation(prediction=pred, label=df2['Sentiment'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final Project Models with LDA')
    parser.add_argument('--path', type=str, default="Womens Clothing E-Commerce Reviews.csv",
                        help='path to womens clothing dataset')
    args = parser.parse_args()
    main(args.path)

'''
Expected Results
After LDA CountVectorizer Logistic Regression Model
Accuracy Score: 0.775
Precision Score: 0.912
Recall Score: 0.775
F1 Score: 0.832

After LDA CountVectorizer Naive Bayes Model
Accuracy Score: 0.764
Precision Score: 0.849
Recall Score: 0.764
F1 Score: 0.8

After LDA CountVectorizer SVM Model
Accuracy Score: 0.77
Precision Score: 1.0
Recall Score: 0.77
F1 Score: 0.87
'''