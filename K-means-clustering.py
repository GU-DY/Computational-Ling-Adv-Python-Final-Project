import warnings
warnings.filterwarnings("ignore")
import argparse
from data_preparation import data_pre, tokenize, countvect, evaluation
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def main(datafile):
    # import the dataframe
    df = data_pre(datafile)
    # import the cleaned text data (review_content in the dataset)
    totalvocab = []
    corpus = list(df['review_content'])
    for word in corpus:
        allwords_stemmed = tokenize(word)
        totalvocab.extend(allwords_stemmed)
    vocab_frame = pd.DataFrame({'words': totalvocab})
    tf_df_cv = countvect(totalvocab, corpus)
    '''
    #Choose the optimal number for K
    range_n_clusters = list(range(4, 25))
    silhouettescore=[]
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        preds = clusterer.fit_predict(tf_df_cv)
        centers = clusterer.cluster_centers_
        score = silhouette_score(tf_df_cv, preds)
        silhouettescore.append(score)
        #print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

    plt.plot(range(4, 25), silhouettescore, marker='o')
    plt.xlabel('Number of clusters')
    plt.title("WSS:Silhouette")
    plt.show()
    #The graph shows that K=9 is the optimal K value
    '''
    kmeans_object_Count = sklearn.cluster.KMeans(n_clusters=9)
    kmeans_object_Count.fit(tf_df_cv)

    # Get cluster assignment labels
    labels = kmeans_object_Count.labels_
    prediction_kmeans = kmeans_object_Count.predict(tf_df_cv)
    # print(labels)
    # print(prediction_kmeans)

    # Format results as a DataFrame
    kmeans_result = pd.DataFrame([labels]).T
    print("Top terms per cluster:")
    order_centroids = kmeans_object_Count.cluster_centers_.argsort()[:, ::-1]
    terms = tf_df_cv.columns
    for i in range(9):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print

    # generate the new dataframe
    newdf = df[['Age', 'Department Name', 'Division Name', 'Sentiment']].reset_index(drop=True)
    data_kmeans = pd.concat([tf_df_cv, kmeans_result, newdf], axis=1).reindex(newdf.index)

    # split data
    train_data_kmeans, test_data_kmeans = train_test_split(data_kmeans, train_size=0.75, random_state=0)
    # select the columns and prepare data for the models
    x_train_kmeans = train_data_kmeans.iloc[:, 0:54]
    y_train_kmeans = train_data_kmeans['Sentiment']
    x_test_kmeans = test_data_kmeans.iloc[:, 0:54]
    y_test_kmeans = test_data_kmeans['Sentiment']

    # Model
    lr = LogisticRegression(max_iter=10000)
    lr_kmeans = lr.fit(x_train_kmeans, y_train_kmeans)
    nb = MultinomialNB()
    nb_kmeans = nb.fit(x_train_kmeans, y_train_kmeans)
    svm = SVC()
    svm_kmeans = svm.fit(x_train_kmeans, y_train_kmeans)

    # Prediction
    df2 = test_data_kmeans.copy()
    df2['Logistic Regression Kmeans'] = lr_kmeans.predict(x_test_kmeans)
    df2['Naive Bayes Kmeans'] = nb_kmeans.predict(x_test_kmeans)
    df2['SVM Kmeans'] = svm_kmeans.predict(x_test_kmeans)

    # Evaluation
    prediction = {"After K-means CountVectorizer Logistic Regression Model": df2['Logistic Regression Kmeans'],
                  "After K-means CountVectorizer Naive Bayes Model": df2['Naive Bayes Kmeans'],
                  "After K-means CountVectorizer SVM Model": df2['SVM Kmeans']}

    for cond, pred in prediction.items():
        print(f"{cond}")
        evl = evaluation(prediction=pred, label=df2['Sentiment'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final Project Models with K-means')
    parser.add_argument('--path', type=str, default="Womens Clothing E-Commerce Reviews.csv",
                        help='path to womens clothing dataset')
    args = parser.parse_args()
    main(args.path)