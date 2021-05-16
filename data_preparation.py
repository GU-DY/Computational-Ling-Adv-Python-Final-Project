import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def data_pre(datafile):
    df = pd.read_csv(datafile)
    # Drop the unnamed column
    df.drop(df.columns[0], inplace=True, axis=1)
    # Drop the null value for title, review text, and division name
    df = df.dropna(subset=['Title', 'Review Text', 'Division Name'], axis=0)
    # Combine the title and review text to a new variable callled review_content
    df['review_content'] = df['Title'] + ' ' + df['Review Text']
    # Transform the  categorical variable
    lb_make = LabelEncoder()
    df['Department Name'] = lb_make.fit_transform(df['Department Name'])
    df['Division Name'] = lb_make.fit_transform(df['Division Name'])
    # Create the sentiment label based on the rating variable
    bin_labels_rating = ['Negative', 'Medium', 'Positive']
    # remove recommendation num
    df['Sentiment'] = pd.cut(df['Rating'],
                             [0, 2, 3, 5],
                             labels=bin_labels_rating)
    return df

def tokenize(text):
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")
    # Tokenize by sentence and then by word
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any stopwords tokens and not containing letters tokens
    for token in tokens:
        if token not in stopwords:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

#TFIDF Vectorizer
def tfidfvect(vectinput,corpus):
    vect_tf = TfidfVectorizer(input=vectinput,
                              stop_words='english',
                              max_features=50)
    df_tf = vect_tf.fit_transform(corpus)
    columnnames_df_tf = vect_tf.get_feature_names()
    tf_df_df = pd.DataFrame(df_tf.toarray(), columns=columnnames_df_tf)
    return tf_df_df


#Count Vectorizer
def countvect(vectinput,corpus):
    vect_cv = CountVectorizer(input=vectinput,
                              stop_words='english',
                              max_features=50)
    df_cv = vect_cv.fit_transform(corpus)
    columnnames_df_cv = vect_cv.get_feature_names()
    tf_df_cv = pd.DataFrame(df_cv.toarray(), columns=columnnames_df_cv)
    return tf_df_cv

#Evaluation
def evaluation(prediction,label):
    accuracy = accuracy_score(prediction, label)
    precision = precision_score(prediction, label, average='weighted')
    recall = recall_score(prediction, label, average='weighted')
    f1 = f1_score(prediction, label, average='weighted')
    print(f"Accuracy Score: {accuracy:.03}\n"
          f"Precision Score: {precision:0.03}\n"
          f"Recall Score: {recall:0.03}\nF1 Score: {f1:0.03}\n")


