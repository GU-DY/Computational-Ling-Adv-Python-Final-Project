# ANLY 521 Final Project 

**Role of Topic Modeling in the Product Reviews Sentiment Analysis** 

**Deyuan Wang, Jieqiao Luo, Yiming Yu**


# Project Overview

The final project, a comparative analysis, which aims at examining the role of topic modeling in the product reviews sentiment analysis. 

The dataset is 2018 Women's E-Commerce Clothing Reviews, which includes 23,000 customer reviews and ratings on various categories clothing. Based on previous research, rating has strong correlation with sentiment, so we generate the sentiment label based on ratings, such as 4 to 5 stars is positive sentiment, 3 stars is neutral sentiment, and 1-2 stars is negative sentiment. 

Then, we would use two topic modeling techniques, including: K-means Clustering and LDA with Bag of Words and apply these results on sentiment prediction analysis. 

At the end, we would use three classification models, including Logistic Regression, Naive Bayes, and Support Vector Machine (SVM) to compare the sentiment prediction matrices between the features with topic modeling and that without topic modeling to evaluate the role of topic modeling in the reviews sentiment analysis. 

The results show that although topic modeling has slight direct impact on improving the models’ performance to predict the sentiment, it can reflect some insightful patterns behind the evaluation matrices, such as users’ reviews overall sentiment, users’ preference, and popular categories. 

# Files Description

## Womens Clothing E-Commerce Reviews.csv

The dataset we used for this study. The Kaggle link is https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

## data_preparation.py

Functions of data preparation (tokenization, vectorization) and evaluation.

## Models-Without-TP.py

We apply Logistic Regression, Naïve Bayes, and Support Vector Machine (SVM) to TF-IDF and bag of words.

Usage `python Models-Without-TP.py`

## K-means-clustering.py

We apply Logistic Regression, Naïve Bayes, and Support Vector Machine (SVM) to bag of words plus K-means Clustering.

Usage `python K-means-clustering.py`

## LDA-Clustering.py

We apply Logistic Regression, Naïve Bayes, and Support Vector Machine (SVM) to bag of words plus LDA.

Usage `python LDA-Clustering.py`

## Dataset_EDA.py

Dataset exploration analysis to help better understand the dataset.

## Requirements.txt

 Library requirements.
