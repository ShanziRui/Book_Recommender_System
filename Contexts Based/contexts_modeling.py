from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def get_recommendation(top, df, scores):
    recommendation = pd.DataFrame(
        columns=['book_id', 'title', 'score'])
    count = 0
    for i in top:
        recommendation.at[count, 'book_id'] = df['book_id'][i]
        recommendation.at[count, 'title'] = df['title'][i]
        recommendation.at[count, 'score'] = scores[count]
        count += 1
    return recommendation


def context_recommender(rated_bids):

    # Use new csv for further analysis
    contexts = pd.read_csv("book_contexts.csv", index_col=0)

    # test_context -> user rated books
    bids = contexts['book_id'].unique()
    bids_user = rated_bids.isin(bids)

    if len(bids_user) == 0:
        return []
    bids_to_pred = np.setdiff1d(bids, bids_user)
    test_contexts = contexts[contexts.book_id.isin(bids_user)]
    train_contexts = contexts[contexts.book_id.isin(bids_to_pred)]

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_train = tfidf_vectorizer.fit_transform(train_contexts['corpus'])
    tfidf_test = tfidf_vectorizer.transform(test_contexts['corpus'])

    # RS with TF-IDF
    cos_similarity_tfidf = map(
        lambda x: cosine_similarity(tfidf_test, x), tfidf_train)

    # KNN
    n_neighbors = 9
    KNN = NearestNeighbors(n_neighbors, p=2)
    KNN.fit(tfidf_train)
    NNs = KNN.kneighbors(tfidf_test, return_distance=True)

    top = NNs[1][0][1:]
    index_score = NNs[0][0][1:]
    rec = get_recommendation(top, contexts, index_score)
    rec = rec[~rec.book_id.isin(bids_user)]
    return rec.book_id.unique()
