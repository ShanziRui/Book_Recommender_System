from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise import Dataset
import surprise
from surprise import Reader
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from random import sample
import pandas as pd
import contexts_modeling


def SVDpp_recommender(df, test):
    # read data
    reader = Reader(rating_scale=(0, 9))
    data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    # train model
    algo = SVDpp()
    test_pre = algo.fit(trainset).test(test)
    rec = pd.DataFrame(test_pre, columns=[
                       'uid', 'bid', 'rui', 'est', 'details'])
    rec['err'] = abs(rec.est - rec.rui)
    best_rec = rec.sort_values(by='err')[:10]
    return best_rec


def add_user():
    max_user = ratings.user_id.max()
    user = max_user + 1
    print('-----------------------------------------------------------------------')
    print('Congratulations! You are successfully registered. Your user_id is', user)
    print('-----------------------------------------------------------------------')
    print()

    print('-----------------------------------------------------------------------')
    print('                 List of Books Avaiable in our Database                ')
    print('-----------------------------------------------------------------------')
    print()

    new_ratings = '0'
    while new_ratings == '0':
        print(books.sample(n=15))
        print()
        print(
            'Please rate for at least three books above in format of (book_id, rating), split each by empty space. 1.0 indcates least liked and 5.0 indicates most liked, press 0 to fresh the book list.')
        new_ratings = input()
        print()
    new_ratings = new_ratings.split(' ')
    rating_list = []
    for pair in new_ratings:
        book_id = int(pair.split(',')[0][1:])
        rating = float(pair.split(',')[1][:-1])
        new_record = (book_id, user, rating)
        rating_list.append(new_record)
    user_ratings = pd.DataFrame(
        rating_list, columns=['book_id', 'user_id', 'rating'])
    return user_ratings


# Use new csv for further analysis
ratings = pd.read_csv("ratings_clean.csv", index_col=0)
ratings = ratings.drop(columns=['counts'])

# print(ratings.head())

books = pd.read_csv("books.csv", index_col=0)
books = books[['book_id', 'title']]
# print('Number of books (before): ', books.shape)

#bids = ratings['book_id'].unique()
bids = books['book_id'].unique()
#books = books[books.book_id.isin(bids)]
ratings = ratings[ratings.book_id.isin(bids)]
books.to_csv('books_train.csv')
# print('Number of books (after): ', books.shape)


# Obtain user information
print('-----------------------------------------------------------------------')
print(
    'Welcome to BookRecommender, enter your user_id or type 0 to register!')
print('-----------------------------------------------------------------------')
user = int(input())
print()

if user == 0:
    new_records = add_user()
    ratings = ratings.append(new_records)

# get a list of all book_ids
bids = ratings['book_id'].unique()
# get a list of book_ids that user has rated
bids_user = ratings.loc[ratings['user_id'] == user, 'book_id']
# remove the book_ids that user has rated
bids_to_pred = np.setdiff1d(bids, bids_user)


# recommender result from context-based system
context_rec = contexts_modeling.context_recommender(bids_user)


# recommender result from ratings-based system
testset = [[user, bid, 5.0] for bid in bids_to_pred]
predictions = SVDpp_recommender(ratings, testset)
book_list = predictions.bid.unique()

# Intersection of two recommendation systems
final_list = []
pending_list = []
for id in book_list:
    if id in context_rec:
        final_list.append(id)
    else:
        pending_list.append(id)
if len(final_list) < 5:
    to_add = 5 - len(final_list)
    final_list.extend(pending_list[:to_add])
book_rec = books[books.book_id.isin(final_list)]

print('------------------------------------------------------------------------')
print('Based on your rating records, we believe you will enjoy these books too!')
print('------------------------------------------------------------------------')
print()
print(book_rec)
print()
