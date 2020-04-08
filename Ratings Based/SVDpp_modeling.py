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

# Use new csv for further analysis
ratings = pd.read_csv("ratings_clean.csv", index_col=0)
ratings = ratings.drop(columns=['counts'])
print(ratings.head())

# Algorithm performance
# print(ratings.head())
reader = Reader(rating_scale=(0, 9))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

# SVDpp gives the best performance, build recommender
algo = SVDpp()
perf = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)
print(perf)

# train and predict
trainset, testset = train_test_split(data, test_size=0.25)
print()
print('####### testset #########')
print(testset[:10])
print()


def predict(train, test):
    prediction = algo.fit(train).test(test)
    return prediction


predictions = predict(trainset, testset)
accuracy.rmse(predictions)


# error analysis


def get_Iu(uid):
    """ return the number of items rated by given user
    args:
      uid: the id of the user
    returns:
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError:  # user was not part of the trainset
        return 0


def get_Ui(bid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try:
        return len(trainset.ir[trainset.to_inner_iid(bid)])
    except ValueError:
        return 0


df = pd.DataFrame(predictions, columns=['uid', 'bid', 'rui', 'est', 'details'])
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.bid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)
best_predictions = df.sort_values(by='err')[:10]
worst_predictions = df.sort_values(by='err')[-10:]

best_predictions.to_csv('best_predictions.csv')
worst_predictions.to_csv('worst_predictions.csv')

print(best_predictions)
print(worst_predictions)
