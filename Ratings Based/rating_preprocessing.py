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

# Read data from file 'ratings.csv'
ratings = pd.read_csv("ratings.csv")
# Preview the first 5 lines of the loaded data
print(ratings)


# Data Cleasning
# first remove the duplicate ratings
rating = ratings.groupby(['book_id', 'user_id'])[
    'rating'].mean().astype(int).tolist()
# print(type(rating))

ratings = ratings.groupby(['book_id', 'user_id']
                          ).size().reset_index(name='counts')
ratings['rating'] = rating
print(ratings)
# print(ratings.head())

print('Number of duplicate ratings: ', len(ratings[ratings['counts'] > 1]))

ratings = ratings[ratings['counts'] == 1]
print(ratings)
nrow_bf = len(ratings)  # 977269

# remove users who rated fewer than 3 books
grp = ratings.groupby('user_id')
ratings = grp.filter(lambda x: len(x) >= 3)
print(ratings)

nrow_af = len(ratings)  # 960595

print('Number users who rated fewer than 3 books: ', (nrow_bf - nrow_af))

# print(ratings.head())


# Select a subset of users
random.seed(0)
sampling_fraction = 0.2
users = ratings['user_id'].unique().tolist()
sample_users = sample(users, round(sampling_fraction * len(users)))
print('Number of ratings (before): ', nrow_af)

ratings = ratings.loc[ratings['user_id'].isin(sample_users)]
print('Number of ratings (after): ', len(ratings))


# save ratings to csv
ratings.to_csv('ratings_clean.csv')


# Use new csv for further analysis
ratings = pd.read_csv("ratings_clean.csv", index_col=0)
ratings = ratings.drop(columns=['counts'])
print(ratings.head())


# Exploration (basic statistics of the data)
# Distribution of ratings
sns.set(style="whitegrid")
ax = sns.countplot(x="rating", data=ratings)
plt.savefig("distribution_of_ratings.png", dpi=300, bbox_inches='tight')
plt.clf()

# Number of ratings per user
per_user = ratings.groupby('user_id').size().reset_index(
    name='number_of_ratings_per_user')
ax1 = sns.countplot(x='number_of_ratings_per_user', data=per_user)
i = 0
for label in ax1.get_xticklabels():
    if i % 20 == 0:
        i = i+1
        continue
    label.set_visible(False)
    i = i+1
# for label in ax1.get_xticklabels()[::10]:
#     label.set_visible(True)
plt.savefig("number_of_ratings_per_user.png", dpi=300, bbox_inches='tight')
plt.clf()


# Distribution of mean user ratings
mean_user = ratings.groupby(
    'user_id', as_index=False).agg({"rating": "mean"})
ax2 = sns.distplot(mean_user['rating'], kde=False)
plt.savefig("mean_user_rating.png", dpi=300, bbox_inches='tight')
plt.clf()


# Number of ratings per book
per_book = ratings.groupby('book_id').size(
).reset_index(name='number_of_ratings_per_book')
ax3 = sns.countplot(x='number_of_ratings_per_book', data=per_book)
for label in ax3.get_xticklabels()[::2]:
    label.set_visible(False)
plt.savefig("number_of_ratings_per_book.png", dpi=300, bbox_inches='tight')
plt.clf()

# Distribution of mean book ratings
mean_book = ratings.groupby(
    'book_id', as_index=False).agg({"rating": "mean"})
ax4 = sns.distplot(mean_book['rating'], kde=False)
plt.savefig("mean_book_rating.png", dpi=300, bbox_inches='tight')
plt.clf()


# Algorithm performance
# print(ratings.head())
reader = Reader(rating_scale=(0, 9))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)


benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=[
                             'RMSE', 'MAE'], cv=3, verbose=False)

    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(
        ' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)

performance = pd.DataFrame(benchmark).set_index(
    'Algorithm').sort_values('test_rmse')
performance.to_csv('performance.csv')
print(performance)
