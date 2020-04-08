import random
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import numpy as np
import pandas as pd
import context_load as load
import glob
import os

txt_files = glob.glob("*.txt")

# load each book from plain text
books = []
for txt in txt_files:
    books.append(load.Book(txt, False,
                           False, os.getcwd()))


print("*"*12+" Top 10 most frequently downloaded books "+"*"*12)
for book in books:
    print(book.title, book.author, sep=': ')
print()

# build dataframe each stands for one chapter
stemmer = SnowballStemmer('english')
words = set(stopwords.words('english'))
population = pd.DataFrame([], columns=['Title', 'Author', 'Documents'])

for book in books:

    for chapter in book.chapters:
        Title = book.title
        Author = book.author

        Contents = " ".join(chapter)
        # remove stopwords and stemmer
        Clean = " ".join([stemmer.stem(i) for i in re.sub(
            "[^a-zA-Z]", " ", Contents).split() if i not in words]).lower()
        Tokens = Clean.split(' ')
        NumDoc = len(Tokens)//150
        if NumDoc == 0:
            Documents = []
        else:
            Documents = np.array_split(Tokens, NumDoc)
        df2 = pd.DataFrame([[Title, Author, Documents]], columns=[
                           'Title', 'Author', 'Documents'])
        population = population.append(df2, ignore_index=True)


training = pd.DataFrame([], columns=['Author', 'title', 'corpus'])

for book in books:
    corpus = []
    Title = book.title
    Author = book.author
    bk = population[population['Author'] == Author]
    bk = bk[['Author', 'Title', 'Documents']]
    Documents = []
    for index, row in bk.iterrows():
        for docs in row['Documents']:
            # doc = ' '.join(docs)
            Documents.append(docs)
    Samples = random.sample(Documents, 100)
    for Document in Samples:
        corpus.extend(Document.tolist())

    df3 = pd.DataFrame([[Author, Title, corpus]], columns=[
        'Author', 'title', 'corpus'])
    training = training.append(df3, ignore_index=True)

# till now, each row in training stands for corpus representation of each book

# add book_id
training['book_id'] = [1852, 1953, 3590, 1885,
                       18254, 18490, 295, 78950, 153747, 2956]
training = training[['book_id', 'title', 'corpus']]
training.to_csv('book_contexts.csv')
print(training)
