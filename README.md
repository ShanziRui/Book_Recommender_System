# Book_Recommender_System
Conjunction of two recommender engines, one is rating based and the other is content based.

There are three folders with PY files:
Contexts Based (PY): contains the processing files for contexts-based recommendation engine, including contexts_loading.py, contexts_modeling.py, contexts_preprocessing.py.

Ratings Based (PY): contains the processing files for ratings-based recommendation engine,  including rating_preprocessing.py, SVDpp_modeling.py.

Final Recommendation System (PY): contains the final recommendation system embedded with both two engines, named as recommender.py.


The Operation Flow
===================


Interactive Recommendation System Instance
------------------------------------------

Go to the "Final Recommendation System (PY)" folder, run recommender.py.

- System Greeting

-----------------------------------------------------------------------
Welcome to BookRecommender, enter your user_id or type 0 to register!
-----------------------------------------------------------------------


- If you are an existing user with a specific user_id, type in your user_id, type 7653 for testing.


- System gives recommended list of 5 books for user 7653.

------------------------------------------------------------------------
Based on your rating records, we believe you will enjoy these books too!
------------------------------------------------------------------------

  	id	book_id                                    title		     
	663      2767  A People's History of the United States
	1299     9566               Still Life with Woodpecker
	1928     3679                                On Beauty
	3141     6862                                Amsterdam
	3255     4708                 The Beautiful and Damned

- If you are a new user without user_id, type 0 to register, system gives your user_id.

-----------------------------------------------------------------------
Congratulations! You are successfully registered. Your user_id is 53425
-----------------------------------------------------------------------


- System provide a list of books for you rate, type your ratings in format of (book_id, rating), or type 0 for a new list of books.

-----------------------------------------------------------------------
                 List of Books Avaiable in our Database                
-----------------------------------------------------------------------

       id	book_id                                              title
       5774      3300                                    Inés of My Soul
       3698   1001896                              The Real Mother Goose
       9939   8034188  Where Good Ideas Come From: The Natural Histor...
       4302   2202049  Such a Pretty Fat: One Narcissist's Quest to D...
       6726    235773                                            Waiting
       4590     84785  The Overlook (Harry Bosch, #13; Harry Bosch Un...
       5984  13477819  Who Could That Be at This Hour? (All the Wrong...
       1250    412732                                    The Dharma Bums
       1023     35231                  Lord of Chaos (Wheel of Time, #6)
       4739   5989573  Scott Pilgrim, Volume 5: Scott Pilgrim Vs. the...
       5605  12924261  This Book Is Full of Spiders: Seriously, Dude,...
       7159     73100             This Heart of Mine (Chicago Stars, #5)
       2155     22076                                     From a Buick 8
       1877     33514                              The Elements of Style
       744       6708  The Power of Now: A Guide to Spiritual Enlight...

Please rate for at least three books above in format of (book_id, rating), split each by empty space. 1.0 indcates least liked and 5.0 indicates most liked, press 0 to fresh the book list.

- After you type in the rating records, system will give you the recommendation result.




Contexts-based Engine
----------------------

Run contexts_preprocessing.py to see the data preparation result, run contexts_modeling.py to see the contexts_recommder engine.

Here is a list of file output you will get.

- 10 folders, named by the corresponding book title, each contains the separated txt file of all chapters in that book.
	e.g. "pride-and-prejudice" contains 61 chapters in total, the program creates 61 txt files for each chapter, stores in the folder named "pride-and-prejudice-chapters"

- A csv file named "book_contexts.csv" which contains three conlumns and 10 rows, each row stands for one book with its corpus representations.




Ratings-based Engine
---------------------

Run rating_preprocessing.py to see the data preparation and basic statistics of the ratings data, run SVDpp_modeling.py to see the ratings_recommender engine.

The output information printed in order:

- A visualization named "distribution_of_ratings.png"

- A visualization named "mean_book_rating.png"

- A visualization named "mean_user_rating.png"

- A visualization named "number_of_ratings_per_book.png"

- A visualization named "number_of_ratings_per_user.png"

- A csv file named "performance.csv" contains the evaluation result for 11 algorithms.

