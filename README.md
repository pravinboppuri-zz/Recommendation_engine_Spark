# Recommendation_System_Spark

https://grouplens.org/datasets/movielens/

A collabrative filtering recommendation model built in python with some of the famous open source data sets available online.
More information on the model can be found from the links below:

https://spark.apache.org/docs/latest/ml-collaborative-filtering.html

# Why collabrative model?

lets say you have sales application that does daily sales transactions based on your customers  & products, but you are always dependent on the sales report from the sales team to identify top rated customers or top rated products that are purchased and etc so that you can build your sales model to give the best customer experiecence. Why dont Spark Machine learning alogrithms do the job and recommend your customers the best products based on their transaction history and vise versa automatically. That where these recommendation engines come into picture. 

You would require a rating system built from your products, you could build this easily from your database or you could build one with pandas frame work. 
collabrative model filters data from other users, who like the same products the user views or likes, also liked a recommended product.

# Train the Model

Spark MLIB requires you to train the model using the ALS algorithm and then apply to your final dataset

from pyspark.mllib.recommendation import ALS
model = ALS.train(training, rank = 10, iterations = 5, lambda_=0.01)

