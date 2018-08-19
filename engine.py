from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('recommender').getOrCreate()
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from pyspark.sql.functions import struct, collect_list, explode
import json
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import FloatType


data_cols = ['userid','movieid','rating','timestamp']
item_cols = ['movieid','movietitle','release date',
'video release date','IMDb URL','unknown','Action',
'Adventure','Animation','Childrens','Comedy','Crime',
'Documentary','Drama','Fantasy','Film-Noir','Horror',
'Musical','Mystery','Romance ','Sci-Fi','Thriller',
'War' ,'Western']
user_cols = ['userid','age','gender','occupation',
'zip code']

users = pd.read_csv('/home/bella/ml-100k/u.user', sep='|',
names=user_cols, encoding='latin-1')

item = pd.read_csv('/home/bella/ml-100k/u.item', sep='|',
names=item_cols, encoding='latin-1')
data = pd.read_csv('/home/bella/ml-100k/u.data', sep='\t',
names=data_cols, encoding='latin-1')

dataset = pd.merge(pd.merge(item, data),users)
df = dataset[['userid','movieid','movietitle','rating','timestamp']]
dataframe =spark.createDataFrame(df)
#ratings = dataframe.rdd

#training the model

training, test = dataframe.randomSplit([0.8,0.2])

als = ALS(maxIter=5, regParam=0.01, userCol='userid', itemCol='movieid', ratingCol='rating')

model = als.fit(training)

predictions = model.transform(test)

predictions = predictions.na.drop()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)

users = dataframe.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
movies = dataframe.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)

# userRecs.show()
# movieRecs.show()
userSubsetRecs.show()
movieSubSetRecs.show()

_df = userSubsetRecs\
.select(explode(userSubsetRecs.recommendations.movieid),'userid')\
#.withColumn("rating",explode(userSubsetRecs.recommendations.movieid))

_df1 = userSubsetRecs\
.withColumn('rating',explode(userSubsetRecs.recommendations.rating))\
.withColumn('movieid',explode(userSubsetRecs.recommendations.movieid))\

rd=_df1.drop('recommendations').collect()

schema = StructType([StructField("userId", StringType(), True), StructField("rating", FloatType(), True), StructField("movieid", StringType(), True)])

dfToSave = spark.createDataFrame(rd, schema)

dataframe.createOrReplaceTempView("m")
dfToSave.createOrReplaceTempView("re")

sql =spark.sql("select  re.userid,m.movietitle  from re join m on re.movieid= m.movieid group by re.userid,m.movietitle order by userid")

_delimiter=','

_output='/home/bella/recommendations'

_xy=sql.coalesce(1).write.format('com.databricks.spark.csv').option('header','true').option('delimiter', _delimiter).mode("overwrite").save(_output)

