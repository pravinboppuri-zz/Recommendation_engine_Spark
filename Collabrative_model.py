from __future__ import print_function

import sys
import os
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import FloatType
import pandas as pd


conf = SparkConf().setAppName("app_collaborative")
sc = SparkContext(conf=conf)
sc.setCheckpointDir('checkpoint/')
sqlContext = SQLContext(sc)

def howFarAreWe(model, against, sizeAgainst):

  againstNoRatings = against.map(lambda x: (int(x[0]), int(x[1])) )
  againstWiRatings = against.map(lambda x: ((int(x[0]),int(x[1])), int(x[2])) )
  predictions = model.predictAll(againstNoRatings).map(lambda p: ( (p[0],p[1]), p[2]) )
  predictionsAndRatings = predictions.join(againstWiRatings).values()
  return sqrt(predictionsAndRatings.map(lambda s: (s[0] - s[1]) ** 2).reduce(add) / float(sizeAgainst))

url ='/home/bella/Downloads/ratings.csv'
df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load(url)

#rddUserRatings = df.filter(df.userid == 0).rdd
rddUserRatings = df.filter(df.userid == 0).rdd
print(rddUserRatings.count())

# Split the data in 3 different sets : training, validating, testing
# 60% 20% 20%
rddRates = df.rdd
rddTraining, rddValidating, rddTesting = rddRates.randomSplit([6,2,2])

#Add user ratings in the training model
rddTraining.union(rddUserRatings)
nbValidating = rddValidating.count()
nbTesting    = rddTesting.count()

print("Training: %d, validation: %d, test: %d" % (rddTraining.count(), nbValidating, rddTesting.count()))

ranks  = [5,10,15,20]
reguls = [0.1, 1,10]
iters  = [5,10,20]

finalModel = None
finalRank  = 0
finalRegul = float(0)
finalIter  = -1
finalDist   = float(100)

for cRank, cRegul, cIter in itertools.product(ranks, reguls, iters):

  model = ALS.train(rddTraining, cRank, cIter, float(cRegul))
  dist = howFarAreWe(model, rddValidating, nbValidating)
  if dist < finalDist:
    print("Best so far:%f" % dist)
    finalModel = model
    finalRank  = cRank
    finalRegul = cRegul
    finalIter  = cIter
    finalDist  = dist
#[END train_model]

print("Rank %i" % finalRank)
print("Regul %f" % finalRegul)
print("Iter %i" % finalIter)
print("Dist %f" % finalDist)

sc.stop()