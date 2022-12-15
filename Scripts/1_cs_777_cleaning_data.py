import sys
from pyspark.sql import SQLContext
from pyspark import SparkContext
import numpy as np
import pandas as pd
from pyspark.sql.functions import when, regexp_extract, col, split, to_timestamp
from pyspark.sql.types import IntegerType, StringType, DoubleType
from pyspark.ml.stat import Correlation
import seaborn as sns
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import UnivariateFeatureSelector, VarianceThresholdSelector, PCA
from pyspark.ml.stat import ChiSquareTest, Summarizer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.storagelevel import StorageLevel

input_file = sys.argv[1]
output_location = sys.argv[2]

sc = SparkContext()

sql_c = SQLContext(sc)

"""# Working with small dataset"""

df_spark_2 = sql_c.read.csv(input_file, header=True, inferSchema=True)

df_spark_2.persist(StorageLevel.MEMORY_ONLY)

# Remove 'CRSDepTime', 'DepDelayMinutes', 'ArrDelayMinutes', 'CRSElapsedTime', ActualElapsedTime, 'Marketing_Airline_Network', 'Operated_or_Branded_Code_Share_Partners',
#      'DOT_ID_Marketing_Airline', 'IATA_Code_Marketing_Airline', 'Flight_Number_Marketing_Airline', 'Operating_Airline',
#       'DOT_ID_Operating_Airline', 'IATA_Code_Operating_Airline', 'Tail_Number', 'Flight_Number_Operating_Airline', 'Flight_Number_Operating_Airline',
#       'OriginAirportID', 'OriginAirportSeqID', 'OriginCityMarketID', 'OriginStateFips', 'OriginStateName', 'OriginWac',
#       'DestAirportID', 'DestAirportSeqID', 'DestCityMarketID', 'DestStateFips', 'DestStateName', 'DestWac', 'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk'
#       'TaxiOut', 'WheelsOff', 'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrDel15', 'ArrivalDelayGroups', 'ArrTimeBlk', 'DistanceGroup'

remove_cols_list = ['FlightDate', 'CRSDepTime', 'DepDelayMinutes', 'ArrDelayMinutes', 'CRSElapsedTime', 'ActualElapsedTime', 'Marketing_Airline_Network', 'Operated_or_Branded_Code_Share_Partners',
                    'DOT_ID_Marketing_Airline', 'IATA_Code_Marketing_Airline', 'Flight_Number_Marketing_Airline', 'Operating_Airline',
                    'DOT_ID_Operating_Airline', 'IATA_Code_Operating_Airline', 'Tail_Number', 'Flight_Number_Operating_Airline', 'Flight_Number_Operating_Airline',
                    'OriginAirportID', 'OriginAirportSeqID', 'OriginCityMarketID', 'OriginStateFips', 'OriginStateName', 'OriginWac',
                    'DestAirportID', 'DestAirportSeqID', 'DestCityMarketID', 'DestStateFips', 'DestStateName', 'DestWac', 'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk',
                    'TaxiOut', 'WheelsOff', 'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrDel15', 'ArrivalDelayGroups', 'ArrTimeBlk', 'DistanceGroup']

df_spark_2 = df_spark_2.select([_ for _ in df_spark_2.columns if _ not in remove_cols_list])

df_spark_3 = df_spark_2.filter(col('Cancelled') == True)

# Keeping only not cancelled flights
df_spark_2 = df_spark_2.filter(col('Cancelled') == False)

# Remove rows with null values
df_spark_2 = df_spark_2.na.drop()

df_spark_3 = df_spark_3.union(df_spark_2)

# Saving this dataset for visualization
df_spark_3.coalesce(1).write.mode("overwrite").options(header=True).csv(output_location + '/Flight_Small_Data_Visualization')

df_spark_2 = df_spark_2.drop(col('Cancelled'))

df_spark_2.schema

# Setting a column delay status which will be the predicted variable (flight delayed or not)
df_spark_2 = df_spark_2.withColumn('Delay_Status', when((df_spark_2['DepDelay'] <= 0) & (df_spark_2['ArrDelay'] <= 0), 0).otherwise(1))

# Dropping columns DepDelay and ArrDelay because they directly drive the delay status
col_drop = ['DepDelay', 'ArrDelay']
df_spark_2 = df_spark_2.drop(*col_drop)

# Removing state from city name
df_spark_2 = df_spark_2.withColumn('OriginCityName', split(col('OriginCityName'),',').getItem(0))
df_spark_2 = df_spark_2.withColumn('DestCityName', split(col('DestCityName'),',').getItem(0))

# Splitting arrival and departure times into columns with hours and minutes
df_spark_2 = df_spark_2.withColumn('DepTimeHour', when(col("DepTime") >= 1000, col("DepTime").cast(IntegerType()).cast(StringType()).substr(0,2))\
                                   .when((col("DepTime") >= 100) & (col("DepTime") < 1000), col("DepTime").cast(IntegerType()).cast(StringType()).substr(0,1))\
                                   .otherwise(0))
df_spark_2 = df_spark_2.withColumn('DepTimeMinute', col("DepTime").cast(IntegerType()).cast(StringType()).substr(-2,2))

df_spark_2 = df_spark_2.withColumn('ArrTimeHour', when(col("ArrTime") >= 1000, col("ArrTime").cast(IntegerType()).cast(StringType()).substr(0,2))\
                                   .when((col("ArrTime") >= 100) & (col("ArrTime") < 1000), col("ArrTime").cast(IntegerType()).cast(StringType()).substr(0,1))\
                                   .otherwise(0))
df_spark_2 = df_spark_2.withColumn('ArrTimeMinute', col("ArrTime").cast(IntegerType()).cast(StringType()).substr(-2,2))

df_spark_2 = df_spark_2.withColumn('DepTimeHour', col('DepTimeHour').cast(IntegerType()))\
.withColumn('DepTimeMinute', col('DepTimeMinute').cast(IntegerType()))\
.withColumn('ArrTimeHour', col('ArrTimeHour').cast(IntegerType()))\
.withColumn('ArrTimeMinute', col('ArrTimeMinute').cast(IntegerType()))

# Dropping the arrival and departure times now
col_drop = ['DepTime', 'ArrTime']
df_spark_2 = df_spark_2.drop(*col_drop)

df_spark_2 = df_spark_2.withColumn('Diverted', col('Diverted').cast(IntegerType()))

print('Reached Here')

df_spark_2.coalesce(1).write.mode("overwrite").options(header=True).csv(output_location + '/Full_Cleaned_Data')