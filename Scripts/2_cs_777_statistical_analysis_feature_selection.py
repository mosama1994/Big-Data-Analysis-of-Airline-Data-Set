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

"""# Statistical Testing

## Correlation
"""

df_spark_2 = sql_c.read.csv(input_file, header=True, inferSchema=True)

df_spark_2.persist(StorageLevel.MEMORY_ONLY)

# Changing all catagorical columns to string index

def def_cat_num_cols (x):
  categorical_cols = []
  numerical_cols = []

  col_types = x.dtypes

  for i in col_types:
    if i[1] == 'string':
      categorical_cols += [i[0]]
    else:
      if i[0] not in ['Year','Delay_Status']:
        numerical_cols += [i[0]]
  
  return categorical_cols, numerical_cols

categorical_cols, numerical_cols = def_cat_num_cols(df_spark_2)

for i in categorical_cols:
  df_spark_2 = StringIndexer(inputCol=i, outputCol=i + '_string', handleInvalid='skip').fit(df_spark_2).transform(df_spark_2)

df_spark_2 = VectorAssembler(inputCols=[i + '_string' for i in categorical_cols]+numerical_cols+['Delay_Status'], outputCol='vector_assem_all', ).transform(df_spark_2)

r2 = Correlation.corr(df_spark_2, 'vector_assem_all').collect()[0][0].toArray()

pd.DataFrame(r2).to_csv(output_location + '/corr_data_before.csv', index=False)

df_spark_2 = df_spark_2.drop('Origin', 'Dest', 'Distance', 'Quarter', 'vector_assem_all', 'Origin_string', 'Dest_string', 'Distance_string', 'Quarter_string', 'Diverted')

df_spark_2

categorical_cols, numerical_cols = def_cat_num_cols(df_spark_2)

temp = []

for i in categorical_cols:
  if i.find('_string') != -1:
    categorical_cols.remove(i)

for i in numerical_cols:
  if i.find('_string', 0) == -1:
    temp.append(i)

numerical_cols = temp

df_spark_2 = VectorAssembler(inputCols=[i + '_string' for i in categorical_cols]+numerical_cols+['Delay_Status'], outputCol='vector_assem_all').transform(df_spark_2)

r2 = Correlation.corr(df_spark_2, 'vector_assem_all').collect()[0][0].toArray()

pd.DataFrame(r2).to_csv(output_location + '/corr_data_after.csv', index=False)

df_spark_2 = VectorAssembler(inputCols=[i + '_string' for i in categorical_cols], outputCol='vector_assem_cat', ).transform(df_spark_2)

r = ChiSquareTest.test(df_spark_2, featuresCol='vector_assem_cat', labelCol='Delay_Status').collect()

pd.DataFrame({'features': categorical_cols,'pValue': r[0][0], 'degreesOfFreedom': r[0][1], 'statistics': r[0][2]}).to_csv(output_location + '/chi_square_statistic.csv', index=False)

# df_spark_2.coalesce(1).write.mode("overwrite").options(header=True).csv(output_location + '/Post_Statistical_Analysis')

"""Feature Selection"""

# create a assembly of numerical features and categorical features
df_spark_2 = VectorAssembler(inputCols=numerical_cols, outputCol='vector_assem_num').transform(df_spark_2)

# selecting from categorical features
selector = UnivariateFeatureSelector(featuresCol="vector_assem_cat", outputCol="uni_cat_selectedFeatures",
                                     labelCol="Delay_Status", selectionMode="numTopFeatures")
selector.setFeatureType("categorical").setLabelType("categorical").setSelectionThreshold(3)

model_cat_uni = selector.fit(df_spark_2)

# selecting from numerical features
selector = UnivariateFeatureSelector(featuresCol="vector_assem_num", outputCol="uni_num_selectedFeatures",
                                     labelCol="Delay_Status", selectionMode="numTopFeatures")
selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(7)

model_num_uni = selector.fit(df_spark_2)

uni_cat_selected_features = [categorical_cols[i] for i in model_cat_uni.selectedFeatures]

uni_num_selected_features = [numerical_cols[i] for i in model_num_uni.selectedFeatures]

pd.DataFrame({'selected_features': uni_cat_selected_features}).to_csv(output_location + '/univariate_categorical_feature_selection.csv', index=False)

pd.DataFrame({'selected_features': uni_num_selected_features}).to_csv(output_location + '/univariate_numerical_feature_selection.csv', index=False)

summarizer = Summarizer.metrics('variance')

val_5 = sorted(df_spark_2.select(summarizer.summary(df_spark_2.vector_assem_num)).collect()[0][0][0].toArray(), reverse=True)[5]

var_selector = VarianceThresholdSelector(featuresCol='vector_assem_num',outputCol='var_num_selectedFeatures', varianceThreshold=val_5)
model_var = var_selector.fit(df_spark_2)

var_selected_features = [numerical_cols[i] for i in model_var.selectedFeatures]

pd.DataFrame({'selected_features': var_selected_features}).to_csv(output_location + '/variance_feature_selection.csv', index=False)

df_spark_2 = df_spark_2.drop('vector_assem_all', 'vector_assem_cat', 'vector_assem_num')

df_spark_2 = df_spark_2.select([i for i in df_spark_2.columns if i.find('_string')==-1])

df_spark_2.coalesce(1).write.mode("overwrite").options(header=True).csv(output_location + '/Post_Statistical_Feature_Selection')