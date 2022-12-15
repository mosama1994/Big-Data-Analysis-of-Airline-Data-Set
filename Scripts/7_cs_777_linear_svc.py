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
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier, LinearSVC
from pyspark.ml.feature import UnivariateFeatureSelector, VarianceThresholdSelector, PCA
from pyspark.ml.stat import ChiSquareTest, Summarizer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.storagelevel import StorageLevel

input_file_1 = sys.argv[1]
input_file_2 = sys.argv[2]
input_file_3 = sys.argv[3]
input_file_4 = sys.argv[4]
output_location = sys.argv[5]

sc = SparkContext()

sql_c = SQLContext(sc)

"""# Machine Learning"""

df_spark_2 = sql_c.read.csv(input_file_1, header=True, inferSchema=True)

df_spark_2.persist(StorageLevel.MEMORY_ONLY)

numerical_cols_var = pd.read_csv(input_file_2).iloc[:,0].tolist()
catagorical_cols_uni = pd.read_csv(input_file_3).iloc[:,0].tolist()
numerical_cols_uni = pd.read_csv(input_file_4).iloc[:,0].tolist()

numerical_cols = []
categorical_cols = []

for i in df_spark_2.dtypes:
  if i[1] == 'string':
    categorical_cols.append(i[0])
  elif i[0] not in ['Year', 'Delay_Status'] :
    numerical_cols.append(i[0])

df_spark_2 = df_spark_2.withColumnRenamed('Delay_Status', 'label')

train = df_spark_2.filter(col('Year') != 2022)
test = df_spark_2.filter(col('Year') == 2022)

# Creating a function to create a pipline
def pipeline_creator (categorical_cols, numerical_cols, estimator):
  string_indexor = [StringIndexer(inputCol=i, outputCol=i + '_string', handleInvalid='skip') for i in categorical_cols]
  encoding = [OneHotEncoder(inputCol=f'{i}_string', outputCol=i + "_encoded") for i in categorical_cols]
  vector_assembler_1 = VectorAssembler(inputCols=numerical_cols, outputCol='vector_num')
  scaled = StandardScaler(inputCol='vector_num', outputCol='vector_num_scaled')
  inputAssembler = [f'{i}_encoded' for i in categorical_cols]
  inputAssembler += ['vector_num_scaled']
  vector_assembler_2 = VectorAssembler(inputCols=inputAssembler, outputCol='features')
  stages = []
  stages += string_indexor
  stages += encoding
  stages += [vector_assembler_1]
  stages += [scaled]
  stages += [vector_assembler_2]
  stages += [estimator]
  pipeline = Pipeline().setStages(stages)
  return pipeline

bcEvaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
mcEvaluator_tpr = MulticlassClassificationEvaluator(metricName="truePositiveRateByLabel")
mcEvaluator_precision = MulticlassClassificationEvaluator(metricName="precisionByLabel")
mcEvaluator_fpr = MulticlassClassificationEvaluator(metricName="falsePositiveRateByLabel")
mcEvaluator_accuracy = MulticlassClassificationEvaluator(metricName="accuracy")
mcEvaluator_f1 = MulticlassClassificationEvaluator(metricName='f1')

lsvc = LinearSVC()

pipeline_full_data_lsvc = pipeline_creator(categorical_cols, numerical_cols, lsvc)
pipeline_uni_data_lsvc = pipeline_creator(catagorical_cols_uni, numerical_cols_uni, lsvc)
pipeline_var_data_lsvc = pipeline_creator(categorical_cols, numerical_cols_var, lsvc)

paramGrid_lsvc = ParamGridBuilder()\
    .addGrid(lsvc.regParam, [0.0, 0.3, 0.5, 1.0, 2.0])\
    .build()

def lsvc_classification (pipeline):
  tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid_lsvc,
                           evaluator=bcEvaluator,
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)
  model = tvs.fit(train)
  predictions = model.transform(test)
  best_parameters = sorted(list(zip(model.validationMetrics,model.getEstimatorParamMaps())), key= lambda x: x[0], reverse=True)[0]
  best_parameters = [best_parameters[1][i] for i in best_parameters[1].keys()]
  area_roc = bcEvaluator.evaluate(predictions)
  accuracy = mcEvaluator_accuracy.evaluate(predictions)
  tpr = mcEvaluator_tpr.evaluate(predictions)
  fpr = mcEvaluator_fpr.evaluate(predictions)
  precision = mcEvaluator_precision.evaluate(predictions)
  f1_score = mcEvaluator_f1.evaluate(predictions)
  df = pd.DataFrame({'parameter': ['reg_param', 'area_ROC', 'accuracy', 'tpr', 'fpr', 'precision', 'f1_score'],\
                     'values': best_parameters + [area_roc,accuracy,tpr,fpr,precision,f1_score]})
  return df

lsvc_full_data = lsvc_classification(pipeline_full_data_lsvc)
lsvc_uni_data = lsvc_classification(pipeline_uni_data_lsvc)
lsvc_var_data = lsvc_classification(pipeline_var_data_lsvc)

lsvc_full_data.to_csv(output_location + '/Linear_SVC/Linear_SVC_Full_Data_Metrics.csv', index=False)
lsvc_uni_data.to_csv(output_location + '/Linear_SVC/Linear_SVC_Uni_Data_Metrics.csv', index=False)
lsvc_var_data.to_csv(output_location + '/Linear_SVC/Linear_SVC_Var_Data_Metrics.csv', index=False)