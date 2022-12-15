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
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
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

lr = LogisticRegression()

pipeline_full_data = pipeline_creator(categorical_cols, numerical_cols, lr)
pipeline_uni_data = pipeline_creator(catagorical_cols_uni, numerical_cols_uni, lr)
pipeline_var_data = pipeline_creator(categorical_cols, numerical_cols_var, lr)

paramGrid_lr = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01, 0.05]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

def cv_lr_classification(pipeline):
  cross_val = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid_lr,
                            evaluator=bcEvaluator,
                            numFolds=10,
                            parallelism=10)
  model = cross_val.fit(train)
  predictions = model.transform(test)
  best_parameters = sorted(list(zip(model.avgMetrics,model.getEstimatorParamMaps())), key= lambda x: x[0], reverse=True)[0]
  best_parameters = [best_parameters[1][i] for i in best_parameters[1].keys()]
  area_roc = bcEvaluator.evaluate(predictions)
  accuracy = mcEvaluator_accuracy.evaluate(predictions)
  tpr = mcEvaluator_tpr.evaluate(predictions)
  fpr = mcEvaluator_fpr.evaluate(predictions)
  precision = mcEvaluator_precision.evaluate(predictions)
  f1_score = mcEvaluator_f1.evaluate(predictions)
  df = pd.DataFrame({'parameter': ['regularization', 'elasticNet', 'area_ROC', 'accuracy', 'tpr', 'fpr', 'precision', 'f1_score'],\
                      'values': best_parameters + [area_roc,accuracy,tpr,fpr,precision,f1_score]})
  return df

cv_lr_full_data_reg = cv_lr_classification(pipeline_full_data)
cv_lr_uni_data_reg = cv_lr_classification(pipeline_uni_data)
cv_lr_var_data_reg = cv_lr_classification(pipeline_var_data)

cv_lr_full_data_reg.to_csv(output_location + '/CV_Logistic_Regression/CV_Logistic_Regression_Regularization_Full_Data_Metrics.csv', index=False)
cv_lr_uni_data_reg.to_csv(output_location + '/CV_Logistic_Regression/CV_Logistic_Regression_Regularization_Uni_Data_Metrics.csv', index=False)
cv_lr_var_data_reg.to_csv(output_location + '/CV_Logistic_Regression/CV_Logistic_Regression_Regularization_Var_Data_Metrics.csv', index=False)