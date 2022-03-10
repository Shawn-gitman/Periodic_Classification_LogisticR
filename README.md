# Periodic_Classification_LogisticR(PCL) - Summer Intern Research Project(2021)

PCL(Periodic Classification Logistic) is a binary classification algorithm that utilizes Logistic Regression to classify time series periodcity. We utilize Python, Tensorflow, pandas, and numpy to deploy PCL. PCL has it's own labeling, pre-processing, and normalizing method to predict periodcity effectively.

## Definition of Periodcity

Time-series is divided into two classes, periodc time series and non-periodc time series. Periodc time series has regular seasonality without noise or pollution. 

![Watch the video](resource1.png)

## Labeling Time Series

[Step.1] Run main.py in Anaconda Virtual Environment
```rb
(timeseries_env) C:\Users\taegu\Desktop\인턴자료\PeriodicClassification>python logistic_regression.py
```
[Step.2] Define periodic or non-periodic time series.

[Step.3] Label Time-Series

If it has periodcity, type
```rb
Type the label(ex: 1- Periodic, 2- Non-Periodic): 1
```

If it has non- periodcity, type
```rb
Type the label(ex: 1- Periodic, 2- Non-Periodic): 2
```

## Normalizing Time Series

Normalizing can unify scale, range, and regularity of time series dataset. 

![Watch the video](resource2.png)

## Divide into Training and Test set

We separted our dataset into 80% of training and 20% of test time series dataset.

## Hyperparameters

We use 3 parameters of learning_rate, batch_size, and iteration number.

```rb
learning_rate = 0.003
batch_size = 30
iter_num = 50000
```

## Evaluate PCL


