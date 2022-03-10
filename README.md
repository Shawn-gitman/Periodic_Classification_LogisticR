# Periodic_Classification_LogisticR(PCL) - Summer Intern Research Project(2021)

PCL(Periodic Classification Logistic) is a binary classification algorithm that utilizes Logistic Regression to classify time series periodcity. We utilize Python, Tensorflow, pandas, and numpy to deploy PCL. PCL has it's own labeling, pre-processing, and normalizing method to predict periodcity effectively.

## Definition of Periodcity

Time-series is divided into two classes, periodc time series and non-periodc time series. Periodc time series has regular seasonality without noise or pollution. 

![Watch the video](resource1.png)

## Labeling Time Series

[Step.1] Run main.py in Anaconda virtual environment
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
```rb
def dataset_split(X, y):
	seed = 5
	np.random.seed(seed)
	tf.set_random_seed(seed)

	# set replace=False, Avoid double sampling
	train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)

	# diff set
	test_index = np.array(list(set(range(len(X))) - set(train_index)))
	train_X = X[train_index]
	train_y = y[train_index]
	test_X = X[test_index]
	test_y = y[test_index]

	train_X = min_max_normalized(train_X)
	test_X = min_max_normalized(test_X)

	print(test_X)

	return train_X, train_y, test_X, test_y
 ```

## Hyperparameters

We use 3 parameters of learning_rate, batch_size, and iteration number.
```rb
learning_rate = 0.003
batch_size = 30
iter_num = 50000
```

## Evaluate PCL


