from PeriodicClassification import ModelConfig as myConfig
from PeriodicClassification import Preprocess as pre
import numpy as np 
import seaborn as sns
sns.set(style='whitegrid')
from pandas import read_csv
import pandas as pd 
from matplotlib import pyplot
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers
from os import listdir
from os.path import isfile, join
import datetime

tf.compat.v1.disable_v2_behavior()

print(tf.__version__)

def _main(path, time_series_length):
	dataset = []
	labels = []

	label_list_length, onlyfiles = dataset_f(path)

	df, df_labels = labeling_df(label_list_length,onlyfiles, time_series_length, labels, dataset)

	X, y = df_normalization(df, df_labels)
 

	train_X, train_y, test_X, test_y = dataset_split(X, y)

	logistic_R(train_X, train_y, test_X, test_y, time_series_length)

def dataset_f(path):
	onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
	for i in range(len(onlyfiles)):
	    onlyfiles[i] = str(path)+str("/")+str(onlyfiles[i])

	label_list_length = len(onlyfiles)

	return label_list_length, onlyfiles

def labeling_df(label_list_length, onlyfiles, time_series_length, labels, dataset):
	#preparing dataset & labeling
	for i in range(label_list_length):
	    print("*************************************"+"("+str(i)+"/"+str(label_list_length)+")"+"*************************************")

	    print(onlyfiles[i])
	    list_time_series = pre._reader(onlyfiles[i])	
	    time_series = pre._resize(list_time_series)

	    series = read_csv(onlyfiles[i], header=0, index_col=0, parse_dates=True, squeeze=True)
	    print("File Location"+onlyfiles[i])
	    series_index=[]
	    series_index.append(series.index[0])

	    print(series_index)
	    print(list_time_series[0][:time_series_length])


	    if isinstance(list_time_series[0][:time_series_length],datetime.datetime):
	        series.plot()
	        pyplot.show()
	    else:
	        import matplotlib.pyplot as pp
	        pp.plot(list_time_series[0][:time_series_length])
	        pp.show()

	    train_set = []
	    for i in range(0,time_series_length):
	    	train_set.append(list_time_series[0][i])

	    label = input("Type the label(ex: 1- Periodic, 2- Non-Periodic): ")
	    pre_label = 0

	    if int(label) == 1:
	    	pre_label = 0
	    else:
	    	pre_label = 1
	    labels.append(pre_label)
	    #train_set.append(pre_label)
	    dataset.append(train_set)

	#visualize dataset
	df = pd.DataFrame(dataset)
	df_labels=pd.DataFrame(labels)
	print(df)

	return df, df_labels

def df_normalization(df, df_labels):
	#dataset normalization
	train_stats = df.describe()
	train_stats = train_stats.transpose()
	print(train_stats)

	normed_train_data = norm(df, train_stats)
	normed_train_data = pd.DataFrame(normed_train_data)
	print("normed_train_data")
	print(normed_train_data)

	#df[-1] = df[-1].replace(to_replace=[2, 4], value=[0, 1])

	X = normed_train_data.values
	y = df_labels[0].values

	print(X)
	print(y)

	return X, y

def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']

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

def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)

def logistic_R(train_X, train_y, test_X, test_y, time_series_length):
	# Begin building the model framework
	# Declare the variables that need to be learned and initialization
	# There are 4 features here, A's dimension is (4, 1)
	A = tf.Variable(tf.random.normal(shape=[(time_series_length), 1]))
	b = tf.Variable(tf.random.normal(shape=[1, 1]))
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)


	data = tf.placeholder(dtype=tf.float32, shape=[None, (time_series_length)])
	target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

	# Declare the model you need to learn
	mod = tf.matmul(data, A) + b

	# Declare loss function
	# Use the sigmoid cross-entropy loss function,
	# first doing a sigmoid on the model result and then using the cross-entropy loss function
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target))

	# Define the learning rate， batch_size etc.
	learning_rate = 0.003
	batch_size = 30
	iter_num = 50000

	# Define the optimizer
	opt = tf.train.GradientDescentOptimizer(learning_rate)

	# Define the goal
	goal = opt.minimize(loss)

	# Define the accuracy
	# The default threshold is 0.5, rounded off directly
	prediction = tf.round(tf.sigmoid(mod))
	# Bool into float32 type
	correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
	# Average
	accuracy = tf.reduce_mean(correct)
	# End of the definition of the model framework

	# Start training model
	# Define the variable that stores the result
	loss_trace = []
	train_acc = []
	test_acc = []

	# training model
	for epoch in range(iter_num):
	    # Generate random batch index
	    batch_index = np.random.choice(len(train_X), size=batch_size)
	    batch_train_X = train_X[batch_index]
	    batch_train_y = np.matrix(train_y[batch_index]).T
	    sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_y})
	    temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
	    # convert into a matrix, and the shape of the placeholder to correspond
	    temp_train_acc = sess.run(accuracy, feed_dict={data: train_X, target: np.matrix(train_y).T})
	    temp_test_acc = sess.run(accuracy, feed_dict={data: test_X, target: np.matrix(test_y).T})
	    # recode the result
	    loss_trace.append(temp_loss)
	    train_acc.append(temp_train_acc)
	    test_acc.append(temp_test_acc)
	    # output
	    if (epoch + 1) % 300 == 0:
	        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
	                                                                          temp_train_acc, temp_test_acc))
	# Visualization of the results
	# loss function
	pyplot.plot(loss_trace)
	pyplot.title('Cross Entropy Loss')
	pyplot.xlabel('epoch')
	pyplot.ylabel('loss')
	pyplot.show()

	# accuracy
	pyplot.plot(train_acc, 'b-', label='train accuracy')
	pyplot.plot(test_acc, 'k-', label='test accuracy')
	pyplot.xlabel('epoch')
	pyplot.ylabel('accuracy')
	pyplot.title('Train and Test Accuracy')
	pyplot.legend(loc='best')
	pyplot.show()

_main("C:/Users/taegu/Desktop/인턴자료/PeriodicClassification/samples_3", 1000)
