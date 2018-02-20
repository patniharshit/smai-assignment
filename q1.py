import sys
import numpy as np
import pandas as pd
import ipdb

def split_data(data, folds):
	splits = []
	for i in range(folds):
		splits.append(data[i::folds])
	return splits

def vanilla_cross_validation(splitted_data):
	accuracies = []
	for i in range(len(splitted_data)):
		test_data = splitted_data[i]

		num_train_split = 0
		for num_train_split in range(len(splitted_data)):
			if num_train_split != i:
				training_data = splitted_data[num_train_split]

		j = num_train_split + 1
		while j < len(splitted_data):
			if j != i:
				np.vstack(training_data, splitted_data[j])
			j = j + 1

		train_y = training_data.iloc[:,training_data.shape[1]-1].values
		train_y[train_y == 2] = -1
		train_y[train_y == 4] = 1
		train_x = training_data.iloc[:,1:training_data.shape[1]-1].values

		test_y = test_data.iloc[:,test_data.shape[1]-1].values
		test_y[test_y == 2] = -1
		test_y[test_y == 4] = 1
		test_x = test_data.iloc[:,1:test_data.shape[1]-1].values

		train_x = np.array(train_x, dtype='int64')
		train_y = np.array(train_y, dtype='int64')
		test_x = np.array(test_x, dtype='int64')
		test_y = np.array(test_y, dtype='int64')

		classifier = vanilla_perceptron(train_x, train_y)

		classified = 0
		for tx, ty in zip(test_x, test_y):
			if np.sign(ty) == np.sign(np.dot(tx, classifier[0]) + classifier[1]):
				classified += 1

		print classified / float(test_y.shape[0])
		accuracies.append(classified/float(test_y.shape[0]))

	return sum(accuracies)/len(accuracies)

def voted_cross_validation(splitted_data):
	accuracies = []
	for i in range(len(splitted_data)):
		test_data = splitted_data[i]

		num_train_split = 0
		for num_train_split in range(len(splitted_data)):
			if num_train_split != i:
				training_data = splitted_data[num_train_split]

		j = num_train_split + 1
		while j < len(splitted_data):
			if j != i:
				np.vstack(training_data, splitted_data[j])
			j = j + 1

		train_y = training_data.iloc[:,training_data.shape[1]-1].values
		train_y[train_y == 2] = -1
		train_y[train_y == 4] = 1
		train_x = training_data.iloc[:,1:training_data.shape[1]-1].values

		test_y = test_data.iloc[:,test_data.shape[1]-1].values
		test_y[test_y == 2] = -1
		test_y[test_y == 4] = 1
		test_x = test_data.iloc[:,1:test_data.shape[1]-1].values

		train_x = np.array(train_x, dtype='int64')
		train_y = np.array(train_y, dtype='int64')
		test_x = np.array(test_x, dtype='int64')
		test_y = np.array(test_y, dtype='int64')

		classifier = voted_perceptron(train_x, train_y)

		classified = 0
		for tx, ty in zip(test_x, test_y):
			Y_cap = 0
			for k in range(len(classifier)):
				Y_cap += classifier[k][2]*np.sign(np.dot(classifier[k][0],tx)+classifier[k][1])
			Y_cap = np.sign(Y_cap)
			if np.sign(ty) == np.sign(Y_cap):
				classified += 1

		print classified / float(test_y.shape[0])
		accuracies.append(classified/float(test_y.shape[0]))

	return sum(accuracies)/len(accuracies)

def vanilla_perceptron(X, Y):
	num_samples = X.shape[0];
	W = np.zeros(X.shape[1])
	B = 0

	for i in range(epochs):
		for j in range(num_samples):
			if (Y[j] * (np.dot(W, X[j,:]) + B)) <= 0:
				W += Y[j] * X[j,:]
				B += Y[j]
	return W, B

def voted_perceptron(X, Y):
	# bad values in data
	num_samples = X.shape[0];
	W = np.zeros(X.shape[1])
	B = 0
	output = []

	C = 1
	for i in range(epochs):
		for j in range(num_samples):
			Y_cap = np.dot(W, X[j,:]) + B
			if (Y[j] * Y_cap) <= 0:
				output.append((W, B, C))
				W += Y[j] * X[j,:]
				B += Y[j]
				C = 1
			else:
				C = C + 1
	# number of outputs == num epochs?
	return output


epochs = 50
#dataset_name = "breast-cancer-wisconsin.data"
dataset_name = "var.data"
data = pd.read_csv(dataset_name, header=None)

splits = split_data(data, 5)
print vanilla_cross_validation(splits)
print "-----------------------"
print voted_cross_validation(splits)
