import pickle
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path,batch_id):
	"""
	Args:
		folder_path: the directory contains data files
		batch_id: training batch id (1,2,3,4,5)
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
	"""

	###load batch using pickle###
	filename = folder_path+'data_batch_'+ str(batch_id)
	data = unpickle(filename)

	###fetch features using the key ['data']###
	features =  data[b'data']
	###fetch labels using the key ['labels']###
	labels = np.array(data[b'labels'])
	return features,labels

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
	"""
	Args:
		folder_path: the directory contains data files
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
	"""

	###load batch using pickle###
	filename=folder_path+"test_batch";
	data = unpickle(filename)

	###fetch features using the key ['data']###
	features = data[b'data']
	###fetch labels using the key ['labels']###
	labels = np.array(data[b'labels'])
	return features,labels

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
	airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():
	raw = unpickle(filename="batches.meta")[b'label_names']
	return [label.decode('utf-8') for label in raw];

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
	"""
	Args:
		features: a numpy array with shape (10000, 3072)
	Return:
		features: a numpy array with shape (10000,32,32,3)
	"""
	refeatures_reshaped = features.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
	return refeatures_reshaped

#Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path,batch_id,data_id):
	"""
	Args:
		folder_path: directory that contains data files
		batch_id: the specific number of batch you want to explore.
		data_id: the specific number of data example you want to visualize
	Return:
		None

	Descrption: 
		1)You can print out the number of images for every class. 
		2)Visualize the image
		3)Print out the minimum and maximum values of pixel 
	"""
	filename = folder_path + 'data_batch_' + str(batch_id)
	data = unpickle(filename)
	X = data[b'data']
	Y = data[b'labels']
	X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
	Y = np.array(Y)
	plt.imshow(X[data_id:data_id + 1][0])
	plt.show()

#Step 6: define a function that does min-max normalization on input
def normalize(x):
	"""
	Args:
		x: features, a numpy array
	Return:
		x: normalized features
	"""
	x_normalized = np.zeros(x.shape)
	for i, x in enumerate(x):
		x_normalized[i,:] = (x-min(x))/float(max(x)-min(x))
	return x_normalized

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
	"""
	Args:
		labels_list: a list of labels
	Return:
		a numpy array that has shape (len(x), # of classes)
	"""
	num_classes = np.max(x) + 1
	return np.eye(num_classes, dtype=float)[x]

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename):
	"""
	Args:
		features: numpy array
		labels: a list of labels
		filename: the file you want to save the preprocessed data
	"""
	features = features_reshape(normalize(features))
	labels = one_hot_encoding(labels)
	with open('cifar-preprocessed/'+filename, 'wb') as handle:
		pickle.dump([features,labels], handle, protocol=pickle.HIGHEST_PROTOCOL)
#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
	"""
	Args:
		folder_path: the directory contains your data files
	"""
	batch_ids = np.arange(1,6)
	features_valid_all,labels_valid_all = np.empty((0, 3072)),[]
	for i, batch_id in enumerate(batch_ids):
		features, labels = load_training_batch(folder_path,batch_id)
		batch_size = features.shape[0]
		# training_samples = np.arange(batch_size)
		# shuffle(training_samples)
		train_size = int(batch_size*0.9)
		features_train = features[:train_size,:]
		labels_train  = labels[:train_size]
		features_valid = features[train_size:,:]
		labels_valid = labels[train_size:]
		preprocess_and_save(features_train, labels_train, 'batch_train_' + str(i+1))
		features_valid_all = np.concatenate([features_valid_all, features_valid])
		labels_valid_all.extend(labels_valid)
	labels_valid_all = np.array(labels_valid_all)
	preprocess_and_save(features_valid_all, labels_valid_all, 'batch_valid')
	features_test, labels_test = load_testing_batch(folder_path)
	preprocess_and_save(features_test, labels_test, 'batch_test')
#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
	"""
	Args:
		features: features for one batch
		labels: labels for one batch
		mini_batch_size: the mini-batch size you want to use.
	Hint: Use "yield" to generate mini-batch features and labels
	"""
	for start_idx in range(0, features.shape[0] - mini_batch_size + 1, mini_batch_size):
		indices = np.arange(features.shape[0])
		np.random.shuffle(indices)
		excerpt = indices[start_idx:start_idx + mini_batch_size]
		yield features[excerpt], labels[excerpt]

#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id,mini_batch_size):
	"""
	Args:
		batch_id: the specific training batch you want to load
		mini_batch_size: the number of examples you want to process for one update
	Return:
		mini_batch(features,labels, mini_batch_size)
	"""
	# file_name = ''
	# features, labels = pass
	# return mini_batch(features,labels,mini_batch_size)
	features, labels = pickle.load(open('cifar-preprocessed/batch_train_'+str(batch_id), 'rb'))
	return mini_batch(features, labels, mini_batch_size)


#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():
	# file_name =
	# features,labels =
	# return features,labels
	valid_features, valid_labels = pickle.load(open('cifar-preprocessed/batch_valid', 'rb'))
	return valid_features, valid_labels

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch():
	# file_name =
	# features,label =
	# return mini_batch(features,labels,test_mini_batch_size)
	test_features, test_labels = pickle.load(open('cifar-preprocessed/batch_test', 'rb'))
	return test_features, test_labels