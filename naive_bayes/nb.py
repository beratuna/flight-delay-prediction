import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

train_ratio = float(99/100)

all_data = np.loadtxt(open("../dataset/final.csv", "rb"), dtype="float", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 5, 6, 7, 8, 10))

print("Data loaded!")

np.random.shuffle(all_data)

all_features = np.zeros(shape= (int(all_data.shape[0]), all_data.shape[1]-1),  dtype="float")
all_labels = np.zeros(shape= (int(all_data.shape[0])),  dtype="float")

for i in range(0, all_data.shape[0]):
	for j in range(0, all_features.shape[1]):
		all_features[i, j] = all_data[i, j+1]
	all_labels[i] = all_data[i, 0]

print("Dataset shuffled!")

train_features = np.zeros(shape= (int(all_features.shape[0]*train_ratio), all_features.shape[1]),  dtype="float")
train_labels = np.zeros(shape= int(all_labels.shape[0]*train_ratio), dtype="float")

test_features = np.zeros(shape= (int(all_features.shape[0]*(1-train_ratio)), all_features.shape[1]),  dtype="float")
test_labels = np.zeros(shape= int(all_labels.shape[0]*(1-train_ratio)), dtype="float")

for i in range(0, train_labels.shape[0]):
	train_features[i] = all_features[i]
	if int(all_labels[i]) > 15:
		train_labels[i] = 1
	else:
		train_labels[i] = 0
	# train_labels[i] = all_labels[i]
	

for i in range(0, test_labels.shape[0]):
	test_features[i] = all_features[i + train_labels.shape[0]]
	if int(all_labels[i + train_labels.shape[0]]) > 15:
		test_labels[i] = 1
	else:
		test_labels[i] = 0
	# test_labels[i] = all_labels[i + train_labels.shape[0]]
	

print("Datasets seperated!")

model = LogisticRegression()
print("Training started!")
model.fit(train_features, train_labels)

print("Training complete!")

print("Prediction started!")

predicted = model.predict(test_features)

print("Prediction complete!")

correct = 0
wrong = 0

for i in range(0, predicted.shape[0]):
	if predicted[i] == test_labels[i]:
		correct = correct + 1
	else:
		wrong = wrong + 1

print("accuracy: " + str(float(correct) / float(test_labels.shape[0])))

print(str(wrong) + " wrong predictions out of " + str(test_labels.shape[0]))


for i in range(0, predicted.shape[0]):
	if predicted[i] == 1:
		print("" + str(predicted[i]) + "----------" + str(test_labels[i]))