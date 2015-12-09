import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb
import pickle
import getopt
import sys
from sklearn.neural_network import MLPClassifier
from datetime import date

MAX_ITER=100
SIZES=(100)
today = date.today()
dataFile = ""
matrixFile = ""
ingredientsFile = ""
try:
      opts, args = getopt.getopt(sys.argv[1:],"d:m:i:",[])
except getopt.GetoptError:
      print 'USAGE: mlpclassifier.py -d <dataFile.json> -i <ingredientsFile> -m <matrixFile>'
      sys.exit(2)
for opt, arg in opts:
	if opt == "-d":
 		dataFile = arg
	elif opt == "-m":
        	matrixFile = arg
	elif opt == "-i":
        	ingredientsFile = arg

with open(dataFile) as json_data:
    data = js.load(json_data)

with open(matrixFile) as file:
	big_data_matrix = pickle.loads(file.read())

with open(ingredientsFile) as file:
	ingredients = pickle.loads(file.read())

classes = [item['cuisine'] for item in data]
unique_ingredients = set(item for sublist in ingredients for item in sublist)
unique_cuisines = set(classes)
examples = len(unique_cuisines)


train_set = big_data_matrix[0:35000,:]
test_set = big_data_matrix[35000:39773,:]
train_classes = classes[0:35000]
test_classes = classes[35000:39773]
            
clf2 = MLPClassifier (algorithm = 'sgd', alpha=1,verbose=True, hidden_layer_sizes=SIZES, max_iter=MAX_ITER, random_state=1, activation='logistic' );
f = clf2.fit(train_set, train_classes)

result = [(ref == res, ref, res) for (ref, res) in zip(test_classes, clf2.predict(test_set))]
accuracy_learn = sum (r[0] for r in result) / float ( len(result) )
storedNetwork = pickle.dumps(clf2)

output = "".join(["network_", str(today), "_maxIter_", str(MAX_ITER), "_sizes_", str(SIZES)])
with open(output, "w") as file:
	file.write(storedNetwork)


confMatrix = dict()
for i in unique_cuisines:
	confMatrix[i] = dict()
	for j in unique_cuisines:
		confMatrix[i][j] = 0
for _, desired, actual in result:
	confMatrix[desired][actual] += 1

for i in unique_cuisines:
	print i, ": ", confMatrix[i]
print('Accuracy on the learning set: ', accuracy_learn)
