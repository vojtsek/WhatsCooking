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

# maximum number of iterations allowed
MAX_ITER=200
# tuple of sizes of the hidden layers
SIZES=(20)
today = date.today()
dataFile = ""
matrixFile = ""
ingredientsFile = ""
networkFile = ""
try:
      opts, args = getopt.getopt(sys.argv[1:],"d:m:i:n:",[])
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
	elif opt == "-n":
        	networkFile = arg

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

clf2 = MLPClassifier (algorithm = 'adam', alpha=0.005, learning_rate="adaptive", verbose=True, hidden_layer_sizes=SIZES, max_iter=MAX_ITER, random_state=1, activation='tanh' );
f = clf2.fit(big_data_matrix, classes)

result = [(ref == res, ref, res) for (ref, res) in zip(classes, clf2.predict(big_data_matrix))]
accuracy_learn = sum (r[0] for r in result) / float ( len(result) )
storedNetwork = pickle.dumps(clf2)

output = "".join(["network_", str(today), "_maxIter_", str(MAX_ITER), "_sizes_", str(SIZES)])
with open(networkFile, "w") as file:
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
