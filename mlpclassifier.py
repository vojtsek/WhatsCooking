import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb
import pickle
import getopt
import sys
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from time import gmtime, strftime

# maximum number of iterations allowed
MAX_ITER=200
# tuple of sizes of the hidden layers
SIZES=(50)
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

clf2 = MLPClassifier (algorithm = 'adam', alpha=0.1, learning_rate="adaptive", verbose=True, hidden_layer_sizes=SIZES, max_iter=MAX_ITER, random_state=1, activation='tanh', early_stopping=False, validation_fraction=0.10)
f = clf2.fit(big_data_matrix, classes)

fig = plt.figure()
plt.plot(range(0, clf2.n_iter_), clf2.loss_curve_, c="red", linestyle="-")
try:
	plt.plot(range(0, clf2.n_iter_), clf2.validation_scores_, c="green", linestyle="-")
except Exception:
	pass
plt.axis([0, clf2.n_iter_, 0, 1.8])
fig.savefig(strftime("plot_%d-%b-%Y_%H-%M.png", gmtime()))
result = [(ref == res, ref, res) for (ref, res) in zip(classes, clf2.predict(big_data_matrix))]
accuracyLearn = sum (r[0] for r in result) / float ( len(result) )
storedNetwork = pickle.dumps(clf2)

with open(networkFile, "w") as file:
	file.write(storedNetwork)

confMatrix = dict()
cells = []
labels = [c for c in unique_cuisines]
for i in unique_cuisines:
	confMatrix[i] = dict()
	for j in unique_cuisines:
		confMatrix[i][j] = 0
for _, desired, actual in result:
	confMatrix[desired][actual] += 1

for idx, i in enumerate(unique_cuisines):
	rowSum = sum(confMatrix[i].values())
	rowPercentage = 100 * confMatrix[i].values()[idx] / rowSum
	cells.append([str(rowPercentage) + "%"] + [str(v) for v in confMatrix[i].values()])
plt.clf()
fig = plt.figure(num=None, figsize=(16, 9), dpi=100)
red = "#ff6666"
green = "#66EE77"
blue = "#6677FF"
gray = "#CCCCCC"
t = plt.table(cellText=cells, rowLabels=labels, rowColours=[gray for _ in range(len(labels) + 1)], colLabels=["%"] + [l[0:3] for l in labels], loc="center", cellLoc="center")
cc = t.get_celld()
for i, _ in enumerate(unique_cuisines):
	for j, _ in enumerate(unique_cuisines):
		c = cc[(i+1, j+1)]
		if (i == j):
			c.set_color(green)
		else:
			c.set_color(red)
for i in range(0, len(labels) + 1):
	cc[(0, i)].set_color(gray)
	if (i > 0):
		cc[(i, 0)].set_color(blue)
t.auto_set_font_size(False)
t.set_fontsize(12)
t.scale(1.1,1.2)
plt.axis('off')
fig.savefig(strftime("confusion_%d-%b-%Y_%H-%M.png", gmtime()))
print('Accuracy on the learning set: ', accuracyLearn)
