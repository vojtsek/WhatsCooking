import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb
import pickle
import sys
import getopt
from sklearn.neural_network import MLPClassifier

dataFile = ""
matrixFile = ""
ingredientsFile = ""
networkFile = ""
try:
      opts, args = getopt.getopt(sys.argv[1:],"d:m:n:i:",[])
except getopt.GetoptError:
      print 'USAGE: recall.py -d <dataFile.json> -n <networkFile> -i <ingredientsFile> -m <matrixFile>'
      sys.exit(2)
for opt, arg in opts:
	if opt == "-d":
 		dataFile = arg
	elif opt == "-n":
        	networkFile = arg
	elif opt == "-m":
        	matrixFile = arg
	elif opt == "-i":
        	ingredientsFile = arg

with open(dataFile) as json_data:
    data = js.load(json_data)

with open("dataset/train.json") as f:
	data2=js.load(f)

with open(matrixFile) as file:
	big_data_matrix = pickle.loads(file.read())

with open(ingredientsFile) as file:
	ingredients = pickle.loads(file.read())

classes = [item['cuisine'] for item in data2]
unique_ingredients = set(item for sublist in ingredients for item in sublist)
unique_cuisines = set(classes)
examples = len(unique_cuisines)

test_set = big_data_matrix
test_classes = classes
 
print ( len (ingredients) )
print ( len ( unique_ingredients ) )
print ( len ( unique_cuisines ) )

with open(networkFile) as net:
	clf = pickle.loads("".join(line for line in net.readlines()))
	print test_set.shape
	for cls, rec in zip(clf.predict(test_set), data):
		print rec["id"], ",", cls
	#accuracy_learn = sum (r[0] for r in result) / float ( len(result) )
	#print accuracy_learn
	confMatrix = dict()
	for i in unique_cuisines:
		confMatrix[i] = dict()
		for j in unique_cuisines:
			confMatrix[i][j] = 0
	#for _, desired, actual in result:
	#	confMatrix[desired][actual] += 1

#for i in unique_cuisines:
	#print i, ": ", confMatrix[i]
