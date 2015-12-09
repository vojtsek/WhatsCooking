import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb
import pickle
from sklearn.neural_network import MLPClassifier

with open('./train.json') as json_data:
    data = js.load(json_data)
    json_data.close()

classes = [item['cuisine'] for item in data]
ingredients = [item['ingredients'] for item in data]
allIngredients = []
for item in ingredients:
	ingredientList = []
	for ingr in item:
		ingredientList.append(ingr.split(" ")[-1])
	allIngredients.append(ingredientList)
ingredients = allIngredients
unique_ingredients = set(item for sublist in ingredients for item in sublist)
unique_cuisines = set(classes)

# print len(data)
# print (classes)
examples = len(unique_cuisines)
print ( len (ingredients) )
print ( len ( unique_ingredients ) )
print ( len ( unique_cuisines ) )

big_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)), dtype=np.dtype(bool))

step=0
for d,dish in enumerate(ingredients):
    step=step+1
#    if (step > 2000):
#	break
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_data_matrix[d,i] = True

test_set = big_data_matrix[35000:39773,:]
test_classes = classes[35000:39773]
            
with open("network.out") as net:
	clf = pickle.loads("".join(line for line in net.readlines()))
	result = [(ref == res, ref, res) for (ref, res) in zip(test_classes, clf.predict(test_set))] 
	accuracy_learn = sum (r[0] for r in result) / float ( len(result) )
	print accuracy_learn
