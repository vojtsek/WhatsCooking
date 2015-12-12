import pickle
import json as js
import scipy as scipy
import scipy.sparse
import numpy as np
import pdb

with open('./train.json') as json_data:
    data = js.load(json_data)

with open('./test.json') as json_data:
    test_data = js.load(json_data)

ingredients_test = [item['ingredients'] for item in test_data]
ingredients_all = [item['ingredients'] for item in data]
unique_ingredients = set(item for sublist in ingredients_all for item in sublist)
unique_ingredients_test = set(item for sublist in ingredients_test for item in sublist)
ingredientsOccurences = dict()

for i in unique_ingredients:
	ingredientsOccurences[i] = 0
for ings in ingredients_all:
	for i in ings:
		ingredientsOccurences[i] += 1
print ( len (ingredients_all) )
print ( len ( unique_ingredients ) )

cutted = []
for item in ingredients_all:
	il = []
	for i in item:
		if ingredientsOccurences[i] > 5:
			il.append(i)
	cutted.append(il)
ingredients_all = cutted

for i in unique_ingredients_test:
	ingredientsOccurences[i] = 0
for ings in ingredients_test:
	for i in ings:
		ingredientsOccurences[i] += 1
cutted = []
for item in ingredients_test:
	il = []
	for i in item:
		if ingredientsOccurences[i] > 5:
			il.append(i)
	cutted.append(il)
ingredients_test = cutted

unique_ingredients = set(item for sublist in ingredients_all for item in sublist)
unique_ingredients_test = set(item for sublist in ingredients_test for item in sublist)
print ( len (ingredients_all) )
print ( len ( unique_ingredients ) )

big_data_matrix = scipy.sparse.dok_matrix((len(ingredients_test), len(unique_ingredients)), dtype=np.dtype(bool))

# matrix for the test set
for d,dish in enumerate(ingredients_test):
    for i,ingredient in enumerate(unique_ingredients_test):
        if ingredient in dish:
            big_data_matrix[d,i] = True

with open("data_test_matrix_significant_ingredients.out", "w") as file:
	file.write(pickle.dumps(big_data_matrix))

# matrix for the training set
big_data_matrix = scipy.sparse.dok_matrix((len(ingredients_all), len(unique_ingredients)), dtype=np.dtype(bool))
for d,dish in enumerate(ingredients_all):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_data_matrix[d,i] = True

with open("data_training_matrix_significant_ingredients.out", "w") as file:
	file.write(pickle.dumps(big_data_matrix))

# save the ingredients sets
with open("test_ingredients_significant.out", "w") as file:
	file.write(pickle.dumps(ingredients_test))

with open("ingredients_significant.out", "w") as file:
	file.write(pickle.dumps(ingredients_all))
