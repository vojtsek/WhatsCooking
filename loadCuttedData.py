import pickle
import json as js
import scipy as scipy
import scipy.sparse
import numpy as np
from sklearn import decomposition
import pdb

def cutIngredients(unique_ingredients_test, unique_ingredients, ingredients_all, ingredients_test):
	def inRange(number):
		return (number > 5) and (number < 1000)

	ingredientsOccurences = dict()
	for i in unique_ingredients:
		ingredientsOccurences[i] = 0
	for ings in ingredients_all:
		for i in ings:
			ingredientsOccurences[i] += 1
	print ( len (ingredients_all) )
	print ( len ( unique_ingredients ) )

	# print sorted(ingredientsOccurences.values())
	cutted = []
	for item in ingredients_all:
		il = []
		for i in item:
			if inRange(ingredientsOccurences[i]):
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
			if inRange(ingredientsOccurences[i]):
				il.append(i)
		cutted.append(il)
	ingredients_test = cutted
	return (ingredients_test, ingredients_all)

def processPCA(unique_ingredients_test, unique_ingredients, ingredients_all, ingredients_test):
	pca = decomposition.IncrementalPCA(n_components=4000)
#	big_data_matrix = scipy.sparse.dok_matrix((len(ingredients_test), len(unique_ingredients)), dtype=np.dtype(bool))

	# matrix for the test set
#	for d,dish in enumerate(ingredients_test):
#	    for i,ingredient in enumerate(unique_ingredients):
#		if ingredient in dish:
#		    big_data_matrix[d,i] = True

#	with open("test_matrix_PCA", "w") as file:
#		file.write(pickle.dumps(big_data_matrix))

#	with open("test_ingredients_PCA", "w") as file:
#		file.write(pickle.dumps(ingredients_test))

	# matrix for the training set
	big_data_matrix = scipy.sparse.dok_matrix((len(ingredients_all), len(unique_ingredients)), dtype=np.dtype(bool))
	for d,dish in enumerate(ingredients_all):
	    for i,ingredient in enumerate(unique_ingredients):
		if ingredient in dish:
		    big_data_matrix[d,i] = True
	big_data_matrix = big_data_matrix.toarray()
	big_data_matrix = pca.fit_transform(big_data_matrix)
	big_data_matrix = scipy.sparse.dok_matrix(big_data_matrix, dtype=np.dtype(bool))
	with open("matrix_PCA", "w") as file:
		file.write(pickle.dumps(big_data_matrix))
	with open("ingredients_PCA", "w") as file:
		file.write(pickle.dumps(ingredients_test))


def writeResults(unique_ingredients_test, unique_ingredients, ingredients_all, ingredients_test, matrix1, matrix2, ingredients1, ingredients2):
	big_data_matrix = scipy.sparse.dok_matrix((len(ingredients_test), len(unique_ingredients)), dtype=np.dtype(bool))

	# matrix for the test set
	for d,dish in enumerate(ingredients_test):
	    for i,ingredient in enumerate(unique_ingredients):
		if ingredient in dish:
		    big_data_matrix[d,i] = True

	with open(matrix1, "w") as file:
		file.write(pickle.dumps(big_data_matrix))

	# matrix for the training set
	big_data_matrix = scipy.sparse.dok_matrix((len(ingredients_all), len(unique_ingredients)), dtype=np.dtype(bool))
	for d,dish in enumerate(ingredients_all):
	    for i,ingredient in enumerate(unique_ingredients):
		if ingredient in dish:
		    big_data_matrix[d,i] = True

	with open(matrix2, "w") as file:
		file.write(pickle.dumps(big_data_matrix))

	# save the ingredients sets
	with open(ingredients1, "w") as file:
		file.write(pickle.dumps(ingredients_test))

	with open(ingredients2, "w") as file:
		file.write(pickle.dumps(ingredients_all))

with open('./train.json') as json_data:
    data = js.load(json_data)

with open('./test.json') as json_data:
    test_data = js.load(json_data)

ingredients_test = [item['ingredients'] for item in test_data]
ingredients_all = [item['ingredients'] for item in data]
unique_ingredients = set(item for sublist in ingredients_all for item in sublist)
unique_ingredients_test = set(item for sublist in ingredients_test for item in sublist)


(ingredients_test, ingredients_all) =  cutIngredients(unique_ingredients_test, unique_ingredients, ingredients_all, ingredients_test)
unique_ingredients = set(item for sublist in ingredients_all for item in sublist)
unique_ingredients_test = set(item for sublist in ingredients_test for item in sublist)
print ( len (ingredients_all) )
print ( len ( unique_ingredients ) )
writeResults(unique_ingredients_test, unique_ingredients, ingredients_all, ingredients_test, "test_matrix_significant.out", "matrix_significant.out", "test_ingredients_significant.out", "ingredients_significant.out")

#processPCA(unique_ingredients_test, unique_ingredients, ingredients_all, ingredients_test)
