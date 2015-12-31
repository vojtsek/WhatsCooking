import pickle
import json as js
import scipy as scipy
import scipy.sparse
import numpy as np
import pdb

with open('./dataset/train.json') as json_data:
    data = js.load(json_data)

with open('./dataset/test.json') as json_data:
    test_data = js.load(json_data)

ingredients_test = [item['ingredients'] for item in test_data]
ingredients_all = [item['ingredients'] for item in data]
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

with open("data_test_matrix.out", "w") as file:
	file.write(pickle.dumps(big_data_matrix))

# matrix for the training set
big_data_matrix = scipy.sparse.dok_matrix((len(ingredients_all), len(unique_ingredients)), dtype=np.dtype(bool))
for d,dish in enumerate(ingredients_all):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_data_matrix[d,i] = True

with open("data_training_matrix.out", "w") as file:
	file.write(pickle.dumps(big_data_matrix))

# save the ingredients sets
with open("test_ingredients.out", "w") as file:
	file.write(pickle.dumps(ingredients_test))

with open("ingredients.out", "w") as file:
	file.write(pickle.dumps(ingredients_all))

# cut the ingredients, leave the last word only
cutted_ingredients = []
cutted_test_ingredients = []
for item in ingredients_all:
	ingredientList = []
	for ingr in item:
		ingredientList.append(ingr.split(" ")[-1])
	cutted_ingredients.append(ingredientList)
for item in ingredients_test:
	ingredientList = []
	for ingr in item:
		ingredientList.append(ingr.split(" ")[-1])
	cutted_test_ingredients.append(ingredientList)
ingredients = cutted_ingredients
ingredients_test = cutted_test_ingredients
unique_ingredients = set(item for sublist in ingredients_all for item in sublist)
unique_ingredients_test = set(item for sublist in ingredients_test for item in sublist)

big_data_matrix = scipy.sparse.dok_matrix((len(ingredients_test), len(unique_ingredients)), dtype=np.dtype(bool))
for d,dish in enumerate(ingredients_test):
    for i,ingredient in enumerate(unique_ingredients_test):
        if ingredient in dish:
            big_data_matrix[d,i] = True

with open("data_test_matrix_last_words.out", "w") as file:
	file.write(pickle.dumps(big_data_matrix))

big_data_matrix = scipy.sparse.dok_matrix((len(ingredients_all), len(unique_ingredients)), dtype=np.dtype(bool))
for d,dish in enumerate(ingredients_all):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_data_matrix[d,i] = True

with open("data_matrix_last_words.out", "w") as file:
	file.write(pickle.dumps(big_data_matrix))

with open("test_ingredients_last_words.out", "w") as file:
	file.write(pickle.dumps(ingredients_test))

with open("ingredients_last_words.out", "w") as file:
	file.write(pickle.dumps(ingredients_all))
