import pickle
import json as js
import scipy as scipy
import scipy.sparse
import numpy as np
import pdb

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

for d,dish in enumerate(ingredients):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_data_matrix[d,i] = True

with open("data_matrix.out", "w") as file:
	file.write(pickle.dumps(big_data_matrix))
