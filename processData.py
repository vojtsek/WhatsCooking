import json
from nltk.stem import PorterStemmer
from sklearn.neural_network import MLP

def useLastWord(ingredient):
    return ingredient.split()[:-1]

with open("/Users/spinoco/Dropbox/Link to skola/neuronky/cooking/train.json") as dataJSON:
    data = json.load(dataJSON)
allIngredients = dict()
allCuisines = set()
count = 0
port = PorterStemmer()
for recipe in data:
        for i in recipe["ingredients"]:
            ingr = " ".join(part for part in i.split())
            try:
                allIngredients[ingr]["frequency"] += 1
            except Exception:
                allIngredients[ingr] = dict()
                allIngredients[ingr]["frequency"] = 1
                allIngredients[ingr]["idx"] = count
                count += 1
        allCuisines.add(recipe["cuisine"])

cuisineIndexes = {}
count = 0
for i in allCuisines:
        cuisineIndexes[i] = count
        count += 1
count = 0
sum = len(allIngredients)
# sortedIngredients = sorted(allIngredients.items(), key = lambda a: a[1]["frequency"])
# cuttedIngredients = [(" ".join(s for s in i[0].split()), i[1]) for i in sortedIngredients if i[1]["frequency"] > 5]
print len(allIngredients)
# print len(cuttedIngredients)

# with open('out2.csv', 'w') as file:
#         for recipe in data:
#                 length = 0
#                 for ingr in recipe['ingredients']:
#                         file.write(str(ingredientsIndexes[ingr]))
#                         file.write(',')
#                         length += 1
#                         if (length == 20):
#                                 break
#                 if (length < 20):
#                         for i in range(0, 20 - length):
#                                 file.write(str(0))
#                                 file.write(',')
#                 file.write(str(cuisineIndexes[recipe['cuisine']]))
#                 file.write('\n')
#with open('out.csv','w') as file:
#       for recipe in data:
#               newLine = [0 for i in range(0, sum)]
#               for ingr in recipe['ingredients']:
#                       newLine[ingredientsIndexes[ingr]] = 1
#               newLine.append(cuisineIndexes[recipe['cuisine']])
#               for c in newLine:
#                       file.write(str(c))
#                       file.write(',')
#               file.write('\n')
