import json
import numpy as np
from numpy.linalg import norm
  
def simility(obj):
	return obj["simility"]

f1 = open('inputdata.json')
data = json.load(f1)

f2 = open('data_2.json')
database = json.load(f2)

for row in database:
	cosine1 = np.dot(data["player_shape"], row["player_shape"])/(norm(data["player_shape"])*norm(row["player_shape"]))
	if ((data["e_shape"][0] == 0) or (row["e_shape"][0] == 0)):
		cosine2 = 0
	else:
		cosine2 = np.dot(data["e_shape"], row["e_shape"])/(norm(data["e_shape"])*norm(row["e_shape"]))
	if ((data["e_pos"][0] == 0) or (row["e_pos"][0] == 0)):
		cosine3 = 0
	else:	
		cosine3 = np.dot(data["e_pos"], row["e_pos"])/(norm(data["e_pos"])*norm(row["e_pos"]))

	cosine = (cosine1*6 + cosine2*3 + cosine1)/10

	row["simility"] = cosine

database.sort(key=simility, reverse=True)

voting = { 
	'bongda' : 0, 
	'bongro' : 0, 
	'chaybo' : 0, 
	'dapxe' : 0, 
	'tennis' : 0, 
	'bongchay' : 0, 
	'vothuat' : 0, 
	'luotsong' : 0, 
	'truottuyet' : 0,
	'caulong' : 0,
}

for i in range(10):
	voting[database[i]["image_class"]] += 1

class_name = ""
max_voting = 0
print(voting)
for value in voting:
	if voting[value] > max_voting:
		max_voting = voting[value]
		class_name = value

print(class_name)