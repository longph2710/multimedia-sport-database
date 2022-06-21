import cv2
import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import math
import json

class Metadata:
	image_dir = "image_dir"
	image_class = "football"
	e_pos = []
	e_shape = []
	player_shape = []

def pickup1(a, n_points = 10):
	(height, width) = a.shape
	min_y = width
	max_y = 0
	for i in range(height):
		for j in range(width):
			if a[i][j] > 0:
				if j < min_y:
					min_y = j
				if j > max_y:
					max_y = j
	step = math.floor((max_y-min_y)/n_points*2)
	res = [0]*n_points*2
	range_y = [0]*math.floor(n_points/2)
	for i in range(math.floor(n_points/2)):
		range_y[i] = min_y+i*step
	idx = 0
	for j in range_y:
		tmp = [0,0,0,0]
		x0 = 0
		y0 = 1
		for i in range(height):
			if j >= width:
				break
			if a[i][j] > 0:
				tmp[x0] = i
				tmp[y0] = j
				x0 = 2
				y0 = 3
		res[idx] = tmp[0]
		res[idx+1] = tmp[1]
		res[idx+2] = tmp[2]
		res[idx+3] = tmp[3]
		idx += 4
	return res

def pickup2(a, n_points = 50):
	(height, width) = a.shape
	min_x = height
	max_x = 0
	for i in range(height):
		for j in range(width):
			if a[i][j] > 0:
				if i < min_x:
					min_x = i
				if i > max_x:
					max_x = i
	step = math.floor((max_x-min_x)/n_points*2)
	res = [0]*n_points*2
	range_x = [0]*math.floor(n_points/2)
	for i in range(math.floor(n_points/2)):
		range_x[i] = min_x+i*step
	idx = 0
	for i in range_x:
		tmp = [0,0,0,0]
		x0 = 0
		y0 = 1
		for j in range(width):
			if a[i][j] > 0:
				tmp[x0] = i
				tmp[y0] = j
				x0 = 2
				y0 = 3
		res[idx] = tmp[0]
		res[idx+1] = tmp[1]
		res[idx+2] = tmp[2]
		res[idx+3] = tmp[3]
		idx += 4
	return res

# const
max_y_distance = 150

# doc anh
image = cv2.imread('data\\bongda\\bongda7.png')

# tach bien anh
edges = cv2.Canny(image,100,200)

# kmeans
n_clust = 5
a = np.asarray([[0,0]])
for i in range(0, edges.shape[0]):
	for j in range(0, edges.shape[1]):
		if edges[i][j] == 255:
			a = np.append(a, [[i, j]], axis=0)

a[0] = a[1]
kmeans = cluster.KMeans(n_clusters=n_clust).fit(a)

# xac dinh vong tron
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.3, 100)

# xac dinh dung cu
sumx = 0
sumy = 0
max_sse = max_y_distance
equid_cluster = -1
for centroid in kmeans.cluster_centers_:
	sumx += centroid[0]
	sumy += centroid[1]
avgx = (sumx)/(n_clust)
avgy = (sumy)/(n_clust)
for idx, centroid in enumerate(kmeans.cluster_centers_):
	# sse = math.sqrt((avgx - centroid[0])*(avgx - centroid[0]) + (avgy - centroid[1])*(avgy - centroid[1]))
	sse = avgy - centroid[1]
	if (sse > max_sse):
		max_sse = sse 
		equid_cluster = idx

mask_centroid_x = 0
mask_centroid_y = 0
isclust = 0;
mask1 = np.zeros(image.shape, dtype=np.uint8)
mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
if equid_cluster > -1:
	mask_centroid_x += kmeans.cluster_centers_[equid_cluster][0]
	mask_centroid_y += kmeans.cluster_centers_[equid_cluster][1]
	isclust += 1
	for idx, label in enumerate(kmeans.labels_):
		if label == equid_cluster:
			x = a[idx][0]
			y = a[idx][1]
			mask1[x][y] = 255

mask2 = np.zeros(image.shape, dtype=np.uint8)
mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

circles_size = 0
if circles is not None:
	circles = np.round(circles[0, :]).astype("int")
	for (x, y, r) in circles:
		cv2.circle(mask2, (x, y), r, 255, -1)
		mask_centroid_x += x
		mask_centroid_y += y
		circles_size += 1
mask2 = cv2.bitwise_and(edges, edges, mask = mask2)
mask = mask1 + mask2
# cv2.imshow("mask", cv2.bitwise_not(edges, edges, mask = mask))

equidment = mask
player = cv2.bitwise_not(edges, edges, mask = mask)

vector_equidment = pickup1(equidment)
vector_player = pickup2(player)

data = Metadata()
# data.image_class = class_name
data.e_shape = vector_equidment
data.player_shape = vector_player
if(circles_size + isclust == 0):
	data.e_pos = [0,0]
else:
	data.e_pos = [math.floor(mask_centroid_x/(circles_size + isclust)), math.floor(mask_centroid_y/(circles_size + isclust))]

jsonStr = json.dumps(data.__dict__)
print(jsonStr)
f = open("inputdata.json", "w")
f.write(jsonStr)
f.close()

# cv2.imshow("mask", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



