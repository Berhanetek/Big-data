import sys
from pyspark import SparkConf, SparkContext


conf = SparkConf()
sc = SparkContext(conf = conf)

with open(sys.argv[1],"r") as f:
    contents = f.readlines()

k = int(sys.argv[2])

points = list(map(lambda x: list(map(float,x.split())), contents))


centroids = []
centroids.append(points[0])


def calculate_distance(pt1, pt2):
    squares = [(p-q) ** 2 for p, q in zip(pt1, pt2)]
    return sum(squares) ** 0.5

def make_clusters(pt):
    distance = []
    for ct in centroids:
        distance.append(calculate_distance(ct, pt))
    return (distance.index(min(distance)), pt)
    
def calculate_diameter(clstr):
    max_distance = 0
    for pt1 in clstr:
        for pt2 in clstr:
            d = calculate_distance(pt1, pt2)
            if d > max_distance:
                max_distance = d
    return(max_distance)

for i in range(k-1):
    d = []
    for pt in points:
        pt_to_ct = []
        for ct in centroids:
            pt_to_ct.append(calculate_distance(pt, ct))
        d.append(min(pt_to_ct))
    max_index = d.index(max(d))
    centroids.append(points[max_index])

points_rdd = sc.parallelize(points)
clusters = points_rdd.map(make_clusters).groupByKey().mapValues(list)
diameter = clusters.map(lambda x: calculate_diameter(x[1]))

average_diameter = sum(diameter.collect())/k
print(average_diameter)

sc.stop()
