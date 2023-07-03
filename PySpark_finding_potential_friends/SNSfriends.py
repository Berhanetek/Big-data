# Potential friends on a social media network using spark.

import re
import sys
import math
import os
from pyspark import SparkConf, SparkContext


conf = SparkConf()
sc = SparkContext(conf=conf)
line = sc.textFile(sys.argv[1]) # each element of this rdd has one line of the text dataset we used.

atleast_two_friends = line.filter(lambda l: len(l) > 4 )

users = atleast_two_friends.map(lambda l: re.split(r'\t+', l))


def make_pairs(i):
    #Input format is of the form: i = ['5', '4,11,7'] where the first entry is the user and the second entry is its friends.
    friends = i[1]
    friends = friends.split(",")
    friends = [i.strip() for i in friends]
    for i in friends:
        if(len(i) <= 0): # just in case there's an empty string for some reason.
            friends.remove(i)
    list_of_pairs = []
    for i in range(len(friends)):
        for j in range(len(friends)):
            if i<j:
                if (int(friends[i]) < int(friends[j])):
                    list_of_pairs.append(( friends[i],friends[j]))
                else:
                    list_of_pairs.append(( friends[j],friends[i]))
    return list_of_pairs




def make_list_of_friends(i):
    user = i[0]
    friends = i[1]
    list_of_friends = []
    friends = friends.split(",")
    friends = [i.strip() for i in friends]
    for i in friends:
        if(len(i) <= 0): # just in case there's an empty string for some reason.
            friends.remove(i)
    
    for j in range(len(friends)):
        if( int(friends[j]) > int(user)):
            list_of_friends.append((user, friends[j]))
        else:
            list_of_friends.append((friends[j], user))
    return list_of_friends



candidate_pairs = users.flatMap(make_pairs)

all_paired_friends = users.flatMap(make_list_of_friends)



not_friends = candidate_pairs.subtract(all_paired_friends)
not_friends_key_value = not_friends.map(lambda i: (i, 1))

total_num_of_mutual_friends = not_friends_key_value.reduceByKey(lambda a,b: a+b)


sorted_final_list = total_num_of_mutual_friends.takeOrdered(10, key = lambda l: (-l[1], l[0][0], l[0][1]))


for entry in sorted_final_list:
    print('{0}\t{1}\t{2}'.format(entry[0][0], entry[0][1], entry[1]))


sc.stop()
