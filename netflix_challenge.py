import numpy as np
import csv
import sys
import math

with open(sys.argv[1],"r") as f:
    ratings = f.readlines()

with open(sys.argv[2],"r") as f:
    ratings_test = f.readlines()

def build_test_matrix(test_set):
    test_matrix_U = dict()
    test_matrix_I = dict()
    
    for entry in test_set: 
        user, movie_id, _, time_stamp =  entry.split(',') 
        user, movie_id, time_stamp = int(user), int(movie_id), float(time_stamp) 

        if user not in test_matrix_U:
            test_matrix_U[user] = []
        test_matrix_U[user].append(movie_id)

        if movie_id not in test_matrix_I:
            test_matrix_I[movie_id] = []
        test_matrix_I[movie_id].append(user)

    return test_matrix_U, test_matrix_I 


test_utility_matrix_u, test_utility_matrix_i = build_test_matrix(ratings_test)

def build_utility_matrix(ratings):
    utility_matrix = dict()
    for rating in ratings:
        user, movie_id, rating, time_stamp =  rating.split(',')
        user, movie_id, rating = int(user), int(movie_id), float(rating)
        if user not in utility_matrix:
            utility_matrix[user] = {}
        utility_matrix[user][movie_id] = rating
    return utility_matrix


utility_matrix = build_utility_matrix(ratings) #utility matrix is a dictionary where the keys are the users and the value is a dictionary of (movie -> rating) pairs.
utility_matrix_II = build_utility_matrix(ratings)

util_matrix = build_utility_matrix(ratings)

# ALTERS THE INPUT ARGUMENTS.
def normalize_utility_matrix(u_matrix): #Input utility matrix is a dictionary where the keys are the users and the value is a dictionary of movie -> rating pairs.
    mean_ratings = dict() #mean_ratings = user -> average rating they gave for the movies they rated.
    for user in u_matrix:
        user_ratings = u_matrix[user]
        if len(user_ratings) == 0: #if the user didn't give any ratings.
            continue
        mean_rating_of_user = sum(user_ratings.values())/len(user_ratings)
        mean_ratings[user] = mean_rating_of_user
        
    for user in u_matrix:
        for movie_id in u_matrix[user]:
            u_matrix[user][movie_id] = u_matrix[user][movie_id] - mean_ratings[user]
    return u_matrix, mean_ratings #Returns a utility matrix that is mean normalized.

mean_normalized_utility_matrix, mean_ratings_per_user = normalize_utility_matrix(utility_matrix)

mean_normalized_utility_matrix_II, mean_ratings_per_user_II = normalize_utility_matrix(utility_matrix_II)

# DOESN'T ALTER THE INPUT ARGUMENTS.
def calculate_norm(vector):
    #big_epsilon = 1000
    norm = math.sqrt(sum([i**2 for i in vector.values()]))
    return norm


# DOESN'T ALTER THE INPUT ARGUMENTS.
def cosine_similarity(vec_i, vec_j):
    norm_vec_i = calculate_norm(vec_i)
    norm_vec_j = calculate_norm(vec_j)
    if (norm_vec_i == 0) or (norm_vec_j == 0):
        return 0
    dot_product = 0
    for entry in vec_i:
        if entry in vec_j:
            dot_product += vec_i[entry] * vec_j[entry]
   
    cos_similarity = dot_product / (norm_vec_i * norm_vec_j)
    return cos_similarity


# DOESN'T ALTER THE INPUT ARGUMENTS.
# Returns a list of n most similar users to user_u
def find_most_similar_users(mean_normalized_utility_matrix, user_u, n):
    distances = dict()
    for user in mean_normalized_utility_matrix:
        if user == user_u:
            continue
        else:
            cosine_distance = cosine_similarity(mean_normalized_utility_matrix[user], mean_normalized_utility_matrix[user_u])
            distances[user] = cosine_distance
    n_most_similar = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:n]
    #n_most_similar_users = [i[0] for i in n_most_similar]
    return(n_most_similar)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


#user-user-cf score
user_user_cf = dict()
for user in test_utility_matrix_u:
    n_most_similar_users = find_most_similar_users(mean_normalized_utility_matrix, user, 330) #330 obtained emperically as explained in hw2.pdf
    #print(n_most_similar_users)
    average_ratings = dict()    
    for movie_id in test_utility_matrix_u[user]:
        number_of_people_who_watched_it = 0
        accumulated_ratings = []
        distance_scaler = []

        #for user_k in most_similar_users:
        for user_k in n_most_similar_users:
            #print(user_k)    
            if movie_id in util_matrix[user_k[0]]: #if that particular similar user has seen it
            #if movie_id in most_similar_users[user_k]:
                number_of_people_who_watched_it += 1
                
                accumulated_ratings.append(util_matrix[user_k[0]][movie_id])
                #print(user_k[1])
                distance_scaler.append(user_k[1]) #append the distance between user and user_k.

    
        if(number_of_people_who_watched_it == 0):
            average_ratings[movie_id] = mean_ratings_per_user[user]
        else:
            average_ratings[movie_id] = np.matmul(accumulated_ratings, (softmax(distance_scaler)))
    user_user_cf[user] = average_ratings
    

U_ratings = list()
for user in user_user_cf:
    for movie in user_user_cf[user]:
        ratings = [user, movie, user_user_cf[user][movie]]
        U_ratings.append(ratings)


#the problem is here I get it now. User a newer version of mean_normalized_utility_matrix cause this one is already altered.
def user_dict_to_movies_dict(mean_normalized_utility_matrix):
    movies_dict = {}
    for user in mean_normalized_utility_matrix:
        for movie_id in mean_normalized_utility_matrix[user]:
            if movie_id not in movies_dict:
                movies_dict[movie_id] = {}
            movies_dict[movie_id][user] = mean_normalized_utility_matrix[user][movie_id]
    return movies_dict

movies_dict = user_dict_to_movies_dict(mean_normalized_utility_matrix_II)


# DOESN'T ALTER THE INPUT ARGUMENTS.
def find_most_similar_items(mean_norm_utility_matrix_II, item_i, n):
    #print(item_i)
    distances = dict()
    #norm_user_600 = calculate_norm(mean_normalized_utility_matrix[user_u])
    for item in mean_norm_utility_matrix_II:
        #print(item)
        if item == item_i:
            continue
        else:
            cosine_distance = cosine_similarity(mean_norm_utility_matrix_II[item], mean_norm_utility_matrix_II[item_i])
            distances[item] = cosine_distance

    n_most_similar = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:n]
    #n_most_similar_items = [i[0] for i in n_most_similar]
    return(n_most_similar)

# ITEM - ITEM COLLABORATIVE FILTERING
item_item_cf = dict()
for item in test_utility_matrix_i:
    if item not in movies_dict: # Only 1 person reviewed it and it's in the test set. So there's no way to find similar movies to it in the training set.
        missing_item_ratings = dict()
        for usr in test_utility_matrix_i[item]:
            missing_item_ratings[usr] = mean_ratings_per_user[usr]
        item_item_cf[item] = missing_item_ratings
        continue
        
    n_most_similar_items = find_most_similar_items(movies_dict, item, 1200)
    #print(n_most_similar_items)
    #break
    average_ratings = dict()    
    for user in test_utility_matrix_i[item]:
        number_of_the_similar_movies_user_has_watched = 0
        accumulated_ratings = []
        distance_scaler = []

        #for user_k in most_similar_users:
        for item_k in n_most_similar_items: #ITEM_K is a tuple of (movie, distance)                 
            #print(user_k)    
            if item_k[0] in util_matrix[user]: #if that particular user has seen it
            #if movie_id in most_similar_users[user_k]:
                number_of_the_similar_movies_user_has_watched += 1

                accumulated_ratings.append(util_matrix[user][item_k[0]])
                #print(user_k[1])
                distance_scaler.append(item_k[1]) #append the distance between user and user_k.

    
        if(number_of_the_similar_movies_user_has_watched == 0):
            average_ratings[user] = mean_ratings_per_user_II[user]
            #I can use the 1. Mean rating of the user(mean of user - row wise). 2. Mean rating all users gave to that particilar movie(mean per movie - ie. column wise). 3. The mean of (mean ratings of the n most similar movies).
            
        else:
            average_ratings[user] = np.matmul(accumulated_ratings, (softmax(distance_scaler).T))
    item_item_cf[item] = average_ratings

#FINAL CALCULATIONS   
output_list = []
for r in ratings_test:
    query = r.split(',')
    test_user = int(query[0])
    test_movie = int(query[1])
    test_time_stamp = query[3]
    u_r = user_user_cf[test_user][test_movie]

    i_r = item_item_cf[test_movie][test_user]
    final_rating = (5.305/7)*i_r + (1.6/7)*u_r
    if (final_rating < 1):
        final_rating = 1
    output_list.append(f"{test_user},{test_movie},{final_rating},{test_time_stamp}")

# SAVE OUTPUT FILE
with open('output.txt', 'w') as f:
    for entry in output_list:
        f.write(f"{entry}")
