# Project Title :  Recommender System based on collaborative filtering

# This code is for the flixter dataaset where it will calculate the trust value of user form the friendship matrix
# and take their ratings from the rating dataset to final predict value.


import math
import scipy
import copy
import numpy as np
from numpy import inf
from scipy import sparse
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
import warnings

# implementation on friend data
line = []
dict2 = {}
file = open('testdemo.txt')
for i in file:
    line.append([int(x) for x in i.split()])
column1 = [int(x[0]) for x in line]
column2 = [int(x[1]) for x in line]

# print column1
# print column2

userlist1 = sorted(set(column1 + column2))
list2 = list(range(0, len(userlist1)))
# print list2
# print userlist1
keys = userlist1
value = list2

# print (len(value)) #Total no. of Unique Users
dictionary = dict(zip(keys, value))
dict2 = sorted(dictionary.items())
# print(dictionary)
w = len(value)
h = len(value)
# FRIENDSHIP MATRIX
friend_mat = np.zeros(shape=(w, h), dtype=int)
file = open('testdemo.txt')
for line in file:
    f = line.strip().split()
    # print int(f[0])
    # print int(f[1])

    friend_mat[dictionary.get(int(f[0]))][dictionary.get(int(f[1]))] = 1
    friend_mat[dictionary.get(int(f[1]))][dictionary.get(int(f[0]))] = 1

print '\nfriendship matrix\n'
print friend_mat
w1 = len(userlist1)


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


cosine_mat = np.zeros(shape=(w1, w1))
for i in range(0, w1):
    for j in range(0, w1):
        value = cos_sim(friend_mat[i, :], friend_mat[j, :])
        cosine_mat[i][j] = value
print "\n\n cosine matrix of friendship matrix \n"
print cosine_mat

new_mat = np.zeros(shape=(w1, w1))
for i in range(0, w1):
    for j in range(0, w1):
        new_mat[i][j] = max(friend_mat[i][j], cosine_mat[i][j])
print '\n\n After put 1 in rating matrix from friendship matrix \n'
print new_mat

rec = np.reciprocal(new_mat)
print '\n\nreciprocal\n'
print rec
dijkstra = shortest_path(rec, method='D', directed=False, unweighted=False, )
print "\n\n dijkstra for shortest path \n"

print dijkstra

trust = np.reciprocal(dijkstra)
print '\n\n Direct Trust \n'
print trust

for i in range(len(trust)):
    for j in range(len(trust)):
        if(trust[i][j]== inf):
            trust[i][j]=0

print "\n Final_Trust : \n \n",trust
#implementation on rating data

line1=[]
file1 = open('ratedemo.txt')
for i in file1:
    line1.append([float(x) for x in i.split()])
#print line1

user_id = [int(x[0]) for x in line1]
movie_id = [int (x[1])for x in line1]
rating = [float(x[2]) for x in line1]

#print user_id
#print movie_id
#print rating


#pearson


unique_user=sorted(set(user_id))
item_list=sorted(set(movie_id))

#print unique_user
#print item_list

key1=unique_user
value1=list(range(0,len(unique_user)))


#Create dictionary for users ID

user_dict=dict(zip(key1,value1))
#print value1
#print user_dict
key2=item_list
value2=list(range(0,len(item_list)))

#creating dictionary for movies ID

movie_dict=dict(zip(key2,value2))
#print movie_dict

w1 =len(unique_user)
h1 =len(item_list)

#print w1
#print h1

rating_mat = np.zeros(shape = (w1,h1))
for i in range(len(user_id)):
   rating_mat[user_dict.get(user_id[i])][movie_dict.get(movie_id[i])]=rating[i]

print "\n\n Rating matrix \n"
print rating_mat

#PEARSN MATRIX
pearsn_mat = np.zeros(shape = (w1,w1))
#print "\n Pearson Cofficient\n:"
for i in range(0, rating_mat.shape[0]):
    for j in range(i+1, rating_mat.shape[0]):
        r, p = pearsonr(rating_mat[i, :], rating_mat[j,: ])
        pearsn_mat[i][j] = r
        pearsn_mat[j][i] = r

print '\n\n PEARSON \n'
print pearsn_mat

# Prediction Function by combining both trust based CF and user-based CF

def prediction(user_id, item_id, ratting_mat, trust_mat, pearsn_mat):
    if(ratting_mat[user_id][item_id] == 0):
        Predicted_value = 0.0
        user_rating_list = ratting_mat[user_id, :]
        #print "u rating",user_rating_list 
        ratting_avg = sum(user_rating_list)/len(user_rating_list)
        #print "avg",ratting_avg
        neighbour_value = trust_mat[user_id, :]
        #print"n_value" ,neighbour_value
        sorted_neighbour_value = sorted(neighbour_value, reverse=True)
        #print"sh",sorted_neighbour_value
        neighbour_list = []
        #print neighbour_list
        temp_list = neighbour_value.copy()
        #print "templist",temp_list
        for i in range(0, len(sorted_neighbour_value)):
            if sorted_neighbour_value[i] != 0:
                index = (temp_list.tolist()).index(sorted_neighbour_value[i])
                #print "index",index
                neighbour_list.append(index)
                temp_list[index] = 0
        #print "Neighbors of target user :",neighbour_list
        nearest_neighbour = []
        for i in neighbour_list:
            if ratting_mat[i][item_id] != 0:
                nearest_neighbour.append(i)
        #print "Nearest Neighbors of target user :",nearest_neighbour
        nominator = 0
        denominator = 0
        for i in nearest_neighbour:
            nominator = nominator + neighbour_value[i] * (ratting_mat[i, :][item_id] - (sum(ratting_mat[i, :])/len(ratting_mat[i, :])))
            denominator = denominator + abs(neighbour_value[i])
        trust_predication_value = nominator / denominator
        #print "Trust prediction value is:", trust_predication_value
        pearson_value = pearsn_mat[user_id, :]
        pear_nominator = 0
        pear_denominator = 0
        for i in nearest_neighbour:
            pear_nominator = nominator + pearson_value[i] * (ratting_mat[i, :][item_id] - (sum(ratting_mat[i, :])/len(ratting_mat[i, :])))
            pear_denominator = denominator + abs(pearson_value[i])
        pear_predication_value = pear_nominator / pear_denominator
        #print "Pearson prediction value is:", pear_predication_value
        Predicted_value = ratting_avg + 0.5*(trust_predication_value) + 0.5*(pear_predication_value)
        #print "\n Predicted Score for target user :",Predicted_value
    return Predicted_value
#prediction(0, 0, rating_mat, trust,pearsn_mat)



'''
# Function for prediction by trust based Collaborative filtering

def prediction_by_trust(user_id, item_id, ratting_mat, trust_mat):
    if(ratting_mat[user_id][item_id] == 0):
        Predicted_value = 0.0
        user_rating_list = ratting_mat[user_id, :] 
        ratting_avg = sum(user_rating_list)/len(user_rating_list)
        print "rating avg",ratting_avg
        neighbour_value = trust_mat[user_id, :]
        print "nav_trust",neighbour_value
        
        sorted_neighbour_value = sorted(neighbour_value, reverse=True)
        neighbour_list = []
        temp_list = neighbour_value.copy()
        for i in range(0, len(sorted_neighbour_value)):
            if sorted_neighbour_value[i] != 0:
                index = (temp_list.tolist()).index(sorted_neighbour_value[i])
                #print "index",index
                neighbour_list.append(index)
                temp_list[index] = 0
        print "Neighbors of target user :",neighbour_list
        nearest_neighbour = []
        for i in neighbour_list:
            if ratting_mat[i][item_id] != 0:
                nearest_neighbour.append(i)
        print "Nearest Neighbors of target user :",nearest_neighbour
        nominator = 0
        denominator = 0
        for i in nearest_neighbour:
            nominator = nominator + neighbour_value[i] * (ratting_mat[i, :][item_id] - (sum(ratting_mat[i, :])/len(ratting_mat[i, :])))
            denominator = denominator + abs(neighbour_value[i])
        trust_predication_value = nominator / denominator
        print "Trust prediction value is:", trust_predication_value
    return trust_predication_value
prediction_by_trust(0,0, rating_mat, trust)

# Function for prediction by pearson correlation

        
def prediction_by_pearsn(user_id, item_id, ratting_mat,  pearsn_mat):
    if(ratting_mat[user_id][item_id] == 0):
        Predicted_value = 0.0
        user_rating_list = ratting_mat[user_id, :]
        ratting_avg = sum(user_rating_list)/len(user_rating_list)
        neighbour_value = pearsn_mat[user_id, :]
        sorted_neighbour_value = sorted(neighbour_value, reverse=True)
        neighbour_list = []
        temp_list = neighbour_value.copy()
        for i in range(0, len(sorted_neighbour_value)):
            if sorted_neighbour_value[i] != 0:
                index = (temp_list.tolist()).index(sorted_neighbour_value[i])
                neighbour_list.append(index)
                temp_list[index] = 0
        print "Neighbors of target user :",neighbour_list
        nearest_neighbour = []
        for i in neighbour_list:
            if ratting_mat[i][item_id] != 0:
                nearest_neighbour.append(i)
        print "Nearest Neighbors of target user :",nearest_neighbour
        
        #print "Trust prediction value is:", trust_predication_value
        pearson_value = pearsn_mat[user_id, :]
        pear_nominator = 0
        pear_denominator = 0
        for i in nearest_neighbour:
            pear_nominator = pear_nominator + pearson_value[i] * ratting_mat[i, :][item_id] - sum(ratting_mat[i, :])/len(ratting_mat[i, :])
            pear_denominator = pear_denominator + abs(pearson_value[i])
        pear_predication_value =  ratting_avg+ pear_nominator / pear_denominator
        print "Pearson prediction value is:", pear_predication_value
        #print "\n Predicted Score for target user :",Predicted_value
    return Predicted_value
#prediction_by_pearsn(5, 85, rating_mat, pearsn_mat)


'''


# In this final_mat we have stored both the prediction values as well as the the original rating data from the dataset

final_mat =np.zeros(shape = (w1,h1))
final_mat = copy.deepcopy(rating_mat)
for i in range(len(final_mat)):
    for j in range(len(final_mat[0])):
        if final_mat[i][j]==0:
            final_mat[i][j]=prediction(i, j, rating_mat, trust,pearsn_mat)
print '\n recommended matrix \n'            
print final_mat

'''
# This matrix contains only prediction values:
final_mat =np.zeros(shape = (w1,h1))
for i in range(len(final_mat)):
    for j in range(len(final_mat[0])):
        if rating_mat[i][j]==0:
            final_mat[i][j]= prediction(i, j, rating_mat, trust,pearsn_mat)
print final_mat

'''

mean_absolute_error =mean_absolute_error(rating_mat,final_mat )
print ("\n")
result = {'mean_absolute_error': mean_absolute_error}
print result




            
        



