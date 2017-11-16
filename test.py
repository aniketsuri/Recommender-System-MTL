import numpy as np
import csv
from sklearn import svm 
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.optimize import minimize
import scipy

f_obj = open("ratings.csv")
reader = csv.DictReader(f_obj, delimiter=',')

userId = []
movieId = []


for line in reader:
    if int(line['userId']) not in userId:
        userId.append(int(line['userId']))
    if int(line['movieId']) not in movieId:
        movieId.append(int(line['movieId']))    
f_obj.close()


num_user = len(userId)
num_movie = len(movieId)


Rating = np.zeros((num_movie,num_user))

IsRated = np.zeros((num_movie,num_user))

f_obj = open("ratings.csv")
reader = csv.DictReader(f_obj, delimiter=',')

for line in reader:
    Rating[movieId.index(int(line['movieId'])),userId.index(int(line['userId']))] = float(line['rating'])
    movieId.index(int(line['movieId']))
    IsRated[movieId.index(int(line['movieId'])),userId.index(int(line['userId']))] = 1


print "Rating : ",Rating[:,0].size," ",Rating[0,:].size
print "IsRated : ",IsRated[:,0].size," ",IsRated[0,:].size

np.savez('Rating.npz',a=Rating,b=IsRated)



num_features = 20

def cost(param, Rating,IsRated,num_features):
    num_movie = Rating[:,0].size
    num_user = Rating[0,:].size
    #print "In cost function : num_user = ",num_user," and num_movie = ",num_movie
    #print "param size = ",param.size
    movies = np.reshape(param[:num_movie*num_features],(num_movie,num_features))
    users = np.reshape(param[num_movie*num_features:],(num_user,num_features))
    
    movies_grad = np.zeros((num_movie,num_features))
    users_grad = np.zeros((num_user,num_features))

    movies = np.matrix(movies)
    users = np.matrix(users)

    pred = movies * users.T

    error = np.multiply(pred - Rating, IsRated)
    squared_error = np.power(error, 2) 
    cost = (1. / 2) * np.sum(squared_error)

    movies_grad = error * users
    users_grad = error.T * movies

    grad = np.concatenate((np.ravel(movies_grad),np.ravel(users_grad)))
    return cost, grad




data = np.load('Rating.npz')
Rating = data['a']
IsRated = data['b']
num_movie = Rating[:,0].size
num_user = Rating[0,:].size

similarity = np.zeros((num_user,num_user))

C = 5

for i in range(num_user):
    for j in range(num_user):
        meani = np.mean(Rating[IsRated[:,i]==1 ,i])
        meanj = np.mean(Rating[IsRated[:,j]==1,j])
        coli = Rating[scipy.logical_and(IsRated[:,i]==1, IsRated[:,j]==1),i] - meani
        colj = Rating[scipy.logical_and(IsRated[:,i]==1, IsRated[:,j]==1),j] - meanj
        numerator = np.sum(np.multiply(coli,colj))
        denominator = np.sqrt(np.sum(np.multiply(coli,coli))) * np.sqrt(np.sum(np.multiply(colj,colj)))
        if denominator == 0:
            similarity[i,j] = 0
        else:
            similarity[i,j] = (min(C,coli.size)/C)*(numerator/denominator)
    if i%5 == 0:
        print i," rows completed"

np.savez('Similarity.npz',a=similarity)


data = np.load('Similarity.npz')
similarity = data['a']

#print "similarity : ",similarity[:,0].size," ",similarity[0,:].size

closest_twenty_index = np.zeros((num_user,20))
closest_twenty_index = np.array(closest_twenty_index,dtype=int)
for i in range(num_user):
    ind = 0
    cutoff = np.sort(similarity[i,:])[-21]
    for j in range(num_user):
    	if j == i:
    	    continue
    	if ind >= 20:
    	    break
        if similarity[i,j] == 1:
            continue
        elif similarity[i,j] >= cutoff:
            closest_twenty_index[i,ind] = j
            ind += 1
#print "closest_twenty_index : ",closest_twenty_index[:,0].size," ",closest_twenty_index[0,:].size
#print "closest_twenty_index[0] : ",closest_twenty_index[0]

Cost_before_learning = []
Cost_after_learning = []
Pred_Rating = np.zeros((num_movie,num_user))
Pred_Rating = np.matrix(Pred_Rating)
active_user = 0
for active_user in range(num_user):
    New_Rating = Rating[:,active_user]
    New_Rating = np.matrix(New_Rating).T
    New_Rating = np.concatenate((New_Rating[:,0],Rating[:,closest_twenty_index[active_user]]),axis=1)
    
    New_IsRated = IsRated[:,active_user]
    New_IsRated = np.matrix(New_IsRated).T
    New_IsRated = np.concatenate((New_IsRated[:,0],IsRated[:,closest_twenty_index[active_user]]),axis=1)

    New_num_user = New_Rating[0,:].size
    New_num_movie = New_Rating[:,0].size

    movies = np.random.random((New_num_movie,num_features))
    users = np.random.random((New_num_user,num_features))
    param = np.concatenate((np.ravel(movies),np.ravel(users)))

    
    param = np.concatenate((np.ravel(movies),np.ravel(users[0])))
    Cost, grad = cost(param=param,Rating=np.matrix(New_Rating[:,0]),IsRated=np.matrix(New_IsRated[:,0]),num_features=num_features)
    print "Cost before learning for active user : ",Cost," for active user index = ",active_user
    Cost_before_learning.append(Cost)

    param = np.concatenate((np.ravel(movies),np.ravel(users)))

    fmin = minimize(fun=cost, x0=param, args=(New_Rating, New_IsRated, num_features),  
                    method='CG', jac=True, options={'maxiter': 50})

    movies = np.matrix(np.reshape(fmin.x[:New_num_movie * num_features], (New_num_movie, num_features)))  
    users = np.matrix(np.reshape(fmin.x[New_num_movie * num_features:], (New_num_user, num_features)))
    param = np.concatenate((np.ravel(movies),np.ravel(users)))


    param = np.concatenate((np.ravel(movies),np.ravel(users[0])))
    Cost, grad = cost(param=param,Rating=np.matrix(New_Rating[:,0]),IsRated=np.matrix(New_IsRated[:,0]),num_features=num_features)
    print "Cost after learning for active user : ",Cost," for active user index = ",active_user
    Cost_after_learning.append(Cost)
    print "\n"

    active_user_vec = np.matrix(users[0]).T
    pred = np.matrix(movies) * active_user_vec
    Pred_Rating[:,active_user] = np.array(pred)
    

Costs = []
Costs.append(Cost_before_learning)
Costs.append(Cost_after_learning)
Costs = np.matrix(Costs).T

np.savetxt("Costs_before_after.csv", Costs, delimiter=",")

np.savetxt("Original_Ratings.csv", Rating, delimiter=",")
np.savetxt("Predicted_Ratings.csv", Pred_Rating, delimiter=",")
