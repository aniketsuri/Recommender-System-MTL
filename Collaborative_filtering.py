import numpy as np
import pandas as pd  
from scipy.sparse.linalg import svds
from lshash import LSHash
import datetime
from sklearn.neighbors import LSHForest
import math

def roundOffRating(rating):
	if rating < 1:
		return 1
	if rating > 5:
		return 5
	doubleOfRating = rating * 2.0
	NearestInteger = int(doubleOfRating)
	if abs(doubleOfRating - int(doubleOfRating)) > abs(doubleOfRating-int(doubleOfRating+1)):
		NearestInteger = int(doubleOfRating+1)
	return float(NearestInteger) / 2.0

def error(Ratings_normalised,movies,users):
	error = 0
	start_time = datetime.datetime.now()
	for index, row in training.iterrows():
		userId = userIdMapping[row['userId']]
		movieId = movieIdMapping[row['movieId']]
		error = error + np.square(Ratings_normalised[movieId,userId] - np.dot(movies[movieId,:],users[:,userId]))
	end_time = datetime.datetime.now()
	print "Error computation time : ",str((end_time-start_time))
	print "Error per rating : ", np.sqrt(error / float(index + 1))
	return error

def train(steps,alpha,beta,Ratings_normalised,movies,users):
	for step in xrange(steps):
		start_time = datetime.datetime.now()
		for index, row in training.iterrows():
			userId = userIdMapping[row['userId']]
			movieId = movieIdMapping[row['movieId']]
			eij = Ratings_normalised[movieId,userId] - np.dot(movies[movieId,:],users[:,userId])
			old_user = users[:,userId]
			users[:,userId] = users[:,userId] + alpha * (2 * eij * np.transpose(movies[movieId,:]) - beta * users[:,userId])
			movies[movieId,:] = movies[movieId,:] + alpha * (2 * eij * np.transpose(old_user) - beta * movies[movieId,:])
			if index % 50000 == 0:
				print int(index + 1), " ratings processed"
		print " step ", step
		if step % 5 == 0:
			 print error(Ratings_normalised,movies,users)
		np.savez('usersMoviesVectors.npz',users=users,movies=movies)
		end_time = datetime.datetime.now()
		print "train time : ",str((end_time-start_time))

def initializeVectors(user_dim,movie_dim):
	start_time = datetime.datetime.now()
	users = 0.1 * np.random.randn(user_dim,num_user)
	movies = 0.1 * np.random.randn(num_movie,movie_dim)
	end_time = datetime.datetime.now()
	print "initializeVectors time : ",str((end_time-start_time))
	return users,movies

def testing(test):
	default_movieIds = avgMovieRating.argsort()[-5:][::-1]
	submission = pd.DataFrame(columns=('userId', 'movieId', 'rating'))
	count = 0
	for index,row in test.iterrows():
		userId_original = int(row['userId'])
		if userId_original in userIdMapping:
			userId = userIdMapping[userId_original]
			ratings = np.dot(movies,users[:,userId])
			ratings = np.multiply(ratings + avgMovieRating,1-IsRated[:,userId])	
			movieIds = ratings.argsort()[-5:][::-1]
			for i in range(5):
				submission.loc[count+i] = [int(userId_original),int(movieIdInverseMapping[movieIds[i]]),roundOffRating(ratings[movieIds[i]])]
				count = count + 1
		else:
			for i in range(5):
				submission.loc[count+i] = [int(userId_original),int(movieIdInverseMapping[default_movieIds[i]]),roundOffRating(ratings[movieIds[i]])]
				count = count + 1
			
		if index % 100 == 0:
			print int(index+1)," users processed"
	submission['userId'] = submission['userId'].astype(np.dtype(np.int32))		
	submission['movieId'] = submission['movieId'].astype(np.dtype(np.int32))		
	submission.to_csv('submission.csv', index=False)

training = pd.read_csv('data/training.csv')
training = training[['userId','movieId','rating']]


def loadData():

	userIds = set(training['userId'].values) # To remove duplicates
	userIds = list(userIds) # For indexing
	userIdMapping = {}
	for i in range(len(userIds)):
		userIdMapping[userIds[i]] = i

	movieIds = set(training['movieId'].values) # To remove duplicates
	movieIds = list(movieIds) # For indexing
	movieIdMapping = {}
	for i in range(len(movieIds)):
		movieIdMapping[movieIds[i]] = i

	num_user = len(userIds)
	num_movie = len(movieIds)

	Rating = np.zeros((num_movie,num_user))
	IsRated = np.zeros((num_movie,num_user),dtype=np.int8)

	prevUserId = -1
	prevNewUserId = -1

	for index, row in training.iterrows():
		newUserId = userIdMapping[row['userId']] # userIds.index(row['userId'])
		newMovieId = movieIdMapping[row['movieId']] # movieIds.index(row['movieId'])
		IsRated[newMovieId,newUserId] = int(1)
		Rating[newMovieId,newUserId] = float(row['rating'])
		if index % 10000 == 0:
			print index, " ratings processed"

	userIdMapping = {}
	userIdInverseMapping = {}
	for i in range(len(userIds)):
		userIdMapping[userIds[i]] = i
		userIdInverseMapping[i] = userIds[i]

	movieIdMapping = {}
	movieIdInverseMapping = {}
	for i in range(len(movieIds)):
		movieIdMapping[movieIds[i]] = i
		movieIdInverseMapping[i] = movieIds[i]

	return IsRated,userIds,movieIds,userIdMapping,userIdInverseMapping,movieIdMapping,movieIdInverseMapping


IsRated,userIds,movieIds,userIdMapping,userIdInverseMapping,movieIdMapping,movieIdInverseMapping = loadData()
print "Loaded successfully"

avgMovieRating = np.true_divide(Rating.sum(axis=1),IsRated.sum(axis=1))
print "avgMovieRating : ", avgMovieRating.shape

# slow, but works
Ratings_normalised = Rating
for movieIndex in range(num_movie):
	mask = (IsRated[movieIndex,:] == 1)
	Ratings_normalised[movieIndex,mask] = Rating[movieIndex,mask] - avgMovieRating[movieIndex]
	if movieIndex % 50 == 0:
		print int(movieIndex + 1), " rows normalised"


users,movies = initializeVectors(user_dim=50,movie_dim=50)

train(steps=20,alpha=0.03,beta=0.02,Ratings_normalised=Ratings_normalised,movies=movies,users=users)

test = pd.read_csv('data/test.csv')
testing(test)

