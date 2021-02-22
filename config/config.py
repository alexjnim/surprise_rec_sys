ratingsPath = "data/ratings.csv"
itemsPath = "data/movies.csv"

userIDColumn = "userId"
itemIDColumn = "movieId"
ratingsColumn = "rating"
itemTitleColumn = "title"

# this is for the removeOutliers in data/DataLoader.py
# set to False here, as we know that the movielens data has been cleaned, so very unlikely to contain bots
removeOutlierUsers = False
outlierStdDev = 3.0