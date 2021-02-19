from surprise import AlgoBase
from surprise import PredictionImpossible
from data.DataLoader import DataLoader
import math
import numpy as np
import heapq


class ContentKNNAlgorithm(AlgoBase):
    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # Compute item similarity matrix based on content attributes

        # Load up genre vectors for every item
        dataLoader = DataLoader()
        genres = dataLoader.getGenres()
        years = dataLoader.getYears()
        # mes = dataLoader.getMiseEnScene()

        print("Computing content-based similarity matrix...")

        # Compute genre distance for every item combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

        for thisRating in range(self.trainset.n_items):
            if thisRating % 100 == 0:
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating + 1, self.trainset.n_items):
                thisItemID = int(self.trainset.to_raw_iid(thisRating))
                otherItemID = int(self.trainset.to_raw_iid(otherRating))
                genreSimilarity = self.computeGenreSimilarity(
                    thisItemID, otherItemID, genres
                )
                yearSimilarity = self.computeYearSimilarity(
                    thisItemID, otherItemID, years
                )
                # mesSimilarity = self.computeMiseEnSceneSimilarity(thisItemID, otherItemID, mes)
                self.similarities[thisRating, otherRating] = (
                    genreSimilarity * yearSimilarity
                )
                self.similarities[otherRating, thisRating] = self.similarities[
                    thisRating, otherRating
                ]

        print("...done.")

        return self

    def computeGenreSimilarity(self, item1, item2, genres):
        genres1 = genres[item1]
        genres2 = genres[item2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for item in range(len(genres1)):
            x = genres1[item]
            y = genres2[item]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y

        return sumxy / math.sqrt(sumxx * sumyy)

    def computeYearSimilarity(self, item1, item2, years):
        diff = abs(years[item1] - years[item2])
        sim = math.exp(-diff / 10.0)
        return sim

    def computeMiseEnSceneSimilarity(self, item1, item2, mes):
        mes1 = mes[item1]
        mes2 = mes[item2]
        if mes1 and mes2:
            shotLengthDiff = math.fabs(mes1[0] - mes2[0])
            colorVarianceDiff = math.fabs(mes1[1] - mes2[1])
            motionDiff = math.fabs(mes1[3] - mes2[3])
            lightingDiff = math.fabs(mes1[5] - mes2[5])
            numShotsDiff = math.fabs(mes1[6] - mes2[6])
            return (
                shotLengthDiff
                * colorVarianceDiff
                * motionDiff
                * lightingDiff
                * numShotsDiff
            )
        else:
            return 0

    def estimate(self, user, item):

        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible("User and/or item is unkown.")

        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[user]:
            genreSimilarity = self.similarities[item, rating[0]]
            neighbors.append((genreSimilarity, rating[1]))

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if simScore > 0:
                simTotal += simScore
                weightedSum += simScore * rating

        if simTotal == 0:
            raise PredictionImpossible("No neighbors")

        predictedRating = weightedSum / simTotal

        return predictedRating
