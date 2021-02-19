import itertools

from surprise import accuracy
from collections import defaultdict


class RecommenderMetrics:
    def __init__(self):
        self.store = 0

    def MAE(self, predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(self, predictions):
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(self, predictions, N=10, minimumRating=4.0):
        topN = defaultdict(list)

        for userID, itemID, actualRating, estimatedRating, _ in predictions:
            if estimatedRating >= minimumRating:
                topN[int(userID)].append((int(itemID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:N]

        return topN

    def HitRate(self, topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutItemID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for itemID, predictedRating in topNPredicted[int(userID)]:
                if int(leftOutItemID) == int(itemID):
                    hit = True
                    break
            if hit:
                hits += 1

            total += 1

        # Compute overall precision
        return hits / total

    def CumulativeHitRate(self, topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for (
            userID,
            leftOutItemID,
            actualRating,
            estimatedRating,
            _,
        ) in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if actualRating >= ratingCutoff:
                # Is it in the predicted top 10 for this user?
                hit = False
                for itemID, predictedRating in topNPredicted[int(userID)]:
                    if int(leftOutItemID) == itemID:
                        hit = True
                        break
                if hit:
                    hits += 1

                total += 1

        # Compute overall precision
        return hits / total

    def RatingHitRate(self, topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for (
            userID,
            leftOutItemID,
            actualRating,
            estimatedRating,
            _,
        ) in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for itemID, predictedRating in topNPredicted[int(userID)]:
                if int(leftOutItemID) == itemID:
                    hit = True
                    break
            if hit:
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print(rating, hits[rating] / total[rating])

    def AverageReciprocalHitRank(self, topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for (
            userID,
            leftOutItemID,
            actualRating,
            estimatedRating,
            _,
        ) in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for itemID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if int(leftOutItemID) == itemID:
                    hitRank = rank
                    break
            if hitRank > 0:
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(self, topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for itemID, predictedRating in topNPredicted[userID]:
                if predictedRating >= ratingThreshold:
                    hit = True
                    break
            if hit:
                hits += 1

        return hits / numUsers

    def Diversity(self, topNPredicted, simsAlgo):
        N = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                Item1 = pair[0][0]
                Item2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(Item1)
                innerID2 = simsAlgo.trainset.to_inner_iid(Item2)
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                N += 1

        S = total / N
        return 1 - S

    def Novelty(self, topNPredicted, rankings):
        N = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                itemID = rating[0]
                rank = rankings[itemID]
                total += rank
                N += 1
        return total / N
