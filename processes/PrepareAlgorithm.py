from metrics.RecommenderMetrics import RecommenderMetrics
from data.PreparedData import PreparedData


class PrepareAlgorithm:
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        self.RecommenderMetrics = RecommenderMetrics()

    def Evaluate(self, preparedData, doTopN, N=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if verbose:
            print("Evaluating accuracy...")
        self.algorithm.fit(preparedData.GetTrainSet())

        predictions = self.algorithm.test(preparedData.GetTestSet())
        metrics["RMSE"] = self.RecommenderMetrics.RMSE(predictions=predictions)
        metrics["MAE"] = self.RecommenderMetrics.MAE(predictions=predictions)

        if doTopN:
            # Evaluate top-10 with Leave One Out testing
            if verbose:
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(preparedData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(preparedData.GetLOOCVTestSet())

            # Build predictions for all ratings not in the training set
            allPredictions = self.algorithm.test(preparedData.GetLOOCVAntiTestSet())

            # Compute top 10 recs for each user
            topNPredicted = self.RecommenderMetrics.GetTopN(allPredictions, N)

            if verbose:
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = self.RecommenderMetrics.HitRate(
                topNPredicted, leftOutPredictions
            )
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = self.RecommenderMetrics.CumulativeHitRate(
                topNPredicted, leftOutPredictions
            )
            # Compute ARHR
            metrics["ARHR"] = self.RecommenderMetrics.AverageReciprocalHitRank(
                topNPredicted, leftOutPredictions
            )

            # Evaluate properties of recommendations on full training set
            if verbose:
                print("Computing recommendations with full data set...")
            self.algorithm.fit(preparedData.GetFullTrainSet())
            allPredictions = self.algorithm.test(preparedData.GetFullAntiTestSet())
            topNPredicted = self.RecommenderMetrics.GetTopN(allPredictions, N)
            if verbose:
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = self.RecommenderMetrics.UserCoverage(
                topNPredicted,
                preparedData.GetFullTrainSet().n_users,
                ratingThreshold=4.0,
            )
            # Measure diversity of recommendations:
            metrics["Diversity"] = self.RecommenderMetrics.Diversity(
                topNPredicted, preparedData.GetSimilarities()
            )

            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = self.RecommenderMetrics.Novelty(
                topNPredicted, preparedData.GetPopularityRankings()
            )

        if verbose:
            print("Analysis complete.")

        return metrics

    def GetName(self):
        return self.name

    def GetAlgorithm(self):
        return self.algorithm

    def GetFullyTrainedAlgorithm(self):
        return self.algorithm
