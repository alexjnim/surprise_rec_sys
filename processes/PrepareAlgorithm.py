from processes.GetEvaluationMetrics import GetEvaluationMetrics
from data.PreparedData import PreparedData


class PrepareAlgorithm:
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        self.GetEvaluationMetrics = GetEvaluationMetrics()

    def Evaluate(self, preparedData, getAdditionalMetrics, N=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if verbose:
            print("Evaluating accuracy...")
        self.algorithm.fit(preparedData.GetTrainSet())

        predictions = self.algorithm.test(preparedData.GetTestSet())
        metrics["RMSE"] = self.GetEvaluationMetrics.RMSE(predictions=predictions)
        metrics["MAE"] = self.GetEvaluationMetrics.MAE(predictions=predictions)

        if getAdditionalMetrics:
            # Evaluate top-10 with Leave One Out testing
            if verbose:
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(preparedData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(preparedData.GetLOOCVTestSet())

            # Build predictions for all ratings not in the training set
            allPredictions = self.algorithm.test(preparedData.GetLOOCVAntiTestSet())

            # Compute top 10 recs for each user
            topNPredicted = self.GetEvaluationMetrics.GetTopN(allPredictions, N)

            if verbose:
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = self.GetEvaluationMetrics.HitRate(
                topNPredicted, leftOutPredictions
            )
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = self.GetEvaluationMetrics.CumulativeHitRate(
                topNPredicted, leftOutPredictions
            )
            # Compute ARHR
            metrics["ARHR"] = self.GetEvaluationMetrics.AverageReciprocalHitRank(
                topNPredicted, leftOutPredictions
            )

            # Evaluate properties of recommendations on full training set
            if verbose:
                print("Computing recommendations with full data set...")
            self.algorithm.fit(preparedData.GetFullTrainSet())
            allPredictions = self.algorithm.test(preparedData.GetFullAntiTestSet())
            topNPredicted = self.GetEvaluationMetrics.GetTopN(allPredictions, N)
            if verbose:
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = self.GetEvaluationMetrics.UserCoverage(
                topNPredicted,
                preparedData.GetFullTrainSet().n_users,
                ratingThreshold=4.0,
            )
            # Measure diversity of recommendations:
            metrics["Diversity"] = self.GetEvaluationMetrics.Diversity(
                topNPredicted, preparedData.GetSimilarities()
            )

            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = self.GetEvaluationMetrics.Novelty(
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
