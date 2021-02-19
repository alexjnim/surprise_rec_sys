from data.PreparedData import PreparedData
from processes.PrepareAlgorithm import PrepareAlgorithm


class AlgorithmStore:

    algorithms = []

    def __init__(self, loadedData, rankings):
        preparedData = PreparedData(loadedData, rankings)
        self.preparedData = preparedData

    def AddAlgorithm(self, algorithm, name):
        preparedAlgorithm = PrepareAlgorithm(algorithm, name)
        self.algorithms.append(preparedAlgorithm)

    def GetMetrics(self, doTopN):
        results = {}
        for preparedAlgorithm in self.algorithms:
            print("Evaluating ", preparedAlgorithm.GetName(), "...")
            results[preparedAlgorithm.GetName()] = preparedAlgorithm.Evaluate(
                self.preparedData, doTopN
            )

        # Print results
        print("\n")

        if doTopN:
            print(
                "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm",
                    "RMSE",
                    "MAE",
                    "HR",
                    "cHR",
                    "ARHR",
                    "Coverage",
                    "Diversity",
                    "Novelty",
                )
            )
            for (name, metrics) in results.items():
                print(
                    "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name,
                        metrics["RMSE"],
                        metrics["MAE"],
                        metrics["HR"],
                        metrics["cHR"],
                        metrics["ARHR"],
                        metrics["Coverage"],
                        metrics["Diversity"],
                        metrics["Novelty"],
                    )
                )
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print(
                    "{:<10} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"]
                    )
                )

        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if doTopN:
            print(
                "HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better."
            )
            print(
                "cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better."
            )
            print(
                "ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better."
            )
            print(
                "Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better."
            )
            print(
                "Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations"
            )
            print("           for a given user. Higher means more diverse.")
            print(
                "Novelty:   Average popularity rank of recommended items. Higher means more novel."
            )

    def SampleTopNRecs(self, dataLoader, testSubject=1, N=10):

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())

            # has to be trained on the full trainset (100% of the data) here to give full recommendations
            # the GetMetrics part were trained on 75%, LOOCV, etc.
            print("\nBuilding recommendation model...")
            trainSet = self.preparedData.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)

            print("Computing recommendations...")
            testSet = self.preparedData.GetAntiTestSetForUser(testSubject)

            predictions = algo.GetAlgorithm().test(testSet)

            recommendations = []

            print("\nFor", algo.GetName(), "we recommend:")
            for userID, itemID, actualRating, estimatedRating, _ in predictions:
                intItemID = int(itemID)
                recommendations.append((intItemID, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:N]:
                print(dataLoader.getItemName(ratings[0]), ratings[1])
