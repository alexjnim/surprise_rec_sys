from data.DataLoader import DataLoader
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

    def GetMetrics(self, getAdditionalMetrics):
        results = {}
        for preparedAlgorithm in self.algorithms:
            print("Evaluating ", preparedAlgorithm.GetName(), "...")
            results[preparedAlgorithm.GetName()] = preparedAlgorithm.Evaluate(
                self.preparedData, getAdditionalMetrics
            )

        # Print results
        print("\n")

        if getAdditionalMetrics:
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
        if getAdditionalMetrics:
            print(
                "HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better."
            )
            print(
                "cHR:       Cumulative Hit Rate; hit rate, confined to recommendation above a certain threshold. Higher is better."
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

    def SampleTopNRecs(self, dataLoader, testSubject=1, N=10, numberOfLatestItems=0):

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())

            # has to be trained on the full trainset (100% of the data) here to give full recommendations
            # the GetMetrics part were trained on 75% data, LOOCV data, etc.
            print("\nBuilding recommendation model...")
            trainSet = self.preparedData.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)

            print("Computing recommendations...")
            testSet = self.preparedData.GetAntiTestSetForUser(testSubject)

            predictions = algo.GetAlgorithm().test(testSet)
            print(predictions[0])
            recommendations = []

            print("\nFor", algo.GetName(), "we recommend:")
            for userID, itemID, actualRating, estimatedRating, _ in predictions:
                intItemID = int(itemID)
                recommendations.append((intItemID, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for recommendation in recommendations[:N]:
                print(dataLoader.getItemName(recommendation[0]), recommendation[1])

            # get the recommendations from the most recent year
            if numberOfLatestItems > 0:
                latestRecommendations = getLatestItemsRecommendations(
                    dataLoader, numberOfLatestItems, recommendations
                )
                for recommendation in latestRecommendations:
                    print(dataLoader.getItemName(recommendation[0]), recommendation[1])


def getLatestItemsRecommendations(dataLoader, numberOfLatestItems, recommendations):
    """This function will get recommendations from the latest year, and match it with the ratings predictions of the given algorithm, thus sorting them by the highest rating prediction and thus allowing the system to recommend a new item with the highest rating prediction.

    Args:
        dataLoader (DataLoader Class): generated from data/DataLoader.py, this will get the list of newest items
        numberOfLatestItems (int): number of new items to have in the predictions
        recommendations (list): list of recommendations containing item IDs and predicted ratings

    Returns:
        latestRecommendations (list): containing item IDs and predicted ratings of the newest items in dataset
    """
    latestItems = dataLoader.getLatestItems()
    latestRecommendations = []
    for recommendation in recommendations:
        if recommendation[0] in latestItems:
            latestRecommendations.append(recommendation)
    latestRecommendations.sort(key=lambda x: x[1], reverse=True)

    if numberOfLatestItems > len(latestRecommendations):
        return latestRecommendations
    else:
        return latestRecommendations[:numberOfLatestItems]