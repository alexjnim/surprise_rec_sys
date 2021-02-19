from data.DataLoader import DataLoader
from processes.AlgorithmStore import AlgorithmStore
from surprise import NormalPredictor
from surprise import SVD, SVDpp
from customAlgorithms.ContentKNNAlgorithm import ContentKNNAlgorithm
from customAlgorithms.TunedSVD import TunedSVD

import random
import numpy as np


def loadDataFunction():
    dataLoader = DataLoader()
    print("Loading data...")
    loadedData = dataLoader.loadData()
    print("\nComputing popularity rankings...")
    rankings = dataLoader.getPopularityRanks()
    return (dataLoader, loadedData, rankings)


def prepareAlgorithmStore(dataLoader, loadedData, rankings):
    # Load up common loadedData set for the recommender algorithms

    # Construct an AlgorithmStore to store all algorithms for evaluation
    algorithmStore = AlgorithmStore(loadedData, rankings)

    # Just make random recommendations
    Random = NormalPredictor()
    algorithmStore.AddAlgorithm(Random, "Random")

    # # Add the content based KNN here
    # contentKNN = ContentKNNAlgorithm()
    # algorithmStore.AddAlgorithm(contentKNN, "ContentKNN")

    # # Add SVD
    # SVDAlgorithm = SVD(random_state=10)
    # algorithmStore.AddAlgorithm(SVDAlgorithm, "SVD")

    # # Add SVDpp
    # SVDPlusPlusAlgorithm = SVDpp(random_state=10)
    # algorithmStore.AddAlgorithm(SVDPlusPlusAlgorithm, "SVD++")

    # # Add TunedSVD
    # TunedSVDAlgorithm = TunedSVD(dataLoader, loadedData, rankings)
    # algorithmStore.AddAlgorithm(TunedSVDAlgorithm, "Tuned SVD")

    return algorithmStore


def run_recsys():
    (dataLoader, loadedData, rankings) = loadDataFunction()
    algorithmStore = prepareAlgorithmStore(dataLoader, loadedData, rankings)

    algorithmStore.GetMetrics(getAdditionalMetrics=False)

    # algorithmStore.SampleTopNRecs(dataLoader=dataLoader, testSubject=85, N=10)


if __name__ == "__main__":
    random.seed(0)
    run_recsys()
