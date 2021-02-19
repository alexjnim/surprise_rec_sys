from data.DataLoader import DataLoader
from processes.ContentKNNAlgorithm import ContentKNNAlgorithm
from processes.AlgorithmStore import AlgorithmStore
from surprise import NormalPredictor

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

    # Add the content based KNN here
    # contentKNN = ContentKNNAlgorithm()
    # algorithmStore.AddAlgorithm(contentKNN, "ContentKNN")

    # Just make random recommendations
    Random = NormalPredictor()
    algorithmStore.AddAlgorithm(Random, "Random")
    return algorithmStore


def run_recsys():
    (dataLoader, loadedData, rankings) = loadDataFunction()
    algorithmStore = prepareAlgorithmStore(dataLoader, loadedData, rankings)

    # algorithmStore.GetMetrics(doTopN=False)

    algorithmStore.SampleTopNRecs(dataLoader=dataLoader, testSubject=85, N=10)


if __name__ == "__main__":

    random.seed(0)
    run_recsys()
