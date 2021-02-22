from surprise import AlgoBase


class HybridAlgorithm(AlgoBase):
    """
    This class will help build an ensemble model out of the input algorithms and the input weights
    """

    def __init__(self, algorithms, weights, sim_options={}):
        AlgoBase.__init__(self)
        self.algorithms = algorithms
        self.weights = weights

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        for algorithm in self.algorithms:
            algorithm.fit(trainset)

        return self

    def estimate(self, user, item):

        sumScores = 0
        sumWeights = 0

        for idx in range(len(self.algorithms)):
            sumScores += self.algorithms[idx].estimate(user, item) * self.weights[idx]
            sumWeights += self.weights[idx]

        return sumScores / sumWeights
