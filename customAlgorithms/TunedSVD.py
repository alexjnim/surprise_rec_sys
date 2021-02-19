from surprise import SVD
from surprise.model_selection import GridSearchCV

import random
import numpy as np


def TunedSVD(dataLoader, loadedData, rankings):
    print("Searching for best parameters...")
    param_grid = {
        "n_epochs": [20, 30],
        "lr_all": [0.005, 0.010],
        "n_factors": [50, 100],
    }
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

    gs.fit(loadedData)

    # best RMSE score
    print("Best RMSE score attained: ", gs.best_score["rmse"])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params["rmse"])

    params = gs.best_params["rmse"]
    SVDtuned = SVD(
        n_epochs=params["n_epochs"],
        lr_all=params["lr_all"],
        n_factors=params["n_factors"],
    )
    return SVDtuned