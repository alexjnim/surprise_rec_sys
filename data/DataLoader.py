import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from config import config
from collections import defaultdict
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self):
        self.ratingsPath = config.ratingsPath
        self.itemsPath = config.itemsPath
        self.itemsDF = pd.read_csv(self.itemsPath)
        self.ratingsDF = pd.read_csv(self.ratingsPath)

    def loadData(self):
        # preparing pandas dataframes
        self.ratingsDF = self.ratingsDF[
            [config.userIDColumn, config.itemIDColumn, config.ratingsColumn]
        ]

        reader = Reader(
            rating_scale=(0, 5),
            # line_format="user item rating timestamp",
            # sep=",",
            # skip_lines=1,
        )
        self.ratingsDataset = Dataset.load_from_df(self.ratingsDF, reader=reader)
        # self.ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)
        return self.ratingsDataset

    def getUserRatings(self, user):
        userRatings = self.ratingsDF[self.ratingsDF[config.userIDColumn] == user][
            [config.itemIDColumn, config.ratingsColumn]
        ]
        return userRatings

    def getPopularityRanks(self):
        item_freq = self.ratingsDF[config.itemIDColumn].value_counts()
        popularity_rankings = pd.Series(
            range(1, len(item_freq) + 1, 1), index=item_freq.index
        )
        return popularity_rankings

    def getGenres(self):
        # this will store the genres for each film
        genres = defaultdict(list)
        # this will store the keys for each genre
        genreIDs = {}
        # this will track the maximum number of IDs
        maxGenreID = 0

        for i in range(len(self.itemsDF)):
            row = self.itemsDF.iloc[i]
            itemID = row[0]
            genreList = row[2].split("|")
            genreIDList = []
            for genre in genreList:
                # if the genre is already listed, it will append to genreIDList
                if genre in genreIDs:
                    genreID = genreIDs[genre]
                # if the genre isn't listed yet, it will be listed and assigned a genreID before being appended
                else:
                    genreID = maxGenreID
                    genreIDs[genre] = genreID
                    maxGenreID += 1
                genreIDList.append(genreID)
            genres[itemID] = genreIDList

        # this will effectively do one-hot-encoding to all the genres and return a list with 0's and 1's for each itemID
        for (itemID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[itemID] = bitfield

        return genres

    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        for i in range(len(self.itemsDF)):
            row = self.itemsDF.iloc[i]
            itemID = row[0]
            title = row[1]
            m = p.search(title)
            year = m.group(1)
            if year:
                years[itemID] = int(year)
        return years

    def getItemName(self, itemID):
        if itemID in list(self.itemsDF[config.itemIDColumn]):
            return self.itemsDF[config.itemTitleColumn][
                self.itemsDF[config.itemIDColumn] == itemID
            ].iloc[0]
        else:
            return "Not available"

    def getItemID(self, itemName):
        if itemName in list(self.itemsDF[config.itemTitleColumn]):
            return self.itemsDF[config.itemIDColumn][
                self.itemsDF[config.itemTitleColumn] == itemName
            ].iloc[0]
        else:
            return "Not available"
