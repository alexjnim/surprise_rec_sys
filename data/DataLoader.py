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

    # def loadData(self):

    #     # Look for files relative to the directory we are running from
    #     # os.chdir(os.path.dirname(sys.argv[0]))

    #     ratingsDataset = 0

    #     reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)

    #     ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

    #     with open(self.itemsPath, newline="", encoding="ISO-8859-1") as csvfile:
    #         itemReader = csv.reader(csvfile)
    #         next(itemReader)  # Skip header line
    #         for row in itemReader:
    #             itemID = int(row[0])
    #             itemName = row[1]
    #             self.itemID_to_name[itemID] = itemName
    #             self.name_to_itemID[itemName] = itemID

    #     return ratingsDataset

    def getUserRatings(self, user):
        userRatings = self.ratingsDF[self.ratingsDF[config.userIDColumn] == user][
            [config.itemIDColumn, config.ratingsColumn]
        ]
        return userRatings

    # def getUserRatings(self, user):
    #     userRatings = []
    #     hitUser = False
    #     with open(self.ratingsPath, newline="") as csvfile:
    #         ratingReader = csv.reader(csvfile)
    #         next(ratingReader)
    #         for row in ratingReader:
    #             userID = int(row[0])
    #             if user == userID:
    #                 itemID = int(row[1])
    #                 rating = float(row[2])
    #                 userRatings.append((itemID, rating))
    #                 hitUser = True
    #             if hitUser and (user != userID):
    #                 break

    #     return userRatings
    def getPopularityRanks(self):
        item_freq = self.ratingsDF[config.itemIDColumn].value_counts()
        popularity_rankings = pd.Series(
            range(1, len(item_freq) + 1, 1), index=item_freq.index
        )
        return popularity_rankings

    # def getPopularityRanks(self):
    #     ratings = defaultdict(int)
    #     rankings = defaultdict(int)
    #     with open(self.ratingsPath, newline="") as csvfile:
    #         ratingReader = csv.reader(csvfile)
    #         next(ratingReader)
    #         for row in ratingReader:
    #             itemID = int(row[1])
    #             ratings[itemID] += 1
    #     rank = 1
    #     for itemID, ratingCount in sorted(
    #         ratings.items(), key=lambda x: x[1], reverse=True
    #     ):
    #         rankings[itemID] = rank
    #         rank += 1
    #     return rankings
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

    # def getGenres(self):
    #     genres = defaultdict(list)
    #     genreIDs = {}
    #     maxGenreID = 0
    #     with open(self.itemsPath, newline="", encoding="ISO-8859-1") as csvfile:
    #         itemReader = csv.reader(csvfile)
    #         next(itemReader)  # Skip header line
    #         for row in itemReader:
    #             itemID = int(row[0])
    #             genreList = row[2].split("|")
    #             genreIDList = []
    #             for genre in genreList:
    #                 if genre in genreIDs:
    #                     genreID = genreIDs[genre]
    #                 else:
    #                     genreID = maxGenreID
    #                     genreIDs[genre] = genreID
    #                     maxGenreID += 1
    #                 genreIDList.append(genreID)
    #             genres[itemID] = genreIDList
    #     # Convert integer-encoded genre lists to bitfields that we can treat as vectors
    #     for (itemID, genreIDList) in genres.items():
    #         bitfield = [0] * maxGenreID
    #         for genreID in genreIDList:
    #             bitfield[genreID] = 1
    #         genres[itemID] = bitfield

    #     return genres

    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.itemsPath, newline="", encoding="ISO-8859-1") as csvfile:
            itemReader = csv.reader(csvfile)
            next(itemReader)
            for row in itemReader:
                itemID = int(row[0])
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

    # def getItemName(self, itemID):
    #     if itemID in self.itemID_to_name:
    #         return self.itemID_to_name[itemID]
    #     else:
    #         return ""

    def getItemID(self, itemName):
        if itemName in list(self.itemsDF[config.itemTitleColumn]):
            return self.itemsDF[config.itemIDColumn][
                self.itemsDF[config.itemTitleColumn] == itemName
            ].iloc[0]
        else:
            return "Not available"

    # def getItemID(self, itemName):
    #     if itemName in self.name_to_itemID:
    #         return self.name_to_itemID[itemName]
    #     else:
    #         return 0
