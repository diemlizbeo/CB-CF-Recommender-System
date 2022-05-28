import math
import numpy as np
import heapq
from surprise import AlgoBase
from MovieLens import MovieLens
from surprise import PredictionImpossible

class ContentBasedAlgorithm(AlgoBase):

    def __init__(self, user_number):
        AlgoBase.__init__(self)
        self.user_number = user_number

    def recommend(self):
        self.fit()
        testing_set = self.test_without_user(int(self.user_number))
        predictions = self.test(testing_set)
        recommendations = []
        print ("Recommend:")
        for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
            intMovieID = int(movie_id)
            recommendations.append((intMovieID , estimated_rating  ))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        for ratings in recommendations[:10]:
            print(self.loader.getMovieName(ratings[0]), ratings[1])

    def fit(self):

        self.loader = MovieLens()
        self.data = self.loader.loadMovieLensLatestSmall()
        self.trainset = self.data.build_full_trainset()
        # start tranning
        AlgoBase.fit(self, self.trainset)
        genres = self.loader.getGenres()
        years = self.loader.getYears()

        self.similarities_matrix = np.zeros((self.trainset.n_items, self.trainset.n_items))

        for itemA in range(self.trainset.n_items):
            for itemB in range(itemA+1, self.trainset.n_items):
                itemAA = int(self.trainset.to_raw_iid(itemA))
                itemBB = int(self.trainset.to_raw_iid(itemB))
                year_similarity = self.compute_year_similarity(itemAA, itemBB, years)
                genre_similarity = self.calculate_genre_similarity(itemAA, itemBB, genres)
                self.similarities_matrix[itemA, itemB] = genre_similarity*year_similarity
                self.similarities_matrix[itemB, itemA] = genre_similarity*year_similarity

        # finished
        return self

    def calculate_genre_similarity(self, itemA, itemB, genres):
        itemA = genres[itemA]
        itemB = genres[itemB]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(itemA)):
            x = itemA[i]
            y = itemB[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y

        return sumxy/math.sqrt(sumxx*sumyy)

    def compute_year_similarity(self, itemA, itemB, years):
        diff = abs(years[itemA] - years[itemB])
        sim = math.exp(-diff / 10.0)
        return sim

    def estimate(self, u, i):
        neighbors = []
        n_neighbors = 40

        for rating in self.trainset.ur[u]:
            genre_similarity = self.similarities_matrix[i,rating[0]]
            neighbors.append( (genre_similarity, rating[1]) )

        k_neighbors = heapq.nlargest(n_neighbors, neighbors, key=lambda t: t[0])

        similarity_total = 0
        sum_weight = 0

        for (temp_similarity, rating) in k_neighbors:
            if (temp_similarity > 0):
                similarity_total += temp_similarity
                sum_weight += temp_similarity * rating

        if (similarity_total == 0):
            raise PredictionImpossible('No neighbors')

        predicted_rating = sum_weight / similarity_total

        return predicted_rating

    def test_without_user(self, test_user):
        trainset = self.trainset
        fill = trainset.global_mean
        test_data = []
        u = trainset.to_inner_uid(str(test_user))

        user_items = []
        for rating in trainset.ur[u]:
            user_items.append(rating[0])

        # print(user_items)

        for i in trainset.all_items():
            if i not in user_items:
                test_data.append((trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill))

        return test_data
