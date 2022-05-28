from surprise import KNNBasic
from collections import defaultdict
import operator
from MovieLens import MovieLens

class CollaborativeUser:

    def __init__(self, user):
        self.userId = user

    def recommend(self):

        loader = MovieLens()
        self.data = loader.loadMovieLensLatestSmall()
        self.rankings = loader.getPopularityRanks()


        sim_options = {
         'name':'cosine',
         'user_based': True
        }

        training_set = self.data.build_full_trainset()

        model = KNNBasic(sim_options=sim_options)
        model.fit(training_set)

        similarities_matrix = model.compute_similarities()

        inner_user_id = training_set.to_inner_uid(self.userId)
        user_row = similarities_matrix[inner_user_id]
        similar_users = []

        for neighboor, score in enumerate(user_row):
            if (neighboor != inner_user_id ):
                similar_users.append( (neighboor, score) )

        best_neighboors = []

        # Tìm những điểm lân cận ...

        for temp_user in similar_users:
            if temp_user[1] > 0.95:
                best_neighboors.append(temp_user)


        combinated_neighboor_ratings = defaultdict(float)

        # Áp dụng bộ lọc Cộng tác ...

        for temp_neighboor in best_neighboors:
            similarity_score = temp_neighboor[1]
            neighboor_id = temp_neighboor[0]
            neighboor_ratings = training_set.ur[neighboor_id]
            for temp_rating in neighboor_ratings:
                temp_score = temp_rating[1]*similarity_score
                temp_key = temp_rating[0]
                actual_value = 0
                if (temp_key in combinated_neighboor_ratings.keys()):
                    actual_value = combinated_neighboor_ratings[temp_key]
                combinated_neighboor_ratings[temp_key] = temp_score + actual_value

        for user_items in training_set.ur[inner_user_id]:
            if user_items[0] in combinated_neighboor_ratings.keys():
                combinated_neighboor_ratings[user_items[0]] = -1

        sorted_ratings = sorted(combinated_neighboor_ratings.items(), key=operator.itemgetter(1), reverse=True)

        top_n = 10
        iterator = 0

        print("\nRecommend:\n")

        for key in sorted_ratings:
            iterator += 1
            movie_id = training_set.to_raw_iid(key[0])
            print(loader.getMovieName(int(movie_id)))
            if iterator == top_n:
                break
