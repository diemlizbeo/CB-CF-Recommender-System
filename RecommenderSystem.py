from urllib.request import Request
from MovieLens import MovieLens
from CollaborativeUser import CollaborativeUser
from CollaborativeItem import CollaborativeItem
from ContentBasedAlgorithm import ContentBasedAlgorithm
class RecommenderSystem:

    def __init__(self):
        self.RecommenderStart()

    def RecommenderStart(self):
        
        user_number = input("Nhập id người dùng của bạn:")
        recommender_method = input("""
        1- Collaborative User-based
        2- Collaborative Item-based
        3- Content Base
        Bạn muốn sử dụng phương pháp đề xuất nào: """)
        
        if(recommender_method == "1"):
            self.collaborative_user = CollaborativeUser(user_number)
            self.collaborative_user.recommend()
        elif(recommender_method == "2"):
            self.collaborative_item = CollaborativeItem(user_number)
            self.collaborative_item.recommend()
        elif(recommender_method == "3"):
            self.content_user = ContentBasedAlgorithm(user_number)
            self.content_user.recommend()
        else:
            print("Số phương pháp không hợp lệ")
RecommenderSystem()
