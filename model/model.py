import numpy as np
import pickle
from sklearn.cluster import KMeans

class recommendation():
    def __init__(self, preferences, id=None):
        self.preferences = preferences
        self.id = id
        
    def get_embbeding_matrix(self, filepath):
        """
        관심사 리트스에 대한 벡터값을 저장하고 있는 임베딩 매트릭스를 불러옵니다.
        
        Arguments:
            filepath: 피클로 저장된 임베딩 벡터 파일의 주소입니다.

        return:
            임베딩 벡터 정보를 담고있는 dict 형태로 return합니다.
        """

        with open(filepath, 'rb') as fw:
            self.embedding_matrix = pickle.load(fw)

    def get_vector(self, preferences):
        """
        사용자의 관심사를 가지고 임베딩 벡터를 만들어내는 코드입니다.
        
        Arguments:
            preferences: 관심사 리스트입니다. id를 받을 것인지, 한글로 받을 것인지는 생각 중
        
        return:
            3개의 관심사를 가지고 만든 고유 벡터를 리턴합니다.
        """

        #4개의 관심사를 사용함
        person_matrix = np.zeros((4, 200))

        #2중 for문 말고 다른 방법을 찾아보는 것이 필요
        for p in preferences:
            for j in range(1, 4):
                person_matrix[j-1] = self.embedding_matrix[p]
        
        vector = np.average(person_matrix, weights= [0.4, 0.3, 0.2, 0.1], axis = 0)
            
        return vector

    def fit(self, data):
        """
        모든 사용자의 임베딩 벡터를 통해 클러스터링하는 함수입니다.
        
        Arguments:
            data: 사용자 ID, 관심사를 나타내는 shape입니다. 추후 받아오는 데이터 형식에 맞추어 변경 예정입니다.
        
        return:
            각 사용자마다 어떤 군집에 포함되어 있는지를 나타내는 
        """
        vector_list = []
        id_list = []
        
        for i, p in data:
            id_list.append(i)
            vector_list.append(self.get_vector(p))

        #print(len(vector_list), vector_list)
        km = KMeans(n_clusters = 4).fit(vector_list)

        return list(zip(id_list,km.predict(vector_list)))