import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random

class recommendation():
    def __init__(self, embedding_matrix=None ,id=None):
        self.embedding_matrix = embedding_matrix
        self.id = id
        self.data = None

        if self.embedding_matrix:
            self.preferences = list(embedding_matrix.keys())
        else:
            self.preferences = None

    def get_embedding_matrix(self, filepath, vector_length):
        """
        관심사 리스트에 대한 벡터값을 저장하고 있는 임베딩 매트릭스를 불러옵니다.
        
        Arguments:
            filepath: 피클로 저장된 임베딩 벡터 파일의 주소입니다.
            vector_length: 사용하는 임베딩 벡터의 길이입니다.(ex 200, 300)
        Return:
            임베딩 벡터 정보를 담고있는 dict 형태로 return합니다.
        """

        with open(filepath, 'rb') as fw:
            self.embedding_matrix = pickle.load(fw)
        
        self.__vector_length = vector_length
        self.preferences = list(self.embedding_matrix.keys())

    def get_vector(self, preferences):
        """
        사용자의 관심사를 가지고 임베딩 벡터를 만들어내는 코드입니다.
        
        Arguments:
            preferences: 관심사 리스트입니다. id를 받을 것인지, 한글로 받을 것인지는 생각 중

        Return:
            4개의 관심사를 가지고 만든 고유 벡터를 리턴합니다.
        """

        #4개의 관심사를 사용함
        person_matrix = np.zeros((len(preferences), self.__vector_length))

        #2중 for문 말고 다른 방법을 찾아보는 것이 필요
        for i, p in enumerate(preferences):
            person_matrix[i] = self.embedding_matrix[p]
        
        vector = np.average(person_matrix, weights = [40, 10, 5, 1],axis = 0)
            
        return vector

    def fit(self, data):
        """
        모든 사용자의 임베딩 벡터를 통해 클러스터링하는 함수입니다.
        
        Arguments:
            data: 사용자 ID, 관심사를 나타내는 shape입니다. 추후 받아오는 데이터 형식에 맞추어 변경 예정입니다.
        
        Return:
            각 사용자마다 어떤 군집에 포함되어 있는지를 리턴합니다. shape = [(id, number of cluster)]
        """

        data_dict = {i:j for i,j in data}
        self.data = data_dict

        vector_list = []
        id_list = []

        for i, p in data:
            id_list.append(i)
            vector_list.append(self.get_vector(p))

        self.id = id_list

        self.km = KMeans(n_clusters = 4).fit(vector_list)
        self.cluster = self.km.predict(vector_list)
        self.result = dict(zip(self.id, self.cluster))
        
        return self.result

    def similarity(self, id1, id2, data= None):
        """
        두 사용자의 유사도를 출력하는 함수입니다.

        Arguments:
            id1, id2 : 비교할 유저의 id입니다.
            data : 만약 data를 입력한 적이 없다면 사용할 데이터를 입력해줍니다.
                    (fit이나 다른 방식을 통해 이미 data를 입력한 적이 있는 경우 입력 필요 x)

        Return:
            두 사용자의 유사도를 리턴합니다.
        """
        if data:
            data_dict = {i:j for i,j in data}
            self.data = data_dict
        
        if self.data:
            vector1 = self.get_vector(self.data[id1]).reshape((1, -1))
            vector2 = self.get_vector(self.data[id2]).reshape((1, -1))
            return cosine_similarity(vector1, vector2)

        else:
            raise Exception('No data')

    def predict(self, id):
        """
        한 명의 유저를 입력했을 때, 해당 유저에게 추천할 id 리스트를 출력하는 함수입니다.
        유저가 포함되어 있는 그룹의 ID를 무작위 순서로 정렬하여 출력합니다.
        
        Arguments:
            id : 추천 대상 id입니다.

        Return:
            추천 id 리스트를 랜덤한 순서로 리턴합니다.
        """
        vector = self.get_vector(self.data[id]).reshape((1,-1))
        group = self.km.predict(vector)[0]

        re_li = [k for k, v in self.result.items() if v == group and k != id]
        random.shuffle(re_li)
        
        return re_li