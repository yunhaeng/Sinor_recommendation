import numpy as np
import json
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import random

class recommendation():
    def __init__(self, embedding_matrix=None ,id=None):
        """
        """
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
            self.embedding_matrix = json.load(fw)
        
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
        
        vector = np.average(person_matrix, weights=[4, 3, 2, 1],axis = 0)
            
        return vector

    def fit(self, data, n_cluster= 3):
        """
        모든 사용자의 임베딩 벡터를 통해 클러스터링하는 함수입니다.
        
        Arguments:
            data: 사용자 ID, 관심사를 나타내는 shape입니다. [유저 id, [선호1, 선호2, 선호3, 선호4]]
                데이터 크기는 최소한 군집의 수보다 많아야합니다.
            n_cluster : 나누고 싶은 군집 수 입니다.
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
        self.__n_cluster = n_cluster

        if len(data) < self.__n_cluster:
            self.cluster = None
            self.result = None
        else:
            self.km = KMeans(n_clusters= self.__n_cluster, random_state=42).fit(vector_list)
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

    def predict(self, id, batch_size = None):
        """
        한 명의 유저를 입력했을 때, 해당 유저에게 추천할 id 리스트를 출력하는 함수입니다.
        유저가 포함되어 있는 그룹의 ID를 우선적으로 출력한 뒤, 남은 cluster의 id를 출력합니다.
        
        Arguments:
            id : 추천 대상 id입니다.
            batch_size : 각 군집 id 리스트의 크기입니다. default = None

        Return:
            추천 id 리스트를 랜덤한 순서로 리턴합니다.
        """
        if self.result:
            vector = self.get_vector(self.data[id]).reshape((1,-1))
            group = self.km.predict(vector)[0]

            re_li = [k for k, v in self.result.items() if v == group and k != id]
            random.shuffle(re_li)
            re_li = re_li[:batch_size]

            #이중 for문
            for i in range(self.__n_cluster):
                if i != group:
                    temp_li = [k for k, v in self.result.items() if v == i]
                    random.shuffle(temp_li)
                    re_li.extend(temp_li[:batch_size])

            return re_li
        
        else:
            re_li = list(self.data.keys())
            re_li.remove(id)
            if re_li:
                random.shuffle(re_li)
            return re_li