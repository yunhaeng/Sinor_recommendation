# Sinor 시놀 (시니어 놀이터)
![image](https://user-images.githubusercontent.com/47114771/169239895-f33f2038-e7cf-4e7f-a466-414088e74996.png)

> 시놀은 시니어 유저들의 관심사 및 요구사항을 이해하여 원하는 단짝과의 매칭을 제공하는 서비스입니다.

## 프로젝트 결과
- [✏️ Final Report](https://drive.google.com/file/d/11w6QcHLcW1S_7OlkjvhFxtO71WSUPCth/view?usp=sharing)
- [📘 Presentation](https://drive.google.com/file/d/1DjCScZ6P2A4Hl_UaMVWNjwLhDv0Qbxwr/view?usp=sharing)
- [🎥 시연 영상 TBA](https://youtube.com)

## 프로젝트 목적

- 시니어 유저들의 관심사 및 요구를 이해하여 원하는 상대와의 매칭을 제공하는 유저 추천 서비스 구현

## 프로젝트 가설 및 예상 결과
- 유저가 제공한 관심사 리스트로 원하는 상대와의 유의미한 매칭이 가능하다. 

## 사용 모델

- K-means 클러스터링 알고리즘
- FastText 단어 유사도

## 추천 알고리즘 구현

1. 유저 프로필 생성 시 24개의 관심사 중 순위별로 4개의 관심사 선택
2. 선택한 4개의 관심사를 이용해 유저별 고유 임베딩 벡터 생성
3. 이때, 관심사 순위별로 가중치 적용 
4. K-Means Clustering을 사용해 모든 유저의 임베딩 벡터를 3개의 군집으로 클러스터링 
5. 가까운 군집 내 유저들을 랜덤으로 추천

## 관심사의 단어 임베딩 벡터 유사도

- FastText를 사용해 관심사별로 임베딩 벡터 생성 후 유사도 측정 
- FastText가 영어를 기반으로 만들어진 임베딩 벡터이므로 24개의 관심사 단어를 영어로 변환 후 임베딩 벡터 생성
- PCA를 사용해 2차원으로 축소 후 시각화

![임베딩시각화](https://user-images.githubusercontent.com/47114771/171775728-c6eb8a0e-147e-4214-a5c4-946220a6f1b9.png)


## 사용 언어 및 라이브러리 
`Python`, `pickle`, `Django`, `sklearn`, `numpy` , `random`

## 사용 가이드(Getting Started)

### 함수(Functions)
- get_embedding_matrix

- get_vector

- fit

- similarity

- predict

### 메소드(Methods)
- embedding_matrix

- preferences

- id

- data

- cluster

- result

## 현 문제점 및 향후 개발 계획

- 콜드 스타트(유저 데이터 부족)로 인하여 성능 향상을 위한 테스트가 어려움
- 데이터가 확보될 경우 실제 프로필 정보와 찜리스트를 활용하여 추천 알고리즘 성능 개선

## 팀원 소개
| 이름   | 담당 업무            |
| ------ | -------------------- |
| [@조윤행](https://github.com/yunhaeng) | 팀장/ 추천 알고리즘 개발 |
| [@강미라](https://github.com/onemira) | 기획/군집수 테스트 |
| [@한승효](https://github.com/monzheld) | 기획/가중치 테스트 |



