# Sinor 시놀 (시니어 놀이터)
![image](https://user-images.githubusercontent.com/47114771/169239895-f33f2038-e7cf-4e7f-a466-414088e74996.png)

> 시놀은 시니어 유저들의 관심사 및 요구사항을 이해하여 원하는 단짝과의 매칭을 제공하는 서비스입니다.

## 프로젝트 목적

- 단짝 추천 서비스을 위한 추천 알고리즘 개발
- DL/ML 모델 파이프라인 구축하여 애플리케이션 프레임워크 내에서 실제 동작할 수 있는 모델 구현

## 프로젝트 가설 및 예상 결과
- 유저가 제공한 관심사 리스트로 원하는 상대와의 유의미한 매칭이 가능하다. 

## 사용 모델

- K-means 클러스터링 알고리즘
- FastText 단어 유사도

![image](https://user-images.githubusercontent.com/47114771/169246078-49709fa2-a491-427e-ad2d-58754d486e27.png)


## 사용 언어 및 라이브러리 
`Python`, `pickle`, `Django`, `sklearn`, `numpy` , `random`

## 필수 설치 모듈
- gensim > =0.13.1 (for Word2Vec)

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

## contributors
[@조윤행](https://github.com/yunhaeng)
[@강미라](https://github.com/onemira)
[@한승효](https://github.com/monzheld)
