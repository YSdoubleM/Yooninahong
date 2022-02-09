
<img src="https://user-images.githubusercontent.com/48639017/152975869-be2c5c86-5302-42e3-a187-415bedd02b26.png"> 

---

# Project Demo

![시연영상](https://user-images.githubusercontent.com/57916633/152984559-19f6a840-87ba-44b6-84f8-f5bd7a9863be.gif)


---

## Main files

### YOLOv5
- https://github.com/ultralytics/yolov5 
- detect.py
- image_augmentation_geo.ipynb
- data: 2nd_data

### image description
- generator_final.ipynb
- gru_generator.py
- senten_generating_model3.h5
- char2idx3.pickle
---

## Notice
---

If you run this project in Google Colaboratory, modify file address in detect.py's line 59 and line 62


```
#### line 59
model = load_model('/content/Yooninahong/description/senten_generating_model3.h5')

#### line 62
with open('/content/Yooninahong/description/data/char2idx3.pickle', 'rb') as fr:
```

---

# Contents🧑🏻‍🦯

1. [프로젝트 소개](#프로젝트-소개) 
    - 수행기간 및 팀원
    - 동기 및 목표
    - 활용 알고리즘
2. [YOLOv5 모델링](#YOLOv5-모델링)
    - 데이터 전처리
    - 모델 성능 개선 과정
    - 영상 인식 결과
3. [GRU 모델링](#GRU-모델링)
    - 문장 생성 모델링 과정
    - YOLOv5 모델과 연결
4. [결론](#결론)
    - 한계점
    - 개선 가능 방향
---
    
# 프로젝트 소개
## 💡 수행기간 및 팀원  

### 기간
- 2022.01.11 ~ 2022.02.03  

### 팀원
- 윤영민: YOLO 모델링
- 이유나: YOLO/GRU 모델링
- 홍성미: GRU 모델링

### 수행일정  

<p align="center"><img width="700" src="https://user-images.githubusercontent.com/48639017/152988514-36f03d1d-87bf-4af8-9553-9fb95d27a2c9.png"></p>   


## 💡 동기 및 목표  

### 프로젝트 배경   

<p><img width="50%" src="https://user-images.githubusercontent.com/48639017/152989614-064fb259-aa8c-4b1a-bebc-d914d16203a6.jpeg"><img width="48.8%" src="https://user-images.githubusercontent.com/48639017/152989697-12714c9b-59a5-41f4-b0ed-45b1460d51da.png"></p>

시각장애인은 점자블록에 의존해 보행하지만, 미관 등의 이유로 저시력 시각 장애인에게는 오히려 방해 요소로 자리잡았으며 점자블록 너무 가까이에 시설물이 설치되어 보행을 방해합니다. </br> 
- 시각 장애인이 전방의 장애물을 미리 인지해 피할 수 있도록 도울 수 없을까?
- 보도에서 안전하게 보행 할 수 있도록 도울 수 없을까?

### 📌 프로젝트 주제

딥러닝을 활용한 시각 장애인 대상 전방 장애물 안내 보행 보조 서비스

### 프로젝트 목표

- 사용자 전방 시야 영상 데이터를 인식해 지정 물체 검출
    - YOLOv5 통해 벡터 추출
- 검출된 물체 활용 장면 description
    - 학습된 RNN 모델에 벡터값 입력해 문장 생성


# YOLOv5 모델링

### 데이터 수집 및 라벨링

- 데이터 수집
    - target object
        - car, bike, motorcycle, electric scooter, person, bollard
        - 보행자가 인지하고 피하거나 주의할 수 있는 물체 위주로 선정.
    - train image data
        - Crawling : Google 이미지 검색 결과 크롤링
        - Kaggle - [People Clothing Segmentation](https://www.kaggle.com/rajkumarl/people-clothing-segmentation) : 다양한 모습의 사람 전신 이미지
    - test video data
        - AI hub - 1인칭 시점 보행영상
        - Youtube - 뉴스 영상
        - Video - 직접 촬영
- Class Labeling
    - image annotation tool : [Supervise.ly](https://supervise.ly/) : Web platform for computer vision, Annotation, traning and deploy.
    - 

### YOLOv5 모델링

### 오류 발견 및 해결

### 성능 평가

# GRU 모델링

# 결론

