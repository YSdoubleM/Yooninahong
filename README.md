
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
    ![image](https://user-images.githubusercontent.com/57916633/153195391-053061fa-e25b-4caa-8eea-73a8f1606d4a.png)


### YOLOv5 모델링
---

- 1차 학습
    - yaml 파일 수정 후 train 바로 진행
    - test video에서의 인식률이 다소 저조
    
- 2차 학습
    - image augmentation후 train - image_augmentation_geo.py
        - 1200장의 image를 3750장으로 augmentation.
        - 회전, 일부 가리기 등 적용
        
    - test video의 물체 검출 인식률 소폭 상승
- 3차 학습
    - class 추가 및 augmentation 후 train
        - 기존의 class를 세분화하여 정면 이미지와 후면 이미지로 분리
        - train batch 확인 - augmentation에 의해서 회전된 이미지에 대한 bounding box의 변형이 올바르지 못한 것을 확인
        
- 4차 학습
    - bounding box augmentation 오류 수정
    - 최종 모델 생성


### 오류 발견 및 해결
---

- Threshold 값 조절
    - 1차 학습시 모델 train 성능은 약 90%
    - test video에서 전혀 인식하지 못해 threshold값을 0.5에서 0.25로 하향조절
    
- Train image 재작업
    - 인식 정확도 낮은 class의 image를 추가 수집 후 labeling
    - 한 class 내 형태 다향성으로 인해 인식 정확도 낮은 경우, class를 세부 분할
    
- Image augmentation
    - 실제 test video 인식률 향상을 위해 train image 밝기, 일부 가리기, 회전 을 적용해 증복
    - YOLOv5 내부 코드의 aumgmeetation을 사용하지 않고 imgaug module을 사용해 이미지 증강 sequence 코드 추가
    
- Bounding box augmentation
    - image augmentation 시 기존의 bounding box가 함꼐 변형되어야 하나, 잘못 변형되는 오류가 발생
    - [x_center, y_center, w, h] 좌표를 실제 좌표값으로 변경하여 augmentation 적용시키는 방식으로 해결
    ![image](https://user-images.githubusercontent.com/57916633/153210451-11caa5b9-e35b-4c71-bc59-591934762788.png)

---

### 성능 평가
---
- PR curve
    ![image](https://user-images.githubusercontent.com/57916633/153210729-e1676809-274a-4ab8-8806-5e06aaca6f1d.png)
    - PR curve의 아래 면적인 mAP(mean Average Precision)값을 가지고 성능을 평가
    - 약 93%의 정확도를 보임
- Confusion matrix
    ![image](https://user-images.githubusercontent.com/57916633/153211692-3dc15a78-4b1b-4c43-89a7-f138da4ed000.png)
    - 대각선이 진하게 나타난 것을 통해 object가 올바르게 예측되고 있다는 것을 알 수 있음
    - 빨간색 박스로 표시한 background FN은 각 object를 background라고 잘못 예측한 경우의 비율
    - 볼라드는 15%의 오차가 있었지만 그 외의 클래스는 대부분 5% 미만으로 준수한 결과
    
    ![image](https://user-images.githubusercontent.com/57916633/153212053-6f85bc8f-efc1-4d74-9c83-c90245b7d1bb.png)
    - 빨간색 박스로 표시된 FP는 탐지할 object가 없는 background를 object가 존재한다고 잘못 예측한 경우
    - 3분의 1 이상이 볼라드 였으며, 이는 볼라드의 특징이 비교적 다른 object에 비해 간단하여 벌어진 문제로 판단
    

---
# GRU 모델링

# 결론

