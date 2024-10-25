[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Tm6AYAOm)
# Receipt Text Detection | 영수증 글자 검출
## Team

| ![image](https://github.com/user-attachments/assets/7c5e40bc-1583-4638-a985-642a9d409e83) | ![image](https://github.com/user-attachments/assets/e268f3be-bd44-436e-b590-eda90df67341) | ![image](https://github.com/user-attachments/assets/cf13f5bb-6340-4fee-a099-5be4b0362318) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [기현우](https://github.com/UpstageAILab)             |            [김홍석](https://github.com/UpstageAILab)             |            [최제우](https://github.com/UpstageAILab)          |   
|                            EDA, preprocessing, Model training, test 결과확인                             |                            EDA, preprocessing, Model training, test 결과확인                             |                            EDA, Annotation 수, Model training, test 결과확인                             |

## 0. Overview
### Environment
- Upstage 제공 서버 사용
    - OS : Ubuntu
    - CPU : AMD Ryzen Threadripper 3970X 32-Core Processor
    - Memory : 252 GB
    - GPU : RTX 3090

### Requirements
`pip install -r requirementes.txt`

## 1. Competiton Info

### Overview

- 본 대회는 인공지능 모델을 이용하여 제공된 영수증 이미지에서 문자의 위치를 정확하게 검출하는 문제에 도전하는 대회입니다.

### Timeline

- Start Date : 2024.10.2
- Final submission deadline : 2024.10.24

## 2. Data descrption

### Dataset overview
![image](https://github.com/user-attachments/assets/50d77058-d136-48d4-ad0b-145f9b0d9942)

- train/val/test로 나누어진 영수증 이미지와 JSON annotation 데이터로 구성된 데이터셋입니다.

### EDA

#### Image Size
![image](https://github.com/user-attachments/assets/600fd11d-1053-43c9-9497-25db607b2492)
- 비율: 세로가 더 길다.
- width: 대부분 960, 720 값을 가진다.
- height: 대부분 1280 값을 가진다.

#### Annotation
![image](https://github.com/user-attachments/assets/c908c6db-be6a-44f9-86f2-d6bc16cf061b)
- 일반문자, 특수문자 모두 검출
- 공백에 box가 존재한다는 문제점 존재
- 뒷면 글자에도 box가 존재한다는 문제점 존재
- 마스킹된 부분도 box가 존재한다는 문제점 존재


### Data Processing

#### SAM2 영수증영역 추출
![image](https://github.com/user-attachments/assets/4fd6f166-523c-4f50-a333-75bf4bc597ec)

#### Preprocessing
![image](https://github.com/user-attachments/assets/ee515257-0623-464b-9822-5256c9851832)
- Grayscale Conversion
- Binarization
- Dilation
- Median Blurring
- Diff Calculation, Normalization



## 3. Methods

### 문제점 파악
![image](https://github.com/user-attachments/assets/01ab2a3b-f1d8-45c6-a308-1c9fe64c3439)

- ‘---’ 등 특수문자 인식하지 못하는 경우 발생
- 작은 글자를 인식하지 못하는 경우 발생
- 다른 글자와 하나로 묶는 경우 발생
- 세로 방향으로 인식

### Epoch 증가
- 30 -> 60
- Precision : 0.9651 -> 0.9685
- Recall : 0.8194 -> 0.8004
- H-Mean : 0.8818 -> 0.8699

### 일반문자 단독 학습
- 특수문자 제거
- 일반문자만 학습하여 inference
- 특수문자 : 'orientation', language' 값이 null

- Precision : 0.8818 -> 0.9794
- Recall : 0.8194 -> 0.6135
- H-Mean : 0.7454

### Image size
- Image size 증가
- 이미지 height 최대값에 맞춤
- 640*640 -> 1280*1280
- Precision : 0.8818 -> 0.9724
- Recall : 0.8194 -> 0.8229
- H-Mean : 0.8818 -> 0.8858

### 일반문자, 특수문자 분할 학습
- 일반문자와 특수문자를 각각 학습
- 두 결과를 병합
- Precision : 0.8818 -> 0.9805
- Recall : 0.8194 -> 0.9094
- H-Mean : 0.8818 -> 0.9408

### Annotation 수정
![image](https://github.com/user-attachments/assets/97d75a55-a22a-450b-9c42-6de92db15cb6)
- Annotation에서 box가 잘못 된 부분 제거
- 공백, 마스킹 등 제거
- Precision : 0.8818 -> 0.9818
- Recall : 0.8194 -> 0.9269
- H-Mean : 0.8818 -> 0.9516

### backbone 변경
#### Resnet 50
- 특수문자 제거
- Precision : 0.8818 -> 0.9835
- Recall : 0.8194 -> 0.9269
- H-Mean : 0.8818 -> 0.9520
- 특수문자 포함
- Precision : 0.8818 -> 0.9835
- Recall : 0.8194 -> 0.9117
- H-Mean : 0.8818 -> 0.9436

#### Efficientnet-b0
- Precision : 0.8818 -> 0.9805
- Recall : 0.8194 -> 0.9225
- H-Mean : 0.8818 -> 0.9480

### parameter 조정
- box_thresh : 0.3~0.7 (’---’와 같은 특수문자를 잘 인식하지 못해 검출된 텍스트 영역의 신뢰도 점수 기준을 조정)
- unclip_ratio : 1.0~2.5 (띄어쓰기가 되어 한글자로 인식해야하는 경우 가까운 아래의 글자와 세로로 합쳐져 인식되어 조정)
- min_size : 2~5 (작은 글자를 인식하지 못하여 인식되는 영역의 최소 값 조정)
- 작은 글자를 잘 인식하고 세로방향으로 잘못 인식되는 경우가 줄어들었지만 ’---’같은 특수문자는 잘 인식하지 못함




## 4. Result

### Leader Board

![image](https://github.com/user-attachments/assets/ff69f292-f2e6-489f-9462-e94b62928095)

- H-Mean : 0.9626
- Precision : 0.9849
- Recall : 0.9440


