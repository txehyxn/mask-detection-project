# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:59:13 2026

@author: Taehyun Kim
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

from matplotlib import font_manager, rc

from tensorflow.keras.preprocessing import image
import numpy as np


font_path = r"C:\Users\Taehyun Kim\OneDrive\바탕 화면\202538002김태현\malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 데이터셋 경로
train_dir = r"C:\Users\Taehyun Kim\OneDrive\바탕 화면\202538002김태현\data"

# 이미지 전처리
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# 학습 데이터
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# 검증 데이터
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 클래스 확인
print(train_data.class_indices)

# CNN 모델 생성
model = Sequential()

# 첫 번째 합성곱 층
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

# 두 번째 합성곱 층
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

# 평탄화
model.add(Flatten())

# 완전 연결층
model.add(Dense(128, activation='relu'))

# 출력층
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 출력
model.summary()

# 모델 학습
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# 정확도 그래프
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('모델 정확도')
plt.ylabel('정확도')
plt.xlabel('학습 횟수')

plt.legend(['훈련 데이터', '검증 데이터'])

plt.show()

# 손실 그래프
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('모델 손실값')
plt.ylabel('손실값')
plt.xlabel('학습횟수')

plt.legend(['훈련 데이터', '검증 데이터'])

plt.show()

model.save("mask_model.h5")

print(train_data.class_indices)