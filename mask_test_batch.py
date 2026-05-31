# -*- coding: utf-8 -*-
"""
Created on Thu May 14 01:06:37 2026

@author: Taehyun Kim
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# 한글 폰트
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 모델 로드
model = tf.keras.models.load_model("mask_model.h5")

# 테스트 이미지 여러 개
img_paths = [
    "test2.jpg",
    "test3.jpg",
    "test4.jpg"
]

images = []

# 전처리
for path in img_paths:
    img = image.load_img(path, target_size=(128, 128))
    img = image.img_to_array(img) / 255.0
    images.append(img)

images = np.array(images)

# 예측 (batch 처리)
preds = model.predict(images)

# 결과 출력 + 시각화
for path, pred in zip(img_paths, preds):

    label = "마스크 미착용" if pred[0] > 0.5 else "마스크 착용"

    print(f"{os.path.basename(path)} -> {label} ({pred[0]:.2f})")

    img_show = image.load_img(path)

    plt.imshow(img_show)
    plt.title(f"{label} ({pred[0]:.2f})")
    plt.axis("off")
    plt.show()