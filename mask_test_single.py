# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:46:09 2026

@author: Taehyun Kim
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# 한글 폰트 (윈도우)
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 모델 로드
model = tf.keras.models.load_model("mask_model.h5")

# 테스트 이미지 1장
img_path = "test10.jpg"

# 이미지 전처리
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 예측
pred = model.predict(img_array)[0][0]

# 결과 라벨
label = "마스크 미착용" if pred > 0.5 else "마스크 착용"

# 출력
print(f"{os.path.basename(img_path)} -> {label} ({pred:.2f})")

# 시각화
img_show = image.load_img(img_path)

plt.imshow(img_show)
plt.title(f"{label} ({pred:.2f})")
plt.axis("off")
plt.show()