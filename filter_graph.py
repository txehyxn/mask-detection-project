
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = r"C:\Users\Taehyun Kim\OneDrive\바탕 화면\202538002김태현\malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 실험 결과
filters = [32, 64, 128]

accuracy = [
    0.939072847366333,
    0.9119205474853516,
    0.9211920499801636
]

# 그래프 크기
plt.figure(figsize=(8,5))

# 선 그래프
plt.plot(filters, accuracy, marker='o')

# 그래프 제목
plt.title('필터 개수에 따른 모델 정확도')

# 축 이름
plt.xlabel('필터 개수')
plt.ylabel('검증 정확도')

# 값 표시 (그래프 위 숫자)
for x, y in zip(filters, accuracy):
    plt.text(x, y, f'{y:.3f}')

# 격자
plt.grid(True)

# 그래프 출력
plt.show()