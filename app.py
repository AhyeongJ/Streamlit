import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


st.write('[2조] 안상후, 오서연, 정아영 :sunglasses:') # 해당 내용을 수정해서 사이트를 자유롭게 꾸밀 수 있다.

st.title('금속 주조 공정 최적화')

# df = pd.read_csv("data/data_week4.csv", encoding='cp949')
# st.write(df)

# # NaN 값을 0으로 변환하고, 'D'는 1로 변환하는 코드 실행
# df['사탕신호'] = df['사탕신호'].replace('D', 1)  # 'D'를 1로 변환
# df['사탕신호'] = df['사탕신호'].fillna(0)  # NaN을 0으로 변환

# # 사탕신호의 unique 값 확인
# df['사탕신호'].unique()

# plt.rcParams['font.family'] = 'Malgun Gothic'

# # 사탕신호가 1일 때 불량판정이 0인 비율과 1인 비율을 계산
# filtered_df = df[df['사탕신호'] == 1]
# filtered_df_zero = df[df['사탕신호'] == 0]

# # 불량판정의 비율 계산
# #passorfail_0_ratio = (filtered_df['불량판정'].value_counts(normalize=True)[0] * 100).round(2)
# #사탕신호 1인 것 중 실제 불량이 0인 비율 0%라 오류 뜸

# passorfail_0_ratio = 0.0
# #사탕신호 1인 것 중 실제 불량이 0인 비율은 0.0으로 수동 대체

# passorfail_1_ratio = (filtered_df['불량판정'].value_counts(normalize=True)[1] * 100).round(2)
# #사탕신호 1인 것 중 실제 불량이 1인 비율 100%

# passorfail_0_ratio_zero = (filtered_df_zero['불량판정'].value_counts(normalize=True)[0] * 100).round(2)
# #사탕신호 0인 것 중 실제 불량도 0인 비율 97.67%

# passorfail_1_ratio_zero = (filtered_df_zero['불량판정'].value_counts(normalize=True)[1] * 100).round(2)
# #사탕신호 0인 것 중 실제 불량이 1인 비율 2.33%

# # 비율 데이터를 리스트로 저장
# labels = ['NaN', 'D']
# fail_0_ratios = [passorfail_0_ratio_zero, passorfail_0_ratio]  # 불량판정 0 비율
# fail_1_ratios = [passorfail_1_ratio_zero, passorfail_1_ratio]  # 불량판정 1 비율

# x = np.arange(len(labels))  # 라벨의 개수만큼 x 위치 생성
# width = 0.35  # 막대 너비

# fig, ax = plt.subplots(figsize=(8, 6))

# # 불량판정 0 막대 그래프 그리기
# bars1 = ax.bar(x - width/2, fail_0_ratios, width, label='정상', color='#7FB3D5')

# # 불량판정 1 막대 그래프 그리기
# bars2 = ax.bar(x + width/2, fail_1_ratios, width, label='불량', color='#F1948A')

# # 각각의 막대에 % 비율 추가
# for bar in bars1:
#     ax.annotate(f'{bar.get_height():.2f}%', 
#                 (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
#                 ha='center', va='bottom')

# for bar in bars2:
#     ax.annotate(f'{bar.get_height():.2f}%', 
#                 (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
#                 ha='center', va='bottom')

# # 그래프 레이블 설정
# ax.set_ylabel('불량율 (%)')
# ax.set_title('사탕신호에 따른 불량판정 비율 비교')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

# # 그래프 레이아웃 조정 및 출력
# plt.tight_layout()
# plt.show()

# region = st.selectbox("Target region", ["East", "West", "South", "Central"])
# df_region = df[df["Region"]==region]
# st.write(region)

# fig, ax = plt.subplots()
# ax = plt.scatter(df_region["Sales"], df_region["Profit"])
# st.pyplot(fig)


# # 탭 생성 : 첫번째 탭의 이름은 Tab A 로, Tab B로 표시합니다.
# tab1, tab2= st.tabs(['다이캐스팅' , 'Tab B'])

# with tab1:
#   #tab A 를 누르면 표시될 내용
#   st.write('hello')

# with tab2:
#   #tab B를 누르면 표시될 내용
#   st.write('hi')

