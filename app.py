
import streamlit as st
from PIL import Image

st.write('Hello, *World!* :sunglasses:') # 해당 내용을 수정해서 사이트를 자유롭게 꾸밀 수 있다.

st.title('this is title')
st.header('this is header')
st.subheader('this is subheader')

# 탭 생성 : 첫번째 탭의 이름은 Tab A 로, Tab B로 표시합니다.
tab1, tab2= st.tabs(['Tab A' , 'Tab B'])

with tab1:
  #tab A 를 누르면 표시될 내용
  st.write('hello')

with tab2:
  #tab B를 누르면 표시될 내용
  st.write('hi')

# 데이터 프레임
import pandas as pd
df = pd.DataFrame({
     '첫 번째 컬럼': [1, 2, 3, 4],
     '두 번째 컬럼': [10, 20, 30, 40]
     })
st.write(df)
