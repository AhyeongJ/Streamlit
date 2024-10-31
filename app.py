import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import itertools
from PIL import Image
import base64


# 랜덤 시드 설정
np.random.seed(42)

# 데이터 불러오기
df = pd.read_csv("data/data_week4.csv", encoding='cp949')

## 전처리 =====================================================================

# 데이터명 변경
df.columns = ['Unnamed:0', '작업라인', '제품명', '금형명', '수집날짜', '수집시각', '일자별제품생산번호',
              '가동여부', '비상정지', '용탕온도', '설비작동사이클시간', '제품생산사이클시간',
              '저속구간속도', '고속구간속도', '용탕량', '주조압력', '비스킷두께', '상금형온도1',
              '상금형온도2', '상금형온도3', '하금형온도1', '하금형온도2', '하금형온도3', '슬리브온도',
              '형체력', '냉각수온도', '전자교반가동시간', '등록일시', '불량판정', '사탕신호', '금형코드',
              '가열로']

df['불량판정'].info()

# 결측치 30% 이상인 행 제거
df = df[df.isnull().mean(axis=1) * 100 <= 30]

# '사탕신호'가 'D'인 행 Drop
df = df[df['사탕신호'] != 'D']

# 열 Drop (등록일시 포함)
df = df.drop(columns=['Unnamed:0', '일자별제품생산번호', '작업라인', '사탕신호', '제품명', '금형명', '비상정지', '수집날짜', '수집시각', '등록일시'])

# '불량판정', '금형코드' 열을 범주형으로 변환
df['불량판정'] = df['불량판정'].astype('category')
df['금형코드'] = df['금형코드'].astype('category')

# '가동여부' 변환: 가동이면 0, 아니면 1
df['가동여부'] = df['가동여부'].apply(lambda x: 0 if x == '가동' else 1)

# 가열로, 용탕온도, 용탕량, 하금형온도, 상금형온도3에서 결측치 발견
df.isna().sum()

# '가열로' 열의 'NaN' 값을 'F'(측정X)로 변경
df['가열로'] = df['가열로'].fillna('F')

# 용탕온도, 용탕량, 하금형온도3에 대해 선형 보간을 적용
df['용탕온도'] = df['용탕온도'].interpolate(method='linear', limit_direction='both')
df['용탕량'] = df['용탕량'].interpolate(method='linear', limit_direction='both')
df['하금형온도3'] = df['하금형온도3'].interpolate(method='linear', limit_direction='both')
df['상금형온도3'] = df['상금형온도3'].interpolate(method='linear', limit_direction='both')

df.isna().sum()

# 이상치 제거
df = df[df['설비작동사이클시간'] <= 400] # 1
df = df[df['제품생산사이클시간'] <= 450] # 2
df = df[df['저속구간속도'] <= 60000] # 1
df = df[df['상금형온도1'] <= 1400] # 1
df = df[df['상금형온도2'] <= 4000] # 1
df = df[df['하금형온도3'] <= 60000] # 1
df = df[df['형체력'] <= 60000] # 3
df = df[df['냉각수온도'] <= 1400] # 9

# 코드 ===========================================================================
## p.25 IV구하는 예시 표
def 동일_데이터_구간_나누기(df, column, num_bins=10):
    # 데이터 수 동일한 구간으로 나누기 (중복 허용)
    df['구간'] = pd.qcut(df[column].rank(method='first'), q=num_bins, labels=False)

    # 각 구간에 대한 통계 정보 계산
    grouped = df.groupby('구간').apply(lambda x: pd.Series({
        '데이터 건수': len(x),
        '불량판정 0 개수': (x['불량판정'] == 0).sum(),
        '불량판정 1 개수': (x['불량판정'] == 1).sum()
    })).reset_index()

    # WOE와 IV 계산
    total_good = df['불량판정'].value_counts()[0]
    total_bad = df['불량판정'].value_counts()[1]

    # 각 구간의 비율 계산
    grouped['불량판정 0 비율'] = grouped['불량판정 0 개수'] / total_good
    grouped['불량판정 1 비율'] = grouped['불량판정 1 개수'] / total_bad

    # WOE 계산
    grouped['WOE'] = np.log(grouped['불량판정 1 비율'] / grouped['불량판정 0 비율'].replace(0, np.nan))

    # IV 계산
    grouped['IV'] = (grouped['불량판정 1 비율'] - grouped['불량판정 0 비율']) * grouped['WOE']
    iv_value = grouped['IV'].sum()  # 전체 IV 값

    # 수치 범위 추가
    grouped['수치 범위'] = grouped.apply(lambda x: f"{df[df['구간'] == x['구간']][column].min()} - {df[df['구간'] == x['구간']][column].max()}", axis=1)

    # 수치 범위를 첫 번째 열로 이동
    grouped = grouped[['수치 범위', '구간', '데이터 건수', '불량판정 0 개수', '불량판정 1 개수', 
                       '불량판정 0 비율', '불량판정 1 비율', 'WOE', 'IV']]

    return grouped, iv_value

# 예시 호출
result, total_iv = 동일_데이터_구간_나누기(df, '주조압력', num_bins=10)


#==============================================================
## p.25 각 변수에 대해 IV계산 함수
def calculate_iv(df, target):
    iv_dict = {}
    
    # 타겟 변수가 숫자로 변환되도록 조정
    if df[target].dtype.name == 'category':
        df[target] = df[target].cat.codes  # 범주형을 정수형으로 변환

    for col in df.select_dtypes(include=[np.number]).columns:
        # 해당 변수의 이벤트 비율과 비이벤트 비율 계산
        total_events = df[target].sum()
        total_non_events = df[target].count() - total_events
        
        # 데이터를 순위 기반으로 나누기 (중복 허용)
        df['ranked'] = df[col].rank(method='first')
        df['bin'] = pd.qcut(df['ranked'], 10, labels=False)  # 수치형 변수를 10개 구간으로 나눕니다.

        # 각 구간에 대한 이벤트와 비이벤트 수 계산
        grouped = df.groupby('bin')[target].agg(['count', 'sum']).reset_index()
        grouped.columns = ['bin', 'total', 'events']
        
        # 비이벤트 계산
        grouped['non_events'] = grouped['total'] - grouped['events']
        
        # WOE 및 IV 계산
        grouped['event_rate'] = grouped['events'] / total_events
        grouped['non_event_rate'] = grouped['non_events'] / total_non_events
        grouped['WOE'] = np.log(grouped['event_rate'] / grouped['non_event_rate']).replace([np.inf, -np.inf], 0)  # 무한대 처리
        grouped['IV'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['WOE']
        
        # IV 값 계산
        iv = grouped['IV'].sum()
        iv_dict[col] = iv

    return iv_dict

# 예시 데이터프레임 df와 타겟 변수 '불량판정'을 사용하여 IV 값 계산
iv_values = calculate_iv(df, '불량판정')

# 결과 출력
iv_values_df = pd.DataFrame(list(iv_values.items()), columns=['Variable', 'IV'])
iv_values_df = iv_values_df.sort_values(by='IV', ascending=False)


## p.26 최적 구간 설정을 위한 표
def 구간별_불량_통계(df, column):
    # 최소값과 최대값 기준으로 10개 구간 정의
    min_value = df[column].min()
    max_value = df[column].max()
    bins = np.linspace(min_value, max_value, num=11)  # 10개 구간을 나누기 위해 11개의 경계 값 생성
    
    # 수치형 변수를 10개 구간으로 나누기
    df['구간'] = pd.cut(df[column], bins=bins, include_lowest=True)

    # 각 구간에 대한 통계 정보 계산
    grouped = df.groupby('구간').apply(lambda x: pd.Series({
        '데이터 수': len(x),
        '불량 갯수 (1)': (x['불량판정'] == 1).sum()
    })).reset_index()

    # 불량률 계산
    grouped['불량률'] = grouped['불량 갯수 (1)'] / grouped['데이터 수']

    # 수치 범위 추가
    grouped['수치 범위'] = grouped['구간'].astype(str)

    # 최종 결과 정리
    결과 = grouped[['수치 범위', '데이터 수', '불량 갯수 (1)', '불량률']]

    return 결과

result1 = 구간별_불량_통계(df, '주조압력')
result2 = 구간별_불량_통계(df, '상금형온도2')
result3 = 구간별_불량_통계(df, '하금형온도2')









# streamlit ==============================================================================================================
st.set_page_config(layout="wide")
with st.sidebar:
  selected = option_menu(
    menu_title = "목차",
    options = ["HOME","비즈니스 배경","EDA & 전처리","공정 최적화","결론","추후 보완점"],
    icons = ["house","building","bar-chart","diagram-3","list-ul","plus-square"],
    menu_icon = "list-ol",
    default_index = 0,
  )
  
if selected == "HOME":
  st.title(f"Project4: 다이캐스팅⚒️")
  st.markdown('### 금속 주조 공정 최적화')
  st.image("img/{20B096A4-8A4A-4417-953D-7114BEE24928}.png",width=700)

if selected == "비즈니스 배경":
  st.title(f"💡{selected}")
  st.markdown('### ✅다이캐스팅')
  st.markdown('#### 📌 금속을 고온에서 녹여 고압으로 금형에 주입해 정말한 금속 부품을 대량 생산하는 주조 방식')
  file_ = open("img/diecasting.gif", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
  )
  st.markdown("")
  st.markdown('### **✅다이캐스팅의 장점**')
  st.markdown('#### 1. 복잡한 형상의 제품을 대량 생산 가능')
  st.markdown('#### 2. 제품의 정밀도와 강도가 높음')
  st.markdown('#### 3. 생산 속도가 빠르며 경량화된 부품 제조에 적합 ')
  st.markdown('')
  st.markdown('### **✅다이캐스팅의 주의점**')
  st.markdown('#### 1. 주조압력과 온도의 미세한 변화로 불량률이 높아질 수 있음')
  st.markdown('#### 2. 금형에 과도한 압력이 가해지면 수명 단축이 발생할 수 있음')

if selected == "EDA & 전처리":
  st.title(f"💡{selected}")
  st.markdown('### ✅ 사탕신호 column 확인')
  st.markdown('#### 📌 사탕신호: 새로운 금형을 테스트 하거나 주조 조건을 최적화하기 위해 초기 시도 주조할 때 해당 초기 시도가 시작되었음을 나타내는 신호')
  tryshot_img = Image.open('img/tryshot_ratio.png')
  target_ratio_img = Image.open('img/target_ratio.png')
  col1, col2 = st.columns(2)
  with col1:
    st.image(tryshot_img, caption='Tryshot Ratio')
  with col2:
    st.image(target_ratio_img, caption='Target Ratio')

  st.markdown('#### 📌 사탕신호 "D"로 측정될 때, 해당 생산품은 전부 불량판정') 
  st.markdown('#### 📌 사탕신호가 "D"인 행 드롭') 
  st.write(df)
  st.write('')
  st.markdown('### ✅ 수치형 변수 boxplot')
  boxplot_img = Image.open('img/boxplot.png')
  st.image(boxplot_img)
  st.markdown('#### 📌 주조압력, 상금형 온도1, 하금형온도1, 하금형온도2의 불량판정에 따른 분포가 상이한 것을 확인할 수 있음') 

  st.write('')
  st.markdown('### ✅ 불량판정평균')
  col1, col2 = st.columns(2)
  with col1:
    st.image("img/{EEF9F063-C81C-44B9-91B6-F023C88BC6D5}.png",width=700)
    st.markdown('> #### 📌**1월 2일, 1월 27일, 2월 12일**의 불량판정평균이 높음')
  with col2:
    tab1, tab2 = st.tabs(["수치형 변수들과 불량률 (1/27)", "범주형 변수들과 불량률 (1/27)"]) 
    with tab1:
      st.image("img/{87382D13-2964-4DAC-A446-250AA71DE96D}.png",width=700)
      st.image("img/{87D77A85-D3C5-4E22-B533-883CB5F9216B}.png",width=700)
    with tab2:
      st.image("img/{ED6B09A6-5022-4F34-B742-7483126B77F5}.png",width=700)
  st.write("")

  st.markdown('### ✅ 선형 보간')
  st.markdown('> #### 📌 **양 끝점**이 주어졌을 때, **끝점을 연결하는 직선**을 그어 결측값을 채우는 방법')
  st.image("img/{F49A2D8B-5080-4494-A692-DEADBF377074}.png",width=1000)

  st.write("")
  col1, col2 = st.columns(2)
  with col1:
    st.markdown('### ✅ Box Plot')
    st.image("img/{CC464941-FD80-4FDB-8889-4459DAA5703B}.png",width=700)
    st.markdown('#### 📌 설비작동사이클시간, 제품생산사이클시간, 저속구간속도, 상금형온도1, 상금형온도2, 하금형온도3, 형체력, 냉각수온도 -> 이상치 제거')
  with col2:
    st.markdown('### ✅ Heat Map')
    st.image("img/{90C9952A-8904-438D-B7EF-F97AC26361CE}.png",width=700)

if selected == "공정 최적화":
  st.title(f"💡 {selected}")
  st.markdown('### ✅ Workflow')
  st.image("img/{DC82AEC1-6BE3-4E7F-9BFC-6C9C95DE49A9}.png", caption="Workflow", width=800)
  st.write("")

  tab_iv, tab_dct = st.tabs(["IV 공정최적화", "결정트리 공정최적화"]) 
  with tab_iv:
    st.markdown('### ✅ IV(Information Value)')
    st.image("img/{C553A21B-6BA7-497C-97EB-4D03E7D1D717}.png", caption="IV", width=500)
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
      st.markdown('### ✅ IV값으로 최적화 변수 선정')
      tab1, tab2 = st.tabs(["주조압력에 대한 IV", "상위 3개 변수 IV"]) 
      with tab1:
          st.write(result)
      with tab2:
          st.write(iv_values_df.iloc[2:5])
    with col2:
      st.markdown('### ✅ 불량률로 최적 구간 설정')
      tab1, tab2, tab3 = st.tabs(["주조압력 최적 구간 설정", "상금형온도2 최적 구간 설정", "하금형온도2 최적 구간 설정"]) 
      with tab1:
        st.write(result1)
        st.markdown('- 주조압력 최적 구간')
        st.write(result1.iloc[8:10])
      with tab2:
        st.write(result2)
        st.markdown('- 상금형온도2 최적 구간')
        st.write(result2.iloc[3:6])
      with tab3:
        st.write(result3)
        st.markdown('- 하금형온도2 최적 구간')
        st.write(result1.iloc[2:6])
    st.markdown('### ✅ 최적화 결과')
    tab1, tab2 = st.tabs(["조건", "불량률"]) 
    with tab1:
      st.image("img/{DBE16B28-3FDD-4785-BB57-CEF2A18DE9F8}.png", width=1000)
    with tab2:
      st.image("img/{306D7A37-60ED-4504-A1D0-A2A28A080549}.png", width=800)

  with tab_dct:
    st.markdown('### ✅ 결정트리(Decision Tree)')
    st.image("img/{F4AC51C1-6CE7-4980-973E-6031ED4B685A}.png", width=800)
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
      st.markdown('### ✅ 결정트리로 최적화')
      tab1, tab2, tab3 = st.tabs(["단계 1", "단계 2", "단계 3"]) 
      with tab1:
        st.image("img/{6015E8D0-18DB-4545-8FB7-8738D4B919E2}.png", width=700)
      with tab2:
        st.image("img/{279FA3DE-BF91-4DBD-96D4-1C5C73A98632}.png", width=700)
      with tab3:
        st.image("img/{506FD252-510D-4137-9605-A6DE4129F393}.png", width=700)
    with col2:
      st.markdown('### ✅ 중요변수')
      st.image("img/{89F1CB95-4432-4EB7-A204-ACBA4668F3C0}.png", width=700)
    st.write("")
    st.markdown('### ✅ 최적화 결과')
    tab1, tab2 = st.tabs(["조건", "불량률"]) 
    with tab1:
      st.image("img/{3F8D75B3-21F5-4819-8258-3C4790D2C510}.png", width=1000)
    with tab2:
      st.image("img/{2ACF50BF-EF31-4877-AFBD-C9391FC10B4B}.png", width=800)


if selected == "결론":
  st.title(f"💡 {selected}")
  st.markdown('### ✅ IV최적화와 결정트리 최적화 비교')
  st.image("img/{4A300F8D-4BAE-4198-AD2F-6AB86DA578C4}.png", width=1000)
  st.write("")
  st.markdown('### ✅ 탕경 주조 불량')
  st.image("img/{62CBF4BC-9725-4F24-85A5-43301664B86C}.png", width=800)

if selected == "추후 보완점":
  st.title(f"💡 {selected}")
  st.markdown('### ✅ 주조압력 변수 통제')
  st.image("img/{D1012FC6-DDC7-4449-B88F-016F849A1E0F}.png", width=1000)
  st.write("")
  st.markdown('### ✅ 기타')
  st.image("img/{B65D6D59-2D98-4992-813F-87391B973EE2}.png", width=1000)
  


