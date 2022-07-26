---
layout: single
title:  "Sentimental_analysis"
categories: ML
tag: [python, blog, jekyll, matplotlib, seaborn, ML]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

**[Notice]** [ML_practical practice_4]
{: .notice--info}

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## 1) Library & Data Import



```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
```


```python
df = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/tripadviser_review.csv")
```


```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>여행에 집중할수 있게 편안한 휴식을 제공하는 호텔이었습니다. 위치선정 또한 적당한 ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>2일 이상 연박시 침대, 이불, 베게등 침구류 교체 및 어메니티 보강이 필요해 보입...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>지인에소개로온 호텔  깨끗하고 좋은거같아요 처음에는 없는게 많아 많이  당황했는데 ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>방에 딱 들어서자마자 눈이 휘둥그레질정도로 이렇게 넓은 호텔 처음 와본 것 같아요!...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>저녁에 맥주한잔 하는게 좋아서 렌트 안하고 뚜벅이 하기로 했는데 호텔 바로 앞에 버...</td>
    </tr>
  </tbody>
</table>
</div>


#### Feature Description

- rating : Rating score of user reviews

- text: Contents of user review evaluation


-----


## 2) Explore the dataset


### 2-1) Explore basic information



```python
df.shape
```

<pre>
(1001, 2)
</pre>

```python
df.isnull().sum()
```

<pre>
rating    0
text      0
dtype: int64
</pre>

```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1001 entries, 0 to 1000
Data columns (total 2 columns):
rating    1001 non-null int64
text      1001 non-null object
dtypes: int64(1), object(1)
memory usage: 15.8+ KB
</pre>

```python
df['text'][0]
```

<pre>
'여행에 집중할수 있게 편안한 휴식을 제공하는 호텔이었습니다. 위치선정 또한 적당한 편이었고 청소나 청결상태도 좋았습니다.'
</pre>

```python
len(df['text'].values.sum())
```

<pre>
223576
</pre>
-----


## 3) Korean text data preprocessing


### 3-1) Apply regular expression



```python
import re

def apply_regular_expression(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]') 
    result = hangul.sub('', text)
    return result
```


```python
apply_regular_expression(df['text'][0])
```

<pre>
'여행에 집중할수 있게 편안한 휴식을 제공하는 호텔이었습니다 위치선정 또한 적당한 편이었고 청소나 청결상태도 좋았습니다'
</pre>
-----


### 3-2) Korean Morphological Analysis - Noun Units


##### Noun morpheme extraction



```python
from konlpy.tag import Okt
from collections import Counter

nouns_tagger = Okt()
nouns = nouns_tagger.nouns(apply_regular_expression(df['text'][0]))
```


```python
nouns
```

<pre>
['여행', '집중', '휴식', '제공', '호텔', '위치', '선정', '또한', '청소', '청결', '상태']
</pre>

```python
nouns = nouns_tagger.nouns(apply_regular_expression("".join(df['text'].tolist())))
```


```python
counter = Counter(nouns)
counter.most_common(10)
```

<pre>
[('호텔', 803),
 ('수', 498),
 ('것', 436),
 ('방', 330),
 ('위치', 328),
 ('우리', 327),
 ('곳', 320),
 ('공항', 307),
 ('직원', 267),
 ('매우', 264)]
</pre>
##### Remove Hangul nouns



```python
available_counter = Counter({x : counter[x] for x in counter if len(x) > 1})
available_counter.most_common(10)
```

<pre>
[('호텔', 803),
 ('위치', 328),
 ('우리', 327),
 ('공항', 307),
 ('직원', 267),
 ('매우', 264),
 ('가격', 245),
 ('객실', 244),
 ('시설', 215),
 ('제주', 192)]
</pre>
-----


### 3-3) stopword dictionary



```python
# source - https://www.ranks.nl/stopwords/korean
stopwords = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt").values.tolist()
print(stopwords[:10])
```

<pre>
[['휴'], ['아이구'], ['아이쿠'], ['아이고'], ['어'], ['나'], ['우리'], ['저희'], ['따라'], ['의해']]
</pre>

```python
jeju_hotel_stopwords = ['제주', '제주도', '호텔', '리뷰', '숙소', '여행', '트립']
for word in jeju_hotel_stopwords:
    stopwords.append(word)
```

-----


### 3-4) Word Count


##### Create BoW Vector



```python
from sklearn.feature_extraction.text import CountVectorizer

def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]')
    result = hangul.sub('', text)
    tagger = Okt()
    nouns = nouns_tagger.nouns(result)
    nouns = [x for x in nouns if len(x) > 1]
    nouns = [x for x in nouns if x not in stopwords]
    return nouns

vect = CountVectorizer(tokenizer = lambda x: text_cleaning(x))
bow_vect = vect.fit_transform(df['text'].tolist())
word_list = vect.get_feature_names()
count_list = bow_vect.toarray().sum(axis=0)
```


```python
word_list
```

<pre>
['가가',
 '가게',
 '가격',
 '가격표',
 '가구',
 '가급',
 '가기',
 '가까이',
 '가끔',
 '가능',
 '가도',
 '가동',
 '가두',
 '가득',
 '가든',
 '가라',
 '가량',
 '가려움',
 '가로',
 '가면',
 '가몬',
 '가무',
 '가물',
 '가미',
 '가방',
 '가버',
 '가성',
 '가세',
 '가스레인지',
 '가스렌지',
 '가슴',
 '가시',
 '가신',
 '가야',
 '가옥',
 '가요',
 '가용',
 '가운데',
 '가을',
 '가인',
 '가장',
 '가정',
 '가정식',
 '가족',
 '가지',
 '가짓수',
 '가차',
 '가치',
 '가품',
 '각각',
 '각오',
 '각자',
 '각종',
 '각층',
 '간격',
 '간곳',
 '간다',
 '간단',
 '간만',
 '간식',
 '간이',
 '간주',
 '간직',
 '간판',
 '간혹',
 '갈껄',
 '갈비',
 '갈비탕',
 '갈수',
 '갈수록',
 '감각',
 '감동',
 '감명',
 '감사',
 '감상',
 '감소',
 '감안',
 '감자',
 '감히',
 '갑인',
 '갑자기',
 '갑작스레',
 '강남',
 '강력',
 '강아지',
 '강압',
 '강제',
 '강조',
 '강추',
 '개념',
 '개략',
 '개미',
 '개발',
 '개방',
 '개별',
 '개보',
 '개뿔',
 '개선',
 '개수대',
 '개월',
 '개인',
 '개인실',
 '개인정보',
 '개조',
 '개층',
 '객수',
 '객실',
 '갤러리',
 '갱스터',
 '거기',
 '거나',
 '거두',
 '거론',
 '거르세',
 '거름',
 '거리',
 '거린데',
 '거림',
 '거문도',
 '거미',
 '거부',
 '거실',
 '거여',
 '거울',
 '거위',
 '거의',
 '거절',
 '거주',
 '거지',
 '거참',
 '거품',
 '걱정',
 '건가',
 '건강',
 '건너',
 '건너편',
 '건물',
 '건의',
 '건조',
 '건조기',
 '건조대',
 '건축',
 '걷기',
 '걸음',
 '걸이',
 '걸즈',
 '검사',
 '검색',
 '검정색',
 '검토',
 '것임',
 '겉보기',
 '게다가',
 '게스트',
 '게스트하우스',
 '게임',
 '게재',
 '겐찮은듯',
 '겔상',
 '겨우',
 '겨울',
 '겨울철',
 '격인',
 '격하',
 '결과',
 '결론',
 '결석',
 '결재',
 '결정',
 '결제',
 '결코',
 '결함',
 '결항',
 '결혼',
 '결혼식',
 '겸비',
 '겸용',
 '겹겹',
 '경고',
 '경관',
 '경내',
 '경로',
 '경매',
 '경영',
 '경영학',
 '경우',
 '경쟁',
 '경쟁력',
 '경찰',
 '경치',
 '경험',
 '계단',
 '계란',
 '계란후라이',
 '계산',
 '계속',
 '계정',
 '계획',
 '고가',
 '고간',
 '고객',
 '고급',
 '고기',
 '고기국수',
 '고깃배',
 '고내포구',
 '고려',
 '고루',
 '고무줄',
 '고문',
 '고민',
 '고봉',
 '고분',
 '고생',
 '고속',
 '고속도로',
 '고아',
 '고양이',
 '고여',
 '고오',
 '고요',
 '고유',
 '고작',
 '고장',
 '고정',
 '고층',
 '고통',
 '고트',
 '고함',
 '고해',
 '곡부',
 '곧바로',
 '곧장',
 '골드스타',
 '골목',
 '골목길',
 '골퍼',
 '골프',
 '골프장',
 '골프텔',
 '곰팡이',
 '곱슬',
 '곳곳',
 '곳곳이',
 '곳도',
 '곳임',
 '공간',
 '공감',
 '공개',
 '공공',
 '공공장소',
 '공급',
 '공기',
 '공덕',
 '공률',
 '공물',
 '공사',
 '공시',
 '공실이',
 '공연',
 '공연장',
 '공영',
 '공용',
 '공원',
 '공유',
 '공짜',
 '공차',
 '공터',
 '공포',
 '공항',
 '과거',
 '과물',
 '과언',
 '과일',
 '과장',
 '관경',
 '관계',
 '관계자',
 '관광',
 '관광객',
 '관광명소',
 '관광지',
 '관덕정',
 '관련',
 '관리',
 '관리인',
 '관리자',
 '관리직',
 '관음사',
 '관해',
 '광경',
 '광고',
 '광천수',
 '괴체',
 '교대',
 '교수',
 '교외',
 '교욱받',
 '교육',
 '교체',
 '교통',
 '교환',
 '교회',
 '구가',
 '구경',
 '구경만',
 '구관',
 '구글',
 '구나',
 '구내',
 '구덩이',
 '구도',
 '구두',
 '구둣주걱',
 '구들장',
 '구류',
 '구만',
 '구매',
 '구멍',
 '구별',
 '구분',
 '구비',
 '구사',
 '구색',
 '구석',
 '구석구석',
 '구성',
 '구식',
 '구암',
 '구역',
 '구역질',
 '구이',
 '구입',
 '구조',
 '구축',
 '국가',
 '국내',
 '국도',
 '국립',
 '국수',
 '국적',
 '국제',
 '국제공항',
 '군더더기',
 '군데',
 '군데군데',
 '굳럭',
 '굳이',
 '굿굿',
 '굿굿굿',
 '굿앤굿',
 '굿임',
 '권내',
 '권장',
 '권한',
 '귀중',
 '규모',
 '규율',
 '규칙',
 '균형',
 '그거',
 '그것',
 '그게',
 '그냥',
 '그네',
 '그녀',
 '그다음',
 '그다지',
 '그닥',
 '그대로',
 '그동안',
 '그때',
 '그랜드',
 '그레이스',
 '그로',
 '그룹',
 '그릇',
 '그린',
 '그림',
 '극복',
 '극악',
 '근래',
 '근무',
 '근본',
 '근육통',
 '근처',
 '근해',
 '글래드',
 '글쎄',
 '금고',
 '금늘',
 '금능',
 '금릉',
 '금방',
 '금속',
 '금액',
 '금연',
 '금요일',
 '금은',
 '금지',
 '금토일',
 '급상승',
 '급속',
 '기간',
 '기계',
 '기구',
 '기기',
 '기념일',
 '기능',
 '기대',
 '기도',
 '기류',
 '기리',
 '기반',
 '기본',
 '기부',
 '기분',
 '기사',
 '기상',
 '기소',
 '기숙사',
 '기술',
 '기술자',
 '기억',
 '기업',
 '기여',
 '기용',
 '기우',
 '기입',
 '기적',
 '기전',
 '기점',
 '기존',
 '기준',
 '기지',
 '기타',
 '기프트샵',
 '기호',
 '기회',
 '기후',
 '긴장',
 '길가',
 '길림',
 '길목',
 '길이',
 '김녕',
 '김녕해변',
 '김밥',
 '김씨',
 '김치',
 '김포공항',
 '까페',
 '깜빡',
 '깜짝',
 '깨끗',
 '깨끗깔끔',
 '께빵',
 '꼭대기',
 '꽃꺽으러',
 '꽃사슴',
 '꾸러미',
 '꾸밈',
 '꿀잠',
 '끝내기',
 '끼리',
 '나기',
 '나누기',
 '나니',
 '나라',
 '나름',
 '나머지',
 '나머진',
 '나무',
 '나물',
 '나보',
 '나오니',
 '나우',
 '나은',
 '나이',
 '나이트',
 '나이프',
 '나중',
 '나탈리',
 '낙후',
 '낚시',
 '난로',
 '난리',
 '난방',
 '난입',
 '난타',
 '날수',
 '날씨',
 '날짜',
 '남녀',
 '남성',
 '남아',
 '남자',
 '남자친구',
 '남짓',
 '남쪽',
 '남편',
 '낭만',
 '내겐',
 '내내',
 '내년',
 '내부',
 '내부시',
 '내시',
 '내야',
 '내외',
 '내용',
 '내의',
 '내인',
 '내일',
 '냄비',
 '냄새',
 '냉동',
 '냉장고',
 '너븐팡',
 '넓이',
 '네스프레소',
 '네이버',
 '년대',
 '년전',
 '녔던',
 '노곤',
 '노래',
 '노래방',
 '노력',
 '노리',
 '노블레스',
 '노선',
 '노을',
 '노크',
 '노트북',
 '노화',
 '노후',
 '녹물',
 '녹음',
 '녹지',
 '논평',
 '놀러와',
 '놀수',
 '놀이',
 '놀이기구',
 '놀이터',
 '농부가',
 '농장',
 '높이',
 '놨더군',
 '누가',
 '누구',
 '누군가',
 '누락',
 '누리',
 '누울',
 '눈앞',
 '뉴타운',
 '느낌',
 '는걸',
 '늘송',
 '능리',
 '다가',
 '다그',
 '다다미',
 '다라',
 '다락방',
 '다른',
 '다른사람',
 '다리미',
 '다만',
 '다미',
 '다발',
 '다섯',
 '다소',
 '다수',
 '다시',
 '다운',
 '다음',
 '다이지',
 '다인',
 '다정',
 '다행',
 '단계',
 '단기',
 '단면',
 '단어',
 '단위',
 '단점',
 '단정',
 '단지',
 '단체',
 '달걀',
 '달걀프라이',
 '달라',
 '달러',
 '달리',
 '달성',
 '닭머르',
 '담당',
 '담배',
 '담소',
 '담요',
 '답변',
 '당구',
 '당근',
 '당나귀',
 '당분간',
 '당시',
 '당신',
 '당일',
 '당황',
 '대가',
 '대가족',
 '대고',
 '대관령',
 '대답',
 '대당',
 '대도',
 '대도시',
 '대뜸',
 '대략',
 '대로',
 '대리',
 '대명',
 '대박',
 '대부분',
 '대비',
 '대상',
 '대신',
 '대안',
 '대여',
 '대요',
 '대욕',
 '대응',
 '대의',
 '대입',
 '대적',
 '대접',
 '대정',
 '대중',
 '대중교통',
 '대처',
 '대체',
 '대충',
 '대포',
 '대표',
 '대하',
 '대한',
 '대한민국',
 '대한항공',
 '대해',
 '대행',
 '대형',
 '대화',
 '대환영',
 '댐핑할',
 '더군다나',
 '더더',
 '더러',
 '더블',
 '더블베드',
 '더욱',
 '더원',
 '덕림사',
 '덕분',
 '덕택',
 '던데',
 '덮어놓고',
 '데리',
 '데스크',
 '데스크톱',
 '데이',
 '데이즈',
 '델문',
 '도구',
 '도달',
 '도대체',
 '도도',
 '도둑',
 '도로',
 '도록',
 '도리어',
 '도미',
 '도보',
 '도서관',
 '도시',
 '도시락',
 '도심',
 '도심지',
 '도어',
 '도어락',
 '도움',
 '도움말',
 '도일',
 '도정',
 '도중',
 '도착',
 '도처',
 '도청',
 '도쿄',
 '도크',
 '독립',
 '독서',
 '독점',
 '독채',
 '돈까스',
 '돌담',
 '돌잔치',
 '동계',
 '동광양',
 '동굴',
 '동남',
 '동남아',
 '동네',
 '동도',
 '동료',
 '동문',
 '동물',
 '동물원',
 '동반',
 '동부',
 '동북',
 '동생',
 '동선',
 '동시',
 '동안',
 '동영상',
 '동의',
 '동이',
 '동인',
 '동작',
 '동전',
 '동정',
 '동쪽',
 '돼지',
 '돼지고기',
 '됏다',
 '될껀',
 '될껄',
 '두루',
 '두번째',
 '두봉',
 '두부',
 '두엄',
 '두운',
 '두툼',
 '둘러보기',
 '둘이서',
 '둘째',
 '둥근지붕',
 '뒤쪽',
 '뒤척',
 '뒷골목',
 '뒷마당',
 '뒷문',
 '뒷쪽',
 '드네',
 '드라이기',
 '드라이버',
 '드라이브',
 '드라이어',
 '드롭',
 '드릴',
 '드타',
 '드하',
 '득시',
 '듭니',
 '듯이',
 '듯해',
 '등급',
 '등대',
 '등등',
 '등반',
 '등산',
 '등정후',
 '디귿',
 '디너',
 '디럭스',
 '디봇',
 '디셈버',
 '디자이너',
 '디자인',
 '디저트',
 '디제이',
 '따라서',
 '때로는',
 '때문',
 '떡국',
 '또오',
 '또한',
 '뚜벅',
 '뜨근뜨근',
 '뜨내기',
 '라그',
 '라마',
 '라며',
 '라면',
 '라서',
 '라스베가스',
 '라우터',
 '라운지',
 '라이센스',
 '라커룸',
 '락스',
 '락심이',
 '락커',
 '락타',
 '란딩',
 '랍니',
 '랜드',
 '랜트',
 '랜트카',
 '랜트하',
 '램프',
 '러닝',
 '러브',
 '럭셔리',
 '런가',
 '렀는데',
 '렀습니',
 '렀으',
 '레노',
 '레드',
 '레벨',
 '레비',
 '레스토랑',
 '레시',
 '레오',
 '레이',
 '레이크',
 '레인지',
 '레저',
 '레프트',
 '렌즈',
 '렌탈업체',
 '렌터',
 '렌터카',
 '렌트',
 '렌트카',
 '려고',
 '려운',
 '로고',
 '로그',
 '로만',
 '로맨틱',
 '로부터',
 '로비',
 '로서',
 '로션',
 '로얄',
 '로움',
 '로컬',
 '로터리',
 '로프트',
 '롯데',
 '롯데리아',
 '롱보드',
 '루온토',
 '루트',
 '루프',
 '룸메이트',
 '룸바닥',
 '룸상태',
 '룸서비스',
 '룸안',
 '룸키',
 '룸타입',
 '를위',
 '리가',
 '리기',
 '리넨',
 '리뉴',
 '리뉴얼',
 '리더',
 '리도',
 '리모콘',
 '리베라',
 '리베로',
 '리빙룸',
 '리셉션',
 '리움',
 '리젠시',
 '리조트',
 '리지',
 '리치',
 '리트',
 '리플렛',
 '린스',
 '링잉',
 '마누카꿀',
 '마늘',
 '마다',
 '마담',
 '마당',
 '마레',
 '마련',
 '마루',
 '마리',
 '마모',
 '마무리',
 '마사지',
 '마술',
 '마스코트',
 '마스크',
 '마스터',
 '마시기',
 '마안',
 '마운트',
 '마을',
 '마음',
 '마이너스',
 '마인드',
 '마일리지',
 '마자',
 '마저',
 '마주',
 '마지막',
 '마지막여행',
 '마차',
 '마찬가지',
 '마치',
 '마침내',
 '마켓',
 '마트',
 '마틸다',
 '막걸리',
 '만끽',
 '만난',
 '만날',
 '만남',
 '만다린',
 '만두',
 '만들기',
 '만료',
 '만약',
 '만요',
 '만원',
 '만점',
 '만족',
 '만족도',
 '만천원',
 '만큼',
 '만하',
 '만해',
 '만화책',
 '말레이시아',
 '말레이시아인',
 '말로',
 '말리',
 '말씀',
 '말투',
 '말함',
 '맘스',
 '맛사지',
 '맛잇엇어',
 '맛집',
 '망각',
 '망신',
 '망치',
 '맞은편',
 '맞이',
 '매년',
 '매니',
 '매니저',
 '매달',
 '매듭',
 '매력',
 '매번',
 '매우',
 '매운탕',
 '매일',
 '매장',
 '매점',
 '매칭',
 '매트',
 '매트리스',
 '매트릭스',
 '매하',
 '맥도날드',
 '맥도널드',
 '맥주',
 '맥주잔',
 '맨발',
 '머리',
 '머리카락',
 '머신',
 '머싱',
 '먹거리',
 '먹기',
 '먹방',
 '먹이',
 '먼저',
 '먼지',
 '멀리',
 '메가박스',
 '메뉴',
 '메리',
 '메리어트',
 '메시지',
 ...]
</pre>

```python
count_list
```

<pre>
array([  4,   8, 245, ...,   1,   7,  14])
</pre>

```python
bow_vect.shape
```

<pre>
(1001, 3599)
</pre>

```python
bow_vect.toarray()
```

<pre>
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 2, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]])
</pre>

```python
bow_vect.toarray().sum(axis=0)
```

<pre>
array([  4,   8, 245, ...,   1,   7,  14])
</pre>

```python
bow_vect.toarray().sum(axis=0).shape
```

<pre>
(3599,)
</pre>

```python
word_count_dict = dict(zip(word_list, count_list))
word_count_dict
```

<pre>
{'가가': 4,
 '가게': 8,
 '가격': 245,
 '가격표': 1,
 '가구': 8,
 '가급': 1,
 '가기': 20,
 '가까이': 20,
 '가끔': 5,
 '가능': 10,
 '가도': 7,
 '가동': 2,
 '가두': 1,
 '가득': 2,
 '가든': 1,
 '가라': 3,
 '가량': 1,
 '가려움': 1,
 '가로': 2,
 '가면': 14,
 '가몬': 1,
 '가무': 1,
 '가물': 1,
 '가미': 1,
 '가방': 4,
 '가버': 1,
 '가성': 49,
 '가세': 3,
 '가스레인지': 1,
 '가스렌지': 1,
 '가슴': 1,
 '가시': 4,
 '가신': 3,
 '가야': 10,
 '가옥': 1,
 '가요': 5,
 '가용': 1,
 '가운데': 3,
 '가을': 4,
 '가인': 1,
 '가장': 42,
 '가정': 4,
 '가정식': 2,
 '가족': 94,
 '가지': 55,
 '가짓수': 3,
 '가차': 1,
 '가치': 15,
 '가품': 1,
 '각각': 7,
 '각오': 1,
 '각자': 2,
 '각종': 3,
 '각층': 1,
 '간격': 2,
 '간곳': 1,
 '간다': 4,
 '간단': 1,
 '간만': 1,
 '간식': 5,
 '간이': 3,
 '간주': 1,
 '간직': 1,
 '간판': 2,
 '간혹': 1,
 '갈껄': 1,
 '갈비': 1,
 '갈비탕': 1,
 '갈수': 7,
 '갈수록': 1,
 '감각': 1,
 '감동': 12,
 '감명': 1,
 '감사': 6,
 '감상': 3,
 '감소': 1,
 '감안': 5,
 '감자': 1,
 '감히': 1,
 '갑인': 1,
 '갑자기': 4,
 '갑작스레': 1,
 '강남': 1,
 '강력': 9,
 '강아지': 7,
 '강압': 2,
 '강제': 1,
 '강조': 1,
 '강추': 8,
 '개념': 1,
 '개략': 1,
 '개미': 1,
 '개발': 3,
 '개방': 2,
 '개별': 3,
 '개보': 1,
 '개뿔': 1,
 '개선': 4,
 '개수대': 1,
 '개월': 1,
 '개인': 23,
 '개인실': 1,
 '개인정보': 2,
 '개조': 5,
 '개층': 1,
 '객수': 1,
 '객실': 244,
 '갤러리': 2,
 '갱스터': 1,
 '거기': 24,
 '거나': 6,
 '거두': 1,
 '거론': 1,
 '거르세': 1,
 '거름': 2,
 '거리': 156,
 '거린데': 1,
 '거림': 1,
 '거문도': 1,
 '거미': 1,
 '거부': 4,
 '거실': 29,
 '거여': 1,
 '거울': 5,
 '거위': 1,
 '거의': 27,
 '거절': 3,
 '거주': 1,
 '거지': 1,
 '거참': 1,
 '거품': 2,
 '걱정': 27,
 '건가': 1,
 '건강': 2,
 '건너': 8,
 '건너편': 11,
 '건물': 55,
 '건의': 1,
 '건조': 2,
 '건조기': 3,
 '건조대': 2,
 '건축': 2,
 '걷기': 2,
 '걸음': 3,
 '걸이': 2,
 '걸즈': 1,
 '검사': 1,
 '검색': 13,
 '검정색': 1,
 '검토': 3,
 '것임': 3,
 '겉보기': 2,
 '게다가': 5,
 '게스트': 25,
 '게스트하우스': 30,
 '게임': 2,
 '게재': 1,
 '겐찮은듯': 1,
 '겔상': 1,
 '겨우': 3,
 '겨울': 15,
 '겨울철': 2,
 '격인': 1,
 '격하': 1,
 '결과': 2,
 '결론': 3,
 '결석': 1,
 '결재': 2,
 '결정': 12,
 '결제': 1,
 '결코': 2,
 '결함': 1,
 '결항': 2,
 '결혼': 1,
 '결혼식': 2,
 '겸비': 1,
 '겸용': 1,
 '겹겹': 2,
 '경고': 1,
 '경관': 3,
 '경내': 1,
 '경로': 1,
 '경매': 1,
 '경영': 2,
 '경영학': 1,
 '경우': 41,
 '경쟁': 1,
 '경쟁력': 2,
 '경찰': 2,
 '경치': 17,
 '경험': 26,
 '계단': 4,
 '계란': 11,
 '계란후라이': 1,
 '계산': 2,
 '계속': 23,
 '계정': 1,
 '계획': 13,
 '고가': 1,
 '고간': 1,
 '고객': 14,
 '고급': 8,
 '고기': 8,
 '고기국수': 1,
 '고깃배': 1,
 '고내포구': 1,
 '고려': 9,
 '고루': 1,
 '고무줄': 1,
 '고문': 2,
 '고민': 9,
 '고봉': 1,
 '고분': 2,
 '고생': 1,
 '고속': 2,
 '고속도로': 2,
 '고아': 1,
 '고양이': 3,
 '고여': 1,
 '고오': 1,
 '고요': 3,
 '고유': 2,
 '고작': 1,
 '고장': 3,
 '고정': 3,
 '고층': 2,
 '고통': 1,
 '고트': 1,
 '고함': 2,
 '고해': 1,
 '곡부': 1,
 '곧바로': 2,
 '곧장': 2,
 '골드스타': 1,
 '골목': 6,
 '골목길': 2,
 '골퍼': 2,
 '골프': 9,
 '골프장': 5,
 '골프텔': 2,
 '곰팡이': 14,
 '곱슬': 1,
 '곳곳': 4,
 '곳곳이': 1,
 '곳도': 8,
 '곳임': 2,
 '공간': 73,
 '공감': 1,
 '공개': 1,
 '공공': 2,
 '공공장소': 1,
 '공급': 2,
 '공기': 8,
 '공덕': 1,
 '공률': 1,
 '공물': 1,
 '공사': 12,
 '공시': 1,
 '공실이': 1,
 '공연': 8,
 '공연장': 2,
 '공영': 1,
 '공용': 8,
 '공원': 17,
 '공유': 5,
 '공짜': 1,
 '공차': 1,
 '공터': 1,
 '공포': 1,
 '공항': 307,
 '과거': 1,
 '과물': 2,
 '과언': 1,
 '과일': 9,
 '과장': 2,
 '관경': 1,
 '관계': 3,
 '관계자': 2,
 '관광': 38,
 '관광객': 15,
 '관광명소': 4,
 '관광지': 12,
 '관덕정': 4,
 '관련': 6,
 '관리': 39,
 '관리인': 1,
 '관리자': 3,
 '관리직': 2,
 '관음사': 1,
 '관해': 5,
 '광경': 2,
 '광고': 4,
 '광천수': 1,
 '괴체': 1,
 '교대': 1,
 '교수': 1,
 '교외': 1,
 '교욱받': 1,
 '교육': 5,
 '교체': 7,
 '교통': 30,
 '교환': 2,
 '교회': 2,
 '구가': 3,
 '구경': 7,
 '구경만': 1,
 '구관': 4,
 '구글': 2,
 '구나': 2,
 '구내': 1,
 '구덩이': 1,
 '구도': 1,
 '구두': 2,
 '구둣주걱': 1,
 '구들장': 1,
 '구류': 1,
 '구만': 2,
 '구매': 14,
 '구멍': 7,
 '구별': 1,
 '구분': 3,
 '구비': 11,
 '구사': 6,
 '구색': 2,
 '구석': 2,
 '구석구석': 5,
 '구성': 7,
 '구식': 1,
 '구암': 1,
 '구역': 3,
 '구역질': 2,
 '구이': 1,
 '구입': 5,
 '구조': 12,
 '구축': 1,
 '국가': 3,
 '국내': 1,
 '국도': 1,
 '국립': 1,
 '국수': 3,
 '국적': 3,
 '국제': 11,
 '국제공항': 1,
 '군더더기': 1,
 '군데': 8,
 '군데군데': 2,
 '굳럭': 1,
 '굳이': 7,
 '굿굿': 1,
 '굿굿굿': 1,
 '굿앤굿': 1,
 '굿임': 1,
 '권내': 1,
 '권장': 5,
 '권한': 2,
 '귀중': 1,
 '규모': 12,
 '규율': 2,
 '규칙': 1,
 '균형': 1,
 '그거': 3,
 '그것': 70,
 '그게': 1,
 '그냥': 42,
 '그네': 1,
 '그녀': 20,
 '그다음': 1,
 '그다지': 4,
 '그닥': 4,
 '그대로': 11,
 '그동안': 4,
 '그때': 3,
 '그랜드': 6,
 '그레이스': 3,
 '그로': 3,
 '그룹': 9,
 '그릇': 3,
 '그린': 1,
 '그림': 4,
 '극복': 1,
 '극악': 1,
 '근래': 1,
 '근무': 4,
 '근본': 1,
 '근육통': 1,
 '근처': 164,
 '근해': 1,
 '글래드': 3,
 '글쎄': 2,
 '금고': 2,
 '금늘': 1,
 '금능': 2,
 '금릉': 1,
 '금방': 3,
 '금속': 1,
 '금액': 8,
 '금연': 6,
 '금요일': 1,
 '금은': 1,
 '금지': 1,
 '금토일': 1,
 '급상승': 1,
 '급속': 1,
 '기간': 3,
 '기계': 4,
 '기구': 2,
 '기기': 4,
 '기념일': 1,
 '기능': 4,
 '기대': 15,
 '기도': 7,
 '기류': 3,
 '기리': 1,
 '기반': 4,
 '기본': 45,
 '기부': 1,
 '기분': 29,
 '기사': 8,
 '기상': 1,
 '기소': 1,
 '기숙사': 7,
 '기술': 3,
 '기술자': 1,
 '기억': 11,
 '기업': 2,
 '기여': 1,
 '기용': 1,
 '기우': 1,
 '기입': 1,
 '기적': 1,
 '기전': 1,
 '기점': 1,
 '기존': 1,
 '기준': 4,
 '기지': 1,
 '기타': 5,
 '기프트샵': 2,
 '기호': 1,
 '기회': 11,
 '기후': 1,
 '긴장': 1,
 '길가': 4,
 '길림': 1,
 '길목': 2,
 '길이': 2,
 '김녕': 1,
 '김녕해변': 1,
 '김밥': 1,
 '김씨': 1,
 '김치': 4,
 '김포공항': 1,
 '까페': 5,
 '깜빡': 1,
 '깜짝': 3,
 '깨끗': 5,
 '깨끗깔끔': 1,
 '께빵': 1,
 '꼭대기': 2,
 '꽃꺽으러': 1,
 '꽃사슴': 1,
 '꾸러미': 1,
 '꾸밈': 1,
 '꿀잠': 2,
 '끝내기': 1,
 '끼리': 18,
 '나기': 2,
 '나누기': 6,
 '나니': 1,
 '나라': 2,
 '나름': 13,
 '나머지': 6,
 '나머진': 1,
 '나무': 13,
 '나물': 1,
 '나보': 1,
 '나오니': 2,
 '나우': 1,
 '나은': 5,
 '나이': 3,
 '나이트': 2,
 '나이프': 2,
 '나중': 8,
 '나탈리': 2,
 '낙후': 3,
 '낚시': 3,
 '난로': 3,
 '난리': 3,
 '난방': 30,
 '난입': 2,
 '난타': 9,
 '날수': 1,
 '날씨': 12,
 '날짜': 1,
 '남녀': 1,
 '남성': 2,
 '남아': 5,
 '남자': 6,
 '남자친구': 2,
 '남짓': 1,
 '남쪽': 1,
 '남편': 10,
 '낭만': 2,
 '내겐': 1,
 '내내': 8,
 '내년': 1,
 '내부': 40,
 '내부시': 1,
 '내시': 1,
 '내야': 1,
 '내외': 2,
 '내용': 2,
 '내의': 2,
 '내인': 1,
 '내일': 2,
 '냄비': 1,
 '냄새': 58,
 '냉동': 1,
 '냉장고': 35,
 '너븐팡': 2,
 '넓이': 1,
 '네스프레소': 1,
 '네이버': 3,
 '년대': 2,
 '년전': 1,
 '녔던': 1,
 '노곤': 2,
 '노래': 1,
 '노래방': 3,
 '노력': 8,
 '노리': 1,
 '노블레스': 1,
 '노선': 2,
 '노을': 1,
 '노크': 1,
 '노트북': 2,
 '노화': 1,
 '노후': 6,
 '녹물': 1,
 '녹음': 4,
 '녹지': 1,
 '논평': 1,
 '놀러와': 2,
 '놀수': 1,
 '놀이': 3,
 '놀이기구': 2,
 '놀이터': 2,
 '농부가': 1,
 '농장': 3,
 '높이': 2,
 '놨더군': 1,
 '누가': 5,
 '누구': 5,
 '누군가': 4,
 '누락': 1,
 '누리': 1,
 '누울': 2,
 '눈앞': 3,
 '뉴타운': 1,
 '느낌': 49,
 '는걸': 2,
 '늘송': 3,
 '능리': 1,
 '다가': 1,
 '다그': 1,
 '다다미': 1,
 '다라': 1,
 '다락방': 1,
 '다른': 88,
 '다른사람': 1,
 '다리미': 2,
 '다만': 54,
 '다미': 1,
 '다발': 1,
 '다섯': 1,
 '다소': 21,
 '다수': 2,
 '다시': 93,
 '다운': 4,
 '다음': 102,
 '다이지': 1,
 '다인': 1,
 '다정': 2,
 '다행': 3,
 '단계': 4,
 '단기': 1,
 '단면': 1,
 '단어': 2,
 '단위': 2,
 '단점': 40,
 '단정': 1,
 '단지': 16,
 '단체': 19,
 '달걀': 3,
 '달걀프라이': 1,
 '달라': 13,
 '달러': 7,
 '달리': 6,
 '달성': 1,
 '닭머르': 1,
 '담당': 2,
 '담배': 19,
 '담소': 2,
 '담요': 1,
 '답변': 3,
 '당구': 2,
 '당근': 2,
 '당나귀': 2,
 '당분간': 1,
 '당시': 1,
 '당신': 21,
 '당일': 3,
 '당황': 7,
 '대가': 3,
 '대가족': 2,
 '대고': 1,
 '대관령': 1,
 '대답': 3,
 '대당': 1,
 '대도': 3,
 '대도시': 2,
 '대뜸': 1,
 '대략': 6,
 '대로': 8,
 '대리': 3,
 '대명': 1,
 '대박': 3,
 '대부분': 23,
 '대비': 64,
 '대상': 1,
 '대신': 8,
 '대안': 2,
 '대여': 3,
 '대요': 2,
 '대욕': 1,
 '대응': 2,
 '대의': 4,
 '대입': 1,
 '대적': 1,
 '대접': 1,
 '대정': 1,
 '대중': 9,
 '대중교통': 6,
 '대처': 2,
 '대체': 2,
 '대충': 3,
 '대포': 1,
 '대표': 4,
 '대하': 1,
 '대한': 19,
 '대한민국': 2,
 '대한항공': 1,
 '대해': 21,
 '대행': 1,
 '대형': 10,
 '대화': 11,
 '대환영': 1,
 '댐핑할': 1,
 '더군다나': 1,
 '더더': 2,
 '더러': 1,
 '더블': 29,
 '더블베드': 4,
 '더욱': 5,
 '더원': 1,
 '덕림사': 1,
 '덕분': 6,
 '덕택': 3,
 '던데': 1,
 '덮어놓고': 1,
 '데리': 5,
 '데스크': 30,
 '데스크톱': 1,
 '데이': 1,
 '데이즈': 1,
 '델문': 2,
 '도구': 18,
 '도달': 3,
 '도대체': 1,
 '도도': 1,
 '도둑': 1,
 '도로': 41,
 '도록': 1,
 '도리어': 1,
 '도미': 9,
 '도보': 35,
 '도서관': 1,
 '도시': 18,
 '도시락': 4,
 '도심': 14,
 '도심지': 1,
 '도어': 3,
 '도어락': 1,
 '도움': 51,
 '도움말': 1,
 '도일': 1,
 '도정': 1,
 '도중': 2,
 '도착': 69,
 '도처': 1,
 '도청': 2,
 '도쿄': 1,
 '도크': 1,
 '독립': 6,
 '독서': 1,
 '독점': 1,
 '독채': 5,
 '돈까스': 1,
 '돌담': 1,
 '돌잔치': 1,
 '동계': 1,
 '동광양': 1,
 '동굴': 1,
 '동남': 1,
 '동남아': 2,
 '동네': 7,
 '동도': 1,
 '동료': 2,
 '동문': 14,
 '동물': 9,
 '동물원': 2,
 '동반': 3,
 '동부': 2,
 '동북': 1,
 '동생': 3,
 '동선': 3,
 '동시': 7,
 '동안': 48,
 '동영상': 1,
 '동의': 3,
 '동이': 1,
 '동인': 2,
 '동작': 1,
 '동전': 1,
 '동정': 1,
 '동쪽': 5,
 '돼지': 16,
 '돼지고기': 4,
 '됏다': 1,
 '될껀': 1,
 '될껄': 1,
 '두루': 2,
 '두번째': 2,
 '두봉': 2,
 '두부': 1,
 '두엄': 1,
 '두운': 2,
 '두툼': 1,
 '둘러보기': 1,
 '둘이서': 3,
 '둘째': 5,
 '둥근지붕': 1,
 '뒤쪽': 4,
 '뒤척': 1,
 '뒷골목': 1,
 '뒷마당': 1,
 '뒷문': 1,
 '뒷쪽': 2,
 '드네': 1,
 '드라이기': 7,
 '드라이버': 1,
 '드라이브': 11,
 '드라이어': 11,
 '드롭': 1,
 '드릴': 1,
 '드타': 1,
 '드하': 2,
 '득시': 1,
 '듭니': 5,
 '듯이': 1,
 '듯해': 1,
 '등급': 3,
 '등대': 3,
 '등등': 8,
 '등반': 3,
 '등산': 6,
 '등정후': 1,
 '디귿': 1,
 '디너': 4,
 '디럭스': 6,
 '디봇': 1,
 '디셈버': 2,
 '디자이너': 1,
 '디자인': 11,
 '디저트': 1,
 '디제이': 2,
 '따라서': 4,
 '때로는': 1,
 '때문': 112,
 '떡국': 2,
 '또오': 1,
 '또한': 76,
 '뚜벅': 3,
 '뜨근뜨근': 1,
 '뜨내기': 1,
 '라그': 1,
 '라마': 4,
 '라며': 3,
 '라면': 15,
 '라서': 1,
 '라스베가스': 1,
 '라우터': 1,
 '라운지': 9,
 '라이센스': 1,
 '라커룸': 1,
 '락스': 2,
 '락심이': 1,
 '락커': 2,
 '락타': 1,
 '란딩': 1,
 '랍니': 1,
 '랜드': 1,
 '랜트': 1,
 '랜트카': 1,
 '랜트하': 1,
 '램프': 2,
 '러닝': 1,
 '러브': 3,
 '럭셔리': 5,
 '런가': 2,
 '렀는데': 1,
 '렀습니': 2,
 '렀으': 1,
 '레노': 1,
 '레드': 1,
 '레벨': 1,
 '레비': 1,
 '레스토랑': 64,
 '레시': 1,
 '레오': 2,
 '레이': 1,
 '레이크': 1,
 '레인지': 3,
 '레저': 1,
 '레프트': 1,
 '렌즈': 1,
 '렌탈업체': 1,
 '렌터': 1,
 '렌터카': 4,
 '렌트': 17,
 '렌트카': 8,
 '려고': 4,
 '려운': 1,
 '로고': 1,
 '로그': 3,
 '로만': 1,
 '로맨틱': 2,
 '로부터': 2,
 '로비': 49,
 '로서': 2,
 '로션': 1,
 '로얄': 1,
 '로움': 1,
 '로컬': 3,
 '로터리': 1,
 '로프트': 1,
 '롯데': 6,
 '롯데리아': 2,
 '롱보드': 1,
 '루온토': 1,
 '루트': 1,
 '루프': 17,
 '룸메이트': 1,
 '룸바닥': 1,
 '룸상태': 2,
 '룸서비스': 9,
 '룸안': 1,
 '룸키': 2,
 '룸타입': 1,
 '를위': 1,
 '리가': 2,
 '리기': 1,
 '리넨': 1,
 '리뉴': 1,
 '리뉴얼': 1,
 '리더': 1,
 '리도': 1,
 '리모콘': 3,
 '리베라': 2,
 '리베로': 1,
 '리빙룸': 2,
 '리셉션': 29,
 '리움': 2,
 '리젠시': 1,
 '리조트': 53,
 '리지': 1,
 '리치': 1,
 '리트': 1,
 '리플렛': 1,
 '린스': 2,
 '링잉': 1,
 '마누카꿀': 1,
 '마늘': 1,
 '마다': 1,
 '마담': 2,
 '마당': 2,
 '마레': 2,
 '마련': 7,
 '마루': 5,
 '마리': 11,
 '마모': 1,
 '마무리': 3,
 '마사지': 4,
 '마술': 1,
 '마스코트': 2,
 '마스크': 1,
 '마스터': 2,
 '마시기': 2,
 '마안': 1,
 '마운트': 1,
 '마을': 9,
 '마음': 31,
 '마이너스': 1,
 '마인드': 4,
 '마일리지': 2,
 '마자': 2,
 '마저': 1,
 '마주': 4,
 '마지막': 21,
 '마지막여행': 1,
 '마차': 1,
 '마찬가지': 4,
 '마치': 12,
 '마침내': 3,
 '마켓': 9,
 '마트': 14,
 '마틸다': 2,
 '막걸리': 1,
 '만끽': 1,
 '만난': 1,
 '만날': 1,
 '만남': 1,
 '만다린': 2,
 '만두': 1,
 '만들기': 1,
 '만료': 1,
 '만약': 6,
 '만요': 1,
 '만원': 20,
 '만점': 1,
 '만족': 12,
 '만족도': 1,
 '만천원': 1,
 '만큼': 2,
 '만하': 2,
 '만해': 2,
 '만화책': 1,
 '말레이시아': 1,
 '말레이시아인': 1,
 '말로': 2,
 '말리': 1,
 '말씀': 7,
 '말투': 3,
 '말함': 2,
 '맘스': 1,
 '맛사지': 1,
 '맛잇엇어': 1,
 '맛집': 25,
 '망각': 1,
 '망신': 2,
 '망치': 2,
 '맞은편': 7,
 '맞이': 5,
 '매년': 2,
 '매니': 1,
 '매니저': 3,
 '매달': 1,
 '매듭': 1,
 '매력': 5,
 '매번': 1,
 '매우': 265,
 '매운탕': 1,
 '매일': 36,
 '매장': 3,
 '매점': 3,
 '매칭': 1,
 '매트': 5,
 '매트리스': 13,
 '매트릭스': 1,
 '매하': 1,
 '맥도날드': 5,
 '맥도널드': 1,
 '맥주': 22,
 '맥주잔': 1,
 '맨발': 3,
 '머리': 7,
 '머리카락': 4,
 '머신': 3,
 '머싱': 1,
 '먹거리': 7,
 '먹기': 2,
 '먹방': 1,
 '먹이': 3,
 '먼저': 3,
 '먼지': 3,
 '멀리': 14,
 '메가박스': 1,
 '메뉴': 15,
 '메리': 1,
 '메리어트': 1,
 '메시지': 1,
 ...}
</pre>
-----


### 3-5) Apply TF-IDF


##### TF-IDF coversion 



```python
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_vectorizer = TfidfTransformer()
tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)
```


```python
print(tf_idf_vect.shape)
print(tf_idf_vect[0])
```

<pre>
(1001, 3599)
  (0, 3588)	0.35673213299026796
  (0, 2927)	0.2582351368959594
  (0, 2925)	0.320251680858207
  (0, 2866)	0.48843555212083145
  (0, 2696)	0.23004450213863206
  (0, 2311)	0.15421663035331626
  (0, 1584)	0.48843555212083145
  (0, 1527)	0.2928089229786031
  (0, 790)	0.2528176728459411
</pre>
##### Vector: word mapping



```python
invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
print(str(invert_index_vectorizer)[:100]+'..')
```

<pre>
{2866: '집중', 3588: '휴식', 2696: '제공', 2311: '위치', 1584: '선정', 790: '또한', 2927: '청소', 2925: '청결', 1527..
</pre>
-----


## 4) Logistic Regression Classification


### 4-1) Create dataset


##### Converting Rating data to binary



```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>여행에 집중할수 있게 편안한 휴식을 제공하는 호텔이었습니다. 위치선정 또한 적당한 ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>2일 이상 연박시 침대, 이불, 베게등 침구류 교체 및 어메니티 보강이 필요해 보입...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>지인에소개로온 호텔  깨끗하고 좋은거같아요 처음에는 없는게 많아 많이  당황했는데 ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>방에 딱 들어서자마자 눈이 휘둥그레질정도로 이렇게 넓은 호텔 처음 와본 것 같아요!...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>저녁에 맥주한잔 하는게 좋아서 렌트 안하고 뚜벅이 하기로 했는데 호텔 바로 앞에 버...</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.rating.hist()
```

<pre>
<matplotlib.axes._subplots.AxesSubplot at 0x7fc1c0d616a0>
</pre>
<img src="data:image/png;base64,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"/>


```python
def rating_to_label(rating):
    if rating > 3:
        return 1
    else:
        return 0

df['y'] = df['rating'].apply(lambda x: rating_to_label(x))
```


```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>text</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>여행에 집중할수 있게 편안한 휴식을 제공하는 호텔이었습니다. 위치선정 또한 적당한 ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>2일 이상 연박시 침대, 이불, 베게등 침구류 교체 및 어메니티 보강이 필요해 보입...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>지인에소개로온 호텔  깨끗하고 좋은거같아요 처음에는 없는게 많아 많이  당황했는데 ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>방에 딱 들어서자마자 눈이 휘둥그레질정도로 이렇게 넓은 호텔 처음 와본 것 같아요!...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>저녁에 맥주한잔 하는게 좋아서 렌트 안하고 뚜벅이 하기로 했는데 호텔 바로 앞에 버...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.y.value_counts()
```

<pre>
1    726
0    275
Name: y, dtype: int64
</pre>
-----




### 4-2) Dataset Separation



```python
from sklearn.model_selection import train_test_split

y = df['y']
x_train, x_test, y_train, y_test = train_test_split(tf_idf_vect, y, test_size=0.30)
```


```python
print(x_train.shape)
print(x_test.shape)
```

<pre>
(700, 3599)
(301, 3599)
</pre>
-----


### 4-3) model training


##### Logistic Regression training



```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train LR model
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)

# classifiacation predict
y_pred = lr.predict(x_test)
```

##### evaluation



```python
# classification result for test dataset
print("accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision : %.3f" % precision_score(y_test, y_pred))
print("Recall : %.3f" % recall_score(y_test, y_pred))
print("F1 : %.3f" % f1_score(y_test, y_pred))
```

<pre>
accuracy: 0.72
Precision : 0.718
Recall : 1.000
F1 : 0.836
</pre>

```python
from sklearn.metrics import confusion_matrix

# print confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
```

<pre>
[[  3  84]
 [  0 214]]
</pre>
-----


### 4-4) Re-sampling


##### 1:1 Sampling



```python
positive_random_idx = df[df['y']==1].sample(275, random_state=33).index.tolist()
negative_random_idx = df[df['y']==0].sample(275, random_state=33).index.tolist()
```


```python
# dataset split to train/test
random_idx = positive_random_idx + negative_random_idx
X = tf_idf_vect[random_idx]
y = df['y'][random_idx]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
```


```python
print(x_train.shape)
print(x_test.shape)
```

<pre>
(412, 3599)
(138, 3599)
</pre>
##### model retraining



```python
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
```

##### evaluation



```python
print("accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision : %.3f" % precision_score(y_test, y_pred))
print("Recall : %.3f" % recall_score(y_test, y_pred))
print("F1 : %.3f" % f1_score(y_test, y_pred))
```

<pre>
accuracy: 0.72
Precision : 0.644
Recall : 0.797
F1 : 0.712
</pre>

```python
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
```

<pre>
[[53 26]
 [12 47]]
</pre>
-----


## 5) Positive/negative keyword analysis


##### Coef Analysis of Logistic Regression Models



```python
# print logistic regression's coef
plt.rcParams['figure.figsize'] = [10, 8]
plt.bar(range(len(lr.coef_[0])), lr.coef_[0])
```

<pre>
<BarContainer object of 3599 artists>
</pre>
<img src="data:image/png;base64,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"/>

##### Positive/negative keyword output



```python
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)[:5])
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)[-5:])
```

<pre>
[(1.3321308087111168, 2400), (1.1098677278465363, 2977), (1.029120247844704, 1247), (0.9474432432978868, 2957), (0.9049132254229898, 26)]
[(-0.6491883332225628, 363), (-0.6683241824194205, 3538), (-0.6811855513119685, 1909), (-0.9632209931825515, 1293), (-1.124500886987929, 515)]
</pre>

```python
coef_pos_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)
coef_neg_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=False)
coef_pos_index
```

<pre>
[(1.3321308087111168, 2400),
 (1.1098677278465363, 2977),
 (1.029120247844704, 1247),
 (0.9474432432978868, 2957),
 (0.9049132254229898, 26),
 (0.8631251640260484, 385),
 (0.8624237330200107, 2730),
 (0.7848182816732695, 578),
 (0.732990219026413, 2311),
 (0.716865493140725, 246),
 (0.7161355390234533, 1809),
 (0.7134163462461057, 956),
 (0.7044600617626677, 115),
 (0.6869152801231841, 1384),
 (0.6556108465327279, 1148),
 (0.6279495890384094, 2849),
 (0.6222266165132151, 2779),
 (0.6161464320403829, 883),
 (0.5993549427526994, 1491),
 (0.5957963623120057, 2680),
 (0.5486926383676386, 2834),
 (0.5396380473836403, 660),
 (0.5293505033175993, 416),
 (0.5268251635528765, 680),
 (0.5162996339456437, 3447),
 (0.515397298688398, 2781),
 (0.5102891400815143, 790),
 (0.5001491197806007, 3428),
 (0.4929472812035707, 1816),
 (0.49144456248364404, 692),
 (0.48756380000669713, 131),
 (0.48684210942709405, 1159),
 (0.45674587837763275, 1217),
 (0.45540907417568605, 1853),
 (0.4539368892907782, 981),
 (0.45364743248610284, 1799),
 (0.4357968956624657, 2988),
 (0.4317750704888898, 2606),
 (0.43044019060177896, 2771),
 (0.42976241036062235, 2455),
 (0.42955255071710025, 1981),
 (0.42879073804182866, 2722),
 (0.4199707273896906, 1779),
 (0.4159154526928186, 19),
 (0.4084324785499222, 2780),
 (0.4081209617829656, 1067),
 (0.4042034857154998, 2683),
 (0.4040469138707011, 3152),
 (0.4004256014611016, 1028),
 (0.39815159541886946, 826),
 (0.3846458902898351, 588),
 (0.38147465088928206, 627),
 (0.37718502893618444, 154),
 (0.3738446974828173, 2351),
 (0.367332320736131, 2678),
 (0.3671732270029874, 3016),
 (0.35993951750388575, 0),
 (0.35967370276499855, 1885),
 (0.34265833016423075, 1826),
 (0.33745159616067577, 2385),
 (0.3371647391165404, 3598),
 (0.335633532472175, 136),
 (0.33354095363617825, 1695),
 (0.325633618317818, 665),
 (0.3224469848677877, 1085),
 (0.3194837275959594, 73),
 (0.3180284758263984, 2685),
 (0.3165087876097102, 1554),
 (0.31480819925031794, 1350),
 (0.31431164320253163, 2225),
 (0.31025312484733686, 43),
 (0.308334722553892, 1263),
 (0.3029159220243568, 3588),
 (0.3020625524276556, 2726),
 (0.30156731773121254, 121),
 (0.2949888109951585, 1671),
 (0.29406267557940335, 282),
 (0.29388674541634185, 3320),
 (0.29319214345781675, 2167),
 (0.29257136559515806, 2),
 (0.2903527229951154, 582),
 (0.2893656045575609, 979),
 (0.2865105037602163, 341),
 (0.28357844878803906, 3347),
 (0.28264031635232884, 658),
 (0.28060836686814006, 1935),
 (0.27970378259167916, 1700),
 (0.2786399841890623, 3201),
 (0.27703202112305, 625),
 (0.27668100652947636, 3029),
 (0.2743375997046002, 1929),
 (0.27413594760437765, 3547),
 (0.27368204754138026, 1926),
 (0.27183018519065966, 1165),
 (0.27137529927855913, 71),
 (0.2685065484060374, 3013),
 (0.2683925002732845, 1078),
 (0.26602345854602616, 960),
 (0.26563651641305047, 2674),
 (0.2655828802203175, 395),
 (0.2637574460673181, 1659),
 (0.2630924155233454, 3425),
 (0.262576984334013, 1205),
 (0.2612445245166083, 1750),
 (0.25955244207216993, 3404),
 (0.2582453145513143, 860),
 (0.2573369451118894, 2038),
 (0.2528728295737944, 3267),
 (0.25117183576365226, 3150),
 (0.2502831420241506, 723),
 (0.24956513735466768, 2075),
 (0.24845709898575633, 1025),
 (0.24565832211918145, 1374),
 (0.2444466387148004, 3238),
 (0.24345854902783198, 714),
 (0.24263088287913587, 3353),
 (0.23959737325461491, 3060),
 (0.2394235215359124, 1453),
 (0.2382302727044769, 2595),
 (0.236693438753923, 3537),
 (0.2352318898029652, 726),
 (0.23441224495353918, 100),
 (0.23406755548226849, 2551),
 (0.23379140860743455, 2566),
 (0.23346109278685567, 532),
 (0.22895808194923195, 3337),
 (0.22880986570599313, 353),
 (0.22841556466198995, 1383),
 (0.22709775600161025, 2691),
 (0.22652789077304955, 2436),
 (0.22627009793868919, 2753),
 (0.22507189125351756, 110),
 (0.22396460652995273, 147),
 (0.22259697920818705, 1482),
 (0.22193733214129846, 2438),
 (0.22175178779418242, 1552),
 (0.22151447958091616, 1199),
 (0.22146319716752702, 544),
 (0.22127225625738847, 3272),
 (0.220781024296084, 2274),
 (0.21982768282476364, 2705),
 (0.21936837987103552, 718),
 (0.2193496947270397, 3299),
 (0.2193496947270397, 172),
 (0.2184497853928657, 717),
 (0.21833407390702603, 1505),
 (0.21809621397373719, 3512),
 (0.21781685606599935, 322),
 (0.2176002036083807, 2913),
 (0.21692615631477355, 3551),
 (0.21671313079089544, 3102),
 (0.21671313079089544, 1541),
 (0.21606244261917687, 3014),
 (0.21562177834082913, 3127),
 (0.21556533928126198, 2677),
 (0.2152860596906335, 769),
 (0.2138168027519045, 2108),
 (0.21380867386959615, 263),
 (0.2132109653607666, 1307),
 (0.21244302599751747, 3061),
 (0.21224270498780334, 206),
 (0.21202778293074742, 192),
 (0.21201290684914603, 1949),
 (0.21145270517927034, 1762),
 (0.21114972847304267, 1455),
 (0.21083036354582643, 2186),
 (0.21051142044700794, 3384),
 (0.21001347415099242, 3286),
 (0.20998040536146637, 68),
 (0.20967616965410357, 235),
 (0.208251046375912, 1768),
 (0.20760478979685312, 3390),
 (0.20740315589887034, 3088),
 (0.20556807206875818, 3414),
 (0.20234144193644174, 3555),
 (0.2018656714571112, 2371),
 (0.2016461772393062, 51),
 (0.19929383346501198, 1721),
 (0.1990149202138041, 1010),
 (0.19896039182139422, 3432),
 (0.19894547087204073, 340),
 (0.19839781641856194, 3002),
 (0.19813175575019054, 3274),
 (0.1966951722367976, 623),
 (0.19660389823049113, 2016),
 (0.19581183146615333, 2168),
 (0.19505097638548372, 1886),
 (0.1945003983255703, 1147),
 (0.19426488558460306, 3408),
 (0.19403487720548504, 1523),
 (0.19371889849533547, 356),
 (0.19358226372345383, 1917),
 (0.19334929562068714, 3057),
 (0.19311769483840216, 2804),
 (0.19291083167167608, 2180),
 (0.19105979566738465, 3359),
 (0.19038712527647178, 40),
 (0.19025428348723153, 839),
 (0.19004592638299575, 278),
 (0.1896791185823034, 2965),
 (0.1892532697410779, 3339),
 (0.18906755452359386, 1961),
 (0.18888996868423136, 1432),
 (0.18781348017991015, 1278),
 (0.18759776633440292, 1550),
 (0.1869393811048611, 3549),
 (0.1864613695267563, 2200),
 (0.18617798946763775, 1249),
 (0.18545273365532508, 1386),
 (0.18465966170413162, 2331),
 (0.18448194605435825, 3283),
 (0.18359704376005423, 773),
 (0.18341863298488395, 2748),
 (0.18265890303695448, 1291),
 (0.18243752629350396, 3370),
 (0.18243752629350396, 663),
 (0.18242433015171347, 791),
 (0.18221035881128547, 1960),
 (0.18170817590567348, 2088),
 (0.18165649605140002, 281),
 (0.18136983329265988, 3486),
 (0.18105293360540595, 109),
 (0.1809803538967947, 3369),
 (0.17985798544663784, 2080),
 (0.17956847966930387, 3034),
 (0.17955990993913923, 3250),
 (0.17891844051459602, 2033),
 (0.17889862025954986, 2764),
 (0.17879595150453056, 1442),
 (0.17838512732829997, 1758),
 (0.17690299960811534, 1918),
 (0.1767714210124328, 801),
 (0.1767285112565217, 2141),
 (0.17659448012000462, 304),
 (0.17647989044490972, 3516),
 (0.17634068214069432, 3297),
 (0.17608186054158145, 2070),
 (0.17576526782661273, 1030),
 (0.17515502627906773, 2622),
 (0.1749737604394968, 3410),
 (0.17410214749925304, 2759),
 (0.17407257079494115, 3271),
 (0.17407257079494115, 2607),
 (0.17407257079494115, 2465),
 (0.17407257079494115, 2398),
 (0.17407257079494115, 571),
 (0.17311325469340594, 2742),
 (0.17276920148824162, 321),
 (0.17256067465778124, 1452),
 (0.17159599314441507, 903),
 (0.17103233804318696, 47),
 (0.16925315589858916, 2350),
 (0.16920767999211647, 423),
 (0.1685975878154169, 489),
 (0.1685895331301071, 440),
 (0.16749119686465136, 2042),
 (0.16702579919941596, 382),
 (0.16700101352781155, 667),
 (0.16610539591043183, 286),
 (0.16507120766930342, 2003),
 (0.16429698469550058, 1120),
 (0.16331700479491104, 2188),
 (0.16331700479491104, 2018),
 (0.16331390385034894, 2793),
 (0.16300508910149644, 916),
 (0.16282275467452148, 1843),
 (0.1624724734816524, 1359),
 (0.1618759037676759, 3395),
 (0.16130519387471293, 2975),
 (0.16025844697165065, 669),
 (0.16013102293033513, 1953),
 (0.1597482903610505, 996),
 (0.15968247106389244, 2767),
 (0.1595935294251936, 259),
 (0.15880675549174347, 1602),
 (0.1587439552243165, 935),
 (0.15869391556965143, 93),
 (0.15866592489114187, 2581),
 (0.15842096948259, 2258),
 (0.15836742047722766, 2787),
 (0.15791266057854855, 3058),
 (0.15765558962932025, 2271),
 (0.15732721123058058, 1292),
 (0.15722879615475854, 720),
 (0.15694585991437593, 3037),
 (0.1565630030203952, 1910),
 (0.15653431345464613, 748),
 (0.15647025349331029, 1485),
 (0.1560861187506581, 1451),
 (0.15579667970897626, 1348),
 (0.15540520631625526, 1326),
 (0.1549720046769624, 2047),
 (0.15452615455169144, 543),
 (0.1541983511016218, 3164),
 (0.1541983511016218, 683),
 (0.1539873238718623, 3162),
 (0.1538488627144806, 888),
 (0.1536071393576577, 3208),
 (0.15298690219214506, 1593),
 (0.15275270485680378, 2386),
 (0.15240308917035653, 991),
 (0.151621197000685, 612),
 (0.15129785599666357, 939),
 (0.15094776923710648, 2895),
 (0.15094776923710648, 2425),
 (0.14962944016742355, 2216),
 (0.14904033568007535, 290),
 (0.1486419531495219, 2794),
 (0.1485083911928923, 2940),
 (0.14824979715450434, 2013),
 (0.14749451289822033, 83),
 (0.1474087004234331, 1805),
 (0.14710629342492257, 3189),
 (0.1468532386313502, 937),
 (0.14658634943466398, 2146),
 (0.1465384816163645, 1053),
 (0.14652908216269708, 3356),
 (0.1464667635763324, 1319),
 (0.1464667635763324, 987),
 (0.146232204378051, 1709),
 (0.14596163518975866, 589),
 (0.14594613731072828, 2170),
 (0.1459333390058329, 2956),
 (0.1456575089375777, 176),
 (0.14544378809072755, 74),
 (0.1452359825672161, 393),
 (0.14490710054390635, 971),
 (0.14459417700731625, 2897),
 (0.1435072329559267, 349),
 (0.1435072329559267, 81),
 (0.14342614982099325, 237),
 (0.14334431387987334, 989),
 (0.14333871647641, 2497),
 (0.1427597861226763, 1891),
 (0.14274532098406992, 453),
 (0.14267158693294138, 2462),
 (0.14241007268476308, 2792),
 (0.14239906559751997, 2191),
 (0.14172778883838677, 2148),
 (0.14078125131967817, 283),
 (0.1406282886456704, 311),
 (0.14042112327259743, 2864),
 (0.14040331443949813, 2150),
 (0.14036503615131526, 910),
 (0.14030418343407003, 1253),
 (0.14030418343407003, 30),
 (0.1402567473881267, 2525),
 (0.1401665654116925, 179),
 (0.14011839714622298, 1796),
 (0.13995369350695977, 3125),
 (0.1390466886476543, 1530),
 (0.13874050516494102, 1938),
 (0.13855961822894952, 1440),
 (0.1385421137946354, 1239),
 (0.13825954738820223, 1614),
 (0.13823477220476124, 2708),
 (0.1377461516784561, 2411),
 (0.13747373989422332, 3366),
 (0.1374181381097756, 2475),
 (0.13638701952286028, 1),
 (0.13633128481810017, 561),
 (0.1358404957999536, 852),
 (0.13575081367693947, 961),
 (0.13573573877465953, 1106),
 (0.1357304444802513, 1740),
 (0.13570313559786057, 326),
 (0.13504425093295547, 1301),
 (0.13449235053930575, 3434),
 (0.13426801036692038, 12),
 (0.13419735575739455, 2267),
 (0.13413197224055048, 817),
 (0.1332035480663822, 2844),
 (0.13293688071865534, 843),
 (0.13293254436624813, 1781),
 (0.1327270864101471, 1616),
 (0.1327270864101471, 818),
 (0.13235034222368947, 3567),
 (0.13192012478547538, 3476),
 (0.13192012478547538, 1112),
 (0.13178963598053114, 3426),
 (0.13167562685657733, 2605),
 (0.13166726335215087, 1557),
 (0.13160015043919898, 3581),
 (0.13160015043919898, 2488),
 (0.13160015043919898, 2094),
 (0.1313547383200888, 2972),
 (0.13031729072402892, 1746),
 (0.12988487499838028, 1172),
 (0.12973246724796839, 2573),
 (0.1296971052433702, 2969),
 (0.1296971052433702, 767),
 (0.1293905727630968, 1795),
 (0.12938453024897784, 1715),
 (0.12933454503707428, 1728),
 (0.1289365379022888, 3027),
 (0.12867150979878933, 1921),
 (0.12860870968792637, 2078),
 (0.12810442768317531, 1739),
 (0.12805439958035297, 3026),
 (0.1278779808209269, 3442),
 (0.12780007554324174, 618),
 (0.12735321130110464, 3552),
 (0.12693382684487228, 1283),
 (0.12693382684487228, 774),
 (0.12681663696770337, 2618),
 (0.1267241604496154, 2421),
 (0.1267241604496154, 2166),
 (0.12564078640260584, 347),
 (0.12536956468677118, 855),
 (0.1253512794237119, 2523),
 (0.12484293651117723, 3015),
 (0.12446046957444332, 3561),
 (0.12438503356051701, 2220),
 (0.12428431698758667, 2335),
 (0.12428431698758667, 1051),
 (0.1241217444105632, 1621),
 (0.12359845049290538, 1506),
 (0.12312446033672597, 872),
 (0.1228540584502789, 375),
 (0.12218510960287418, 1321),
 (0.12212076174130922, 3121),
 (0.12191095229953881, 1092),
 (0.12191095229953881, 122),
 (0.12178744304758918, 1060),
 (0.12155319948980126, 3577),
 (0.12128424197829846, 3508),
 (0.12125712391650044, 806),
 (0.12094559865788808, 2298),
 (0.12081130224026823, 2171),
 (0.12081130224026823, 703),
 (0.12081130224026823, 379),
 (0.1206667914049753, 519),
 (0.12053482818464879, 671),
 (0.12016451781173339, 3518),
 (0.11957468357601171, 3168),
 (0.11957468357601171, 2632),
 (0.11930669990535478, 3309),
 (0.11907549449107033, 3593),
 (0.11907309519569548, 3513),
 (0.11907309519569548, 3143),
 (0.11907309519569548, 401),
 (0.11900817960475607, 271),
 (0.11899208138806625, 2869),
 (0.11895958968877159, 1539),
 (0.11880440997267704, 3435),
 (0.11849763128788413, 2546),
 (0.11809597331801082, 2911),
 (0.11809597331801082, 2862),
 (0.11806555686548152, 3262),
 (0.11806555686548152, 2036),
 (0.11755582720566768, 2824),
 (0.1158635500136846, 3412),
 (0.1158635500136846, 1690),
 (0.1158635500136846, 867),
 (0.1158635500136846, 182),
 (0.1146049615072046, 59),
 (0.11457215302554612, 758),
 (0.11452048101166765, 284),
 (0.11437158955438233, 2899),
 (0.11437158955438233, 415),
 (0.11421370258446362, 2533),
 (0.11402351764493407, 814),
 (0.11398329625091098, 400),
 (0.11398004111441176, 2062),
 (0.11388712615281078, 457),
 (0.11380483755755394, 825),
 (0.11376257728774304, 2953),
 (0.11374483910338776, 2496),
 (0.11363061333793036, 3427),
 (0.11363061333793036, 390),
 (0.11331142513554919, 217),
 (0.11323824243411867, 2848),
 (0.11269570106913031, 2158),
 (0.11260561285741831, 1553),
 (0.11170726385475563, 2594),
 (0.11146074951366217, 2765),
 (0.11143204200991574, 2303),
 (0.11121357044433934, 337),
 (0.11121357044433934, 272),
 (0.11096001854374073, 3089),
 (0.11095263793825426, 2613),
 (0.11095263793825426, 1406),
 (0.11095263793825426, 970),
 (0.11087589389709121, 2315),
 (0.11087589389709121, 425),
 (0.11072422457342575, 3558),
 (0.11072422457342575, 3419),
 (0.11072422457342575, 2608),
 (0.11072422457342575, 2598),
 (0.11060675699993265, 2426),
 (0.11022343631120256, 2457),
 (0.11015920452585921, 2051),
 (0.11015920452585921, 1725),
 (0.11015920452585921, 1684),
 (0.11015920452585921, 95),
 (0.11010581079286663, 3067),
 (0.10998171923083665, 1612),
 (0.10998171923083665, 266),
 (0.10988929327193322, 1483),
 (0.10968349322053741, 3423),
 (0.10968349322053741, 1945),
 (0.10968349322053741, 1095),
 (0.10946316414746812, 2832),
 (0.1090479208217944, 2456),
 (0.1090479208217944, 493),
 (0.1090479208217944, 146),
 (0.10885202678651067, 2971),
 (0.1085737346294867, 1073),
 (0.10811028544693015, 1991),
 (0.1079924384456164, 2085),
 (0.10794562030046964, 1743),
 (0.10773901543501041, 267),
 (0.10712091911923531, 3445),
 (0.10682046421939902, 741),
 (0.10682046421939902, 140),
 (0.10675298330383282, 159),
 (0.10618581865111706, 2165),
 (0.10618581865111706, 498),
 (0.10618581865111706, 432),
 (0.10605425200156694, 3430),
 (0.10605425200156694, 2948),
 (0.10605425200156694, 857),
 (0.10605425200156694, 346),
 (0.10605425200156694, 265),
 (0.10557486423652133, 2766),
 (0.10557486423652133, 822),
 (0.10534575807425914, 2491),
 (0.10519179459973899, 2676),
 (0.10496503453539867, 479),
 (0.10473498183402759, 1562),
 (0.10457183285346207, 472),
 (0.10425444516416421, 1465),
 (0.10425423248252984, 2709),
 (0.10421806120046974, 1676),
 (0.10403053110147295, 1798),
 (0.1040240341000106, 2823),
 (0.1040240341000106, 2501),
 (0.1039699081798286, 3504),
 (0.1039699081798286, 942),
 (0.10388368558329736, 751),
 (0.10368637066103771, 2250),
 (0.10368637066103771, 730),
 (0.10276872872705302, 2719),
 (0.10276872872705302, 2459),
 (0.10247708643417394, 2961),
 (0.1024231846229235, 1339),
 (0.10236183219704427, 3277),
 (0.10232708110688793, 735),
 (0.10228458667189501, 3198),
 (0.10228458667189501, 2962),
 (0.10228458667189501, 995),
 (0.10136053789274621, 2304),
 (0.10134976554256996, 2688),
 (0.10134976554256996, 1636),
 (0.10134976554256996, 1337),
 (0.10134976554256996, 1324),
 (0.10134976554256996, 707),
 (0.1011396883323341, 765),
 (0.10069601186190272, 1939),
 (0.10064515702280895, 3344),
 (0.10064515702280895, 2173),
 (0.10064515702280895, 1138),
 (0.10064515702280895, 461),
 (0.10064515702280895, 34),
 (0.10042325887796415, 300),
 (0.10029575753140649, 1989),
 (0.09998749056664377, 3107),
 (0.09930697078845709, 2221),
 (0.09914935926854875, 244),
 (0.0987669339896394, 3078),
 (0.0987669339896394, 2164),
 (0.09836837816997851, 3315),
 (0.09836837816997851, 875),
 (0.09817944238454097, 2699),
 (0.09794705270385755, 1027),
 (0.0975399452441692, 2587),
 (0.0975399452441692, 2077),
 (0.09748748613843083, 332),
 (0.09735701024742209, 2396),
 (0.09725019916278516, 258),
 (0.09699466636036946, 1546),
 (0.09699466636036946, 1335),
 (0.0968815397155067, 2907),
 (0.09686500832458751, 777),
 (0.09686052023444106, 135),
 (0.09673046950545752, 1444),
 (0.09673046950545752, 715),
 (0.09661595034015592, 446),
 (0.09653772583290808, 3207),
 (0.09653772583290808, 2138),
 (0.09626960058032391, 3120),
 (0.09618513758438467, 3515),
 (0.09565542764095197, 2863),
 (0.09565542764095197, 894),
 (0.09547947763468337, 2162),
 (0.09531096377076571, 702),
 (0.09528243773263785, 1974),
 (0.09509363527560924, 3003),
 (0.09498866156805089, 56),
 (0.09474781428844517, 684),
 (0.09470186479356585, 3105),
 (0.0942830918446496, 451),
 (0.09415396440857329, 2308),
 (0.09415396440857329, 1642),
 (0.09405941869361335, 1670),
 (0.09361290381636478, 3167),
 (0.09361290381636478, 2182),
 (0.09355616462549204, 3035),
 (0.09355616462549204, 1896),
 (0.09355616462549204, 1184),
 (0.09341909409291495, 3072),
 (0.09337462696070649, 438),
 (0.09296267142463642, 1603),
 (0.0928458913431107, 2710),
 (0.0928458913431107, 1213),
 (0.09284154955685534, 1371),
 (0.09266999833156657, 406),
 (0.09207228230404421, 1163),
 (0.09205966214580999, 2997),
 (0.09178619545179537, 2752),
 (0.09143824914038053, 3064),
 (0.09118758190755498, 601),
 (0.0904102863515016, 1752),
 (0.0904102863515016, 518),
 (0.08994784131054855, 1626),
 (0.08992899272331892, 1381),
 (0.08992899272331892, 274),
 (0.0895027880278346, 749),
 (0.08922252170461646, 234),
 (0.08893701129431314, 1801),
 (0.08888778897986298, 377),
 (0.08850344945738366, 3461),
 (0.08850344945738366, 1488),
 (0.08829971029325993, 674),
 (0.08820524746883318, 3011),
 (0.08820524746883318, 946),
 (0.08820524746883318, 693),
 (0.08817034107034716, 2919),
 (0.08817034107034716, 1007),
 (0.08817034107034716, 421),
 (0.08804345721280432, 1361),
 (0.08804345721280432, 383),
 (0.08762934372453222, 2228),
 (0.08757617301522154, 2408),
 (0.08743961293482269, 1848),
 (0.08723026371545585, 1439),
 (0.08711481957056837, 1181),
 (0.08702300996955852, 2415),
 (0.08665193755012388, 3381),
 (0.08665193755012388, 2287),
 (0.08665193755012388, 1836),
 (0.08665193755012388, 1769),
 (0.08613066304157803, 1118),
 (0.08604351125758435, 2905),
 (0.08604351125758435, 569),
 (0.08548454984937787, 3179),
 (0.08548454984937787, 2648),
 (0.08548454984937787, 230),
 (0.0851745887022458, 3048),
 (0.0851745887022458, 1994),
 (0.0851745887022458, 1204),
 (0.0847098159525699, 2515),
 (0.08462148604526211, 2429),
 (0.08426216025543003, 153),
 (0.08396067945350467, 1744),
 (0.08395033773740962, 1591),
 (0.08392814428727823, 2027),
 (0.08382123724955291, 2749),
 (0.08382123724955291, 2473),
 (0.08382123724955291, 1409),
 (0.08382123724955291, 1124),
 (0.08353088199290906, 444),
 (0.08317371353393369, 2642),
 (0.08309673441860337, 3111),
 (0.08309673441860337, 245),
 (0.08292925047434145, 3259),
 (0.08292925047434145, 1302),
 (0.08292925047434145, 293),
 (0.08291617096613416, 3140),
 (0.08269105630936234, 2901),
 (0.08250700752808858, 696),
 (0.08240929260535937, 1716),
 (0.0821143663223302, 3529),
 (0.0821143663223302, 3213),
 (0.0821143663223302, 2127),
 (0.0821143663223302, 1525),
 (0.0821143663223302, 1126),
 (0.08177476895231811, 2994),
 (0.08177476895231811, 2993),
 (0.08177476895231811, 2917),
 (0.08177476895231811, 1711),
 (0.08177476895231811, 1304),
 (0.08151167856717134, 2198),
 (0.08151167856717134, 945),
 (0.08151167856717134, 566),
 (0.081292170917407, 2998),
 (0.08105200153726197, 2484),
 (0.08077003694760779, 3076),
 (0.0806878432489464, 2360),
 (0.08068159929883512, 721),
 (0.08061100817847923, 2096),
 (0.08061100817847923, 1080),
 (0.08061100817847923, 892),
 (0.08061100817847923, 116),
 (0.08054857135944754, 3284),
 (0.08054857135944754, 2821),
 (0.08054857135944754, 1143),
 (0.08054857135944754, 963),
 (0.08054857135944754, 208),
 (0.08035021335850029, 1330),
 (0.08000278922985744, 592),
 (0.07992964611435344, 2480),
 (0.07984395488180059, 466),
 (0.07967648486597259, 3460),
 (0.07942048197153923, 1248),
 (0.07942048197153923, 800),
 (0.07936184382084922, 103),
 (0.07934695778482571, 1594),
 (0.0790924779245287, 2836),
 (0.07878339653169947, 921),
 (0.07872352699109669, 1427),
 (0.07832911303100597, 1274),
 (0.07826986945258207, 2529),
 (0.07826986945258207, 548),
 (0.07812731180375966, 2757),
 (0.07782256783754223, 1394),
 (0.07772413797438447, 1563),
 (0.0776677243681878, 1770),
 (0.0776677243681878, 431),
 (0.07763906076073131, 1969),
 (0.07750076957730183, 3443),
 (0.07750076957730183, 1597),
 (0.07750076957730183, 141),
 (0.07725627254289316, 3181),
 (0.07725627254289316, 2217),
 (0.07709690524884415, 3431),
 (0.07709690524884415, 2482),
 (0.07699366193593116, 2735),
 (0.0769319438415435, 2157),
 (0.0769319438415435, 1677),
 (0.07675730092547896, 909),
 (0.07667082959004155, 1219),
 (0.07608187763041152, 2026),
 (0.07602294882487318, 863),
 (0.07600285375632602, 1002),
 (0.0759181697785818, 3454),
 (0.07577029842086451, 1864),
 (0.07564976198948453, 2629),
 (0.07563006827415941, 2876),
 (0.0754734026281844, 1306),
 (0.07523752970018145, 893),
 (0.07523752970018145, 482),
 (0.07514147802049713, 2773),
 (0.07506743730551602, 4),
 (0.07496398600693509, 646),
 (0.07488884792464619, 3539),
 (0.07488884792464619, 388),
 (0.07470692889125978, 3114),
 (0.07470692889125978, 759),
 (0.07436171988469886, 1298),
 (0.07401855876239286, 88),
 (0.07393391675480251, 2052),
 (0.07393391675480251, 954),
 (0.07393391675480251, 812),
 (0.07355314671246128, 1316),
 (0.07331021916892902, 3270),
 (0.07327690911717218, 107),
 (0.07291562786035617, 2417),
 (0.0728684083665527, 1962),
 (0.07280881447765751, 1759),
 (0.0727652146198062, 619),
 (0.07247090110087959, 477),
 (0.07229078508957919, 590),
 (0.07137989306133816, 2435),
 (0.07121096313036739, 209),
 (0.0711621604468216, 3338),
 (0.07106080111048188, 3199),
 (0.07093515542997689, 3303),
 (0.07093515542997689, 1008),
 (0.07093515542997689, 805),
 (0.07064298286891789, 2022),
 (0.07050951911510893, 1033),
 (0.06999901722433252, 2035),
 (0.06983294154740267, 1128),
 (0.06977993867564153, 3210),
 (0.06969183120786246, 986),
 (0.06955745792024413, 3453),
 (0.06927229373883934, 404),
 (0.06901888758929042, 2352),
 (0.06901888758929042, 2061),
 (0.06901888758929042, 1526),
 (0.06901888758929042, 21),
 (0.06830194446478273, 1503),
 (0.06828111020089074, 2740),
 (0.06828111020089074, 2736),
 (0.06828111020089074, 1943),
 (0.06828111020089074, 1847),
 (0.06828111020089074, 1180),
 (0.06828111020089074, 829),
 (0.0682032266181214, 3248),
 (0.0682032266181214, 1619),
 (0.0682032266181214, 1257),
 (0.0682032266181214, 876),
 (0.06805916422035019, 2918),
 (0.06805916422035019, 2293),
 (0.06805916422035019, 1544),
 (0.06785022063152461, 3503),
 (0.06780627218311859, 2556),
 (0.06771935959217608, 2541),
 (0.06770398717597208, 3364),
 (0.06770398717597208, 3119),
 (0.06770398717597208, 3075),
 (0.06770398717597208, 859),
 (0.06770398717597208, 450),
 (0.06754168403023177, 264),
 (0.06694966279504407, 2806),
 (0.06694966279504407, 2369),
 (0.06694966279504407, 1832),
 (0.06669378295200436, 2082),
 (0.06648844683125392, 771),
 (0.06645518694941502, 3251),
 (0.06645518694941502, 1252),
 (0.06645518694941502, 1117),
 (0.06645292906426767, 3041),
 (0.06645292906426767, 2564),
 (0.06645292906426767, 2259),
 (0.06645292906426767, 2059),
 (0.06645292906426767, 2014),
 (0.06645292906426767, 1579),
 (0.06645292906426767, 842),
 (0.06645292906426767, 815),
 (0.06645292906426767, 526),
 (0.06626802899587075, 2797),
 (0.06584406166184281, 435),
 (0.06571699962537708, 1194),
 (0.06559989448715318, 2356),
 (0.0655031569982287, 396),
 (0.06412191076896116, 2260),
 (0.06412191076896116, 1894),
 (0.06409990552638617, 2246),
 (0.06396406009757603, 1625),
 (0.0629882398131742, 3594),
 (0.06272140140211431, 1565),
 (0.06272140140211431, 1513),
 (0.06272140140211431, 232),
 (0.06272140140211431, 18),
 (0.06271927037627348, 308),
 (0.06251092516187161, 725),
 (0.0621509926564875, 333),
 (0.061604560505147975, 2530),
 (0.06152215539240401, 3313),
 (0.06152215539240401, 3223),
 (0.06152215539240401, 556),
 (0.061201092717504006, 2599),
 (0.061201092717504006, 52),
 (0.060822883069674344, 2093),
 (0.06077659974490063, 3305),
 (0.06077659974490063, 2880),
 (0.06077659974490063, 2859),
 (0.06077659974490063, 1627),
 (0.06077659974490063, 1024),
 (0.06077659974490063, 657),
 (0.06061527321340795, 3184),
 (0.06061527321340795, 3031),
 (0.06061527321340795, 2944),
 (0.06061527321340795, 2647),
 (0.06061527321340795, 2528),
 (0.06061527321340795, 2261),
 (0.06061527321340795, 1269),
 (0.06061527321340795, 1109),
 (0.06061527321340795, 554),
 (0.0605113418129938, 10),
 (0.06025557611898251, 1538),
 (0.06025557611898251, 1129),
 (0.06013641813402835, 1131),
 (0.05999262335652932, 3149),
 (0.05999262335652932, 2846),
 (0.05999262335652932, 2624),
 (0.05999262335652932, 2461),
 (0.05999262335652932, 2359),
 (0.05999262335652932, 2279),
 (0.05999262335652932, 1069),
 (0.05999262335652932, 364),
 (0.05927023626041681, 1617),
 (0.059123169887753284, 3187),
 (0.05877791360283384, 3455),
 (0.05877791360283384, 1392),
 (0.05877791360283384, 704),
 (0.05826465255311857, 1835),
 (0.05826465255311857, 516),
 (0.05820238903108268, 189),
 (0.05790138179416361, 1342),
 (0.05763974687012355, 761),
 (0.05736186569390286, 480),
 (0.05724320662088986, 2239),
 (0.05724320662088986, 1734),
 (0.05720960329569059, 2809),
 (0.05720960329569059, 2057),
 (0.05720960329569059, 1691),
 (0.05664040415694123, 2029),
 (0.056621937029158506, 3379),
 (0.056621937029158506, 1774),
 (0.0564278865235491, 3062),
 (0.055853771367528685, 2619),
 (0.05576417002155924, 2951),
 (0.05576417002155924, 2224),
 (0.05576417002155924, 1026),
 (0.05576417002155924, 57),
 (0.05547547575885028, 2585),
 (0.055435054504402954, 1850),
 (0.055435054504402954, 1097),
 (0.055435054504402954, 807),
 (0.05502062125426616, 1600),
 (0.054991052287921414, 2650),
 (0.05493907332775764, 165),
 (0.05493907332775764, 45),
 (0.054863416369423136, 8),
 (0.054793242034454126, 276),
 (0.05402699630874494, 2193),
 (0.05402699630874494, 42),
 (0.05370254799315957, 1877),
 (0.05305980956943215, 3295),
 (0.05305980956943215, 3005),
 (0.05293101100444882, 1923),
 (0.05292864156660283, 2811),
 (0.052918966628638694, 1317),
 (0.0528978884762766, 2222),
 (0.05268951274578193, 1090),
 (0.05266839593639291, 3362),
 (0.05266839593639291, 1476),
 (0.05263105940007825, 1417),
 (0.05259054089966629, 1818),
 (0.05232043524255482, 1620),
 (0.05232043524255482, 497),
 (0.05194868290767959, 487),
 (0.05183816185568962, 1005),
 (0.05170169366608983, 2136),
 (0.05170169366608983, 700),
 (0.05170169366608983, 565),
 (0.05138757211673676, 3417),
 (0.05090385531872281, 2233),
 (0.05085971069291413, 1162),
 (0.050453398152483155, 1901),
 (0.050408082037084294, 94),
 (0.05033746451820631, 106),
 (0.05017128862015281, 1966),
 (0.05012558519824015, 3575),
 (0.050110577708012825, 1977),
 (0.04997505051609792, 1260),
 (0.04957823522570809, 1653),
 (0.04948672513322782, 2666),
 (0.04939479567119575, 705),
 (0.049026286011909795, 3298),
 (0.04882517448802302, 629),
 (0.048796833331736136, 2472),
 (0.048796833331736136, 1775),
 (0.048796833331736136, 494),
 (0.04846817068404057, 1595),
 (0.04817904263490586, 2361),
 (0.04785381019455731, 3017),
 (0.04778438162905383, 1857),
 (0.047749937179018, 3326),
 (0.047749937179018, 3227),
 (0.047749937179018, 2305),
 (0.047749937179018, 2156),
 (0.04725047923528368, 1489),
 (0.04673580212471322, 1988),
 (0.04668402365113107, 1903),
 (0.04573312371101507, 3361),
 (0.04573312371101507, 676),
 (0.0452051431757508, 3510),
 (0.0452051431757508, 3055),
 (0.0452051431757508, 2737),
 (0.0452051431757508, 2292),
 (0.0452051431757508, 1475),
 (0.045137142121380584, 2822),
 (0.04498602637267015, 2161),
 (0.044786100911622696, 896),
 (0.0443122935728851, 3235),
 (0.0443122935728851, 3049),
 (0.0443122935728851, 2154),
 (0.0443122935728851, 1978),
 (0.0443122935728851, 546),
 (0.04416876165065787, 3279),
 (0.04416876165065787, 2582),
 (0.04403938115580219, 2064),
 (0.04375757986584301, 97),
 (0.04349807110759742, 3570),
 (0.043347710079499176, 3399),
 (0.042665291615604624, 3310),
 (0.042665291615604624, 2076),
 (0.042665291615604624, 1407),
 (0.042665291615604624, 586),
 (0.041822606779345184, 3287),
 (0.041818373480833865, 724),
 (0.04176434834421346, 2532),
 (0.04125350376404429, 3336),
 (0.04125350376404429, 2656),
 (0.04125350376404429, 2570),
 (0.04125350376404429, 2522),
 ...]
</pre>

```python
invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
```


```python
for coef in coef_pos_index[:15]:
    print(invert_index_vectorizer[coef[1]], coef[0])
```

<pre>
이용 1.3321308087111168
추천 1.1098677278465363
버스 1.029120247844704
최고 0.9474432432978868
가성 0.9049132254229898
근처 0.8631251640260484
조식 0.8624237330200107
다음 0.7848182816732695
위치 0.732990219026413
공간 0.716865493140725
시설 0.7161355390234533
맛집 0.7134163462461057
거리 0.7044600617626677
분위기 0.6869152801231841
바다 0.6556108465327279
</pre>

```python
for coef in coef_neg_index[:15]:
    print(invert_index_vectorizer[coef[1]], coef[0])
```

<pre>
냄새 -1.124500886987929
별로 -0.9632209931825515
아무 -0.6811855513119685
화장실 -0.6683241824194205
그냥 -0.6491883332225628
모기 -0.6302873381425533
수건 -0.6243491941007028
느낌 -0.5975494080979522
모텔 -0.5971174361320487
다른 -0.5966138818945081
최악 -0.593317479621261
음식 -0.5443424935120069
주위 -0.5321043465183405
진짜 -0.5254380815734122
목욕 -0.5087212885846032
</pre>

```python
```
