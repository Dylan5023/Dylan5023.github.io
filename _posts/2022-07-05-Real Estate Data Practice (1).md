---
layout: single
title:  "Real Estate Data Practice"
categories: Pandas
tag: [python, blog, jekyll]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
**[Notice]** [utilize data from public data in **data.or.kr**]
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


# Practice Real Estate Data 


## DataFrame load


utilize data from public data in **data.or.kr**



```python
import pandas as pd
```


```python
df = pd.read_csv('https://bit.ly/ds-house-price')
```


```python
df
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격(㎡)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>전체</td>
      <td>2015</td>
      <td>10</td>
      <td>5841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>서울</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5652</td>
    </tr>
    <tr>
      <th>2</th>
      <td>서울</td>
      <td>전용면적 60㎡초과 85㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5882</td>
    </tr>
    <tr>
      <th>3</th>
      <td>서울</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5721</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서울</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td>5879</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>제주</td>
      <td>전체</td>
      <td>2020</td>
      <td>2</td>
      <td>3955</td>
    </tr>
    <tr>
      <th>4501</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>4039</td>
    </tr>
    <tr>
      <th>4502</th>
      <td>제주</td>
      <td>전용면적 60㎡초과 85㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>3962</td>
    </tr>
    <tr>
      <th>4503</th>
      <td>제주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4504</th>
      <td>제주</td>
      <td>전용면적 102㎡초과</td>
      <td>2020</td>
      <td>2</td>
      <td>3601</td>
    </tr>
  </tbody>
</table>
<p>4505 rows × 5 columns</p>
</div>


## 1. Column(rename)



```python
df = df.rename(columns = {'분양가격(㎡)' : '분양가격'})
df
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>전체</td>
      <td>2015</td>
      <td>10</td>
      <td>5841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>서울</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5652</td>
    </tr>
    <tr>
      <th>2</th>
      <td>서울</td>
      <td>전용면적 60㎡초과 85㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5882</td>
    </tr>
    <tr>
      <th>3</th>
      <td>서울</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5721</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서울</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td>5879</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>제주</td>
      <td>전체</td>
      <td>2020</td>
      <td>2</td>
      <td>3955</td>
    </tr>
    <tr>
      <th>4501</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>4039</td>
    </tr>
    <tr>
      <th>4502</th>
      <td>제주</td>
      <td>전용면적 60㎡초과 85㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>3962</td>
    </tr>
    <tr>
      <th>4503</th>
      <td>제주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4504</th>
      <td>제주</td>
      <td>전용면적 102㎡초과</td>
      <td>2020</td>
      <td>2</td>
      <td>3601</td>
    </tr>
  </tbody>
</table>
<p>4505 rows × 5 columns</p>
</div>


## 2. Data Overview 


### 2-1. check values of NaN and Data Type



```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4505 entries, 0 to 4504
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   지역명     4505 non-null   object
 1   규모구분    4505 non-null   object
 2   연도      4505 non-null   int64 
 3   월       4505 non-null   int64 
 4   분양가격    4210 non-null   object
dtypes: int64(2), object(3)
memory usage: 176.1+ KB
</pre>

```python
df.describe()
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
      <th>연도</th>
      <th>월</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4505.000000</td>
      <td>4505.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2017.452830</td>
      <td>6.566038</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.311432</td>
      <td>3.595519</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2015.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2016.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2017.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2019.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2020.000000</td>
      <td>12.000000</td>
    </tr>
  </tbody>
</table>
</div>


## 3. Data Preprocessing 



```python
df.loc[df['분양가격'] == '  ']
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>광주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td></td>
    </tr>
    <tr>
      <th>29</th>
      <td>광주</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td>대전</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td></td>
    </tr>
    <tr>
      <th>81</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>광주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>11</td>
      <td></td>
    </tr>
    <tr>
      <th>114</th>
      <td>광주</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>11</td>
      <td></td>
    </tr>
    <tr>
      <th>119</th>
      <td>대전</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>11</td>
      <td></td>
    </tr>
    <tr>
      <th>166</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>11</td>
      <td></td>
    </tr>
    <tr>
      <th>198</th>
      <td>광주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>199</th>
      <td>광주</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>204</th>
      <td>대전</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>251</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>283</th>
      <td>광주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2016</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>284</th>
      <td>광주</td>
      <td>전용면적 102㎡초과</td>
      <td>2016</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>289</th>
      <td>대전</td>
      <td>전용면적 102㎡초과</td>
      <td>2016</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>336</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2016</td>
      <td>1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



```python
#remove blank
df['분양가격'] = df['분양가격'].str.strip()
```


```python
df.loc[df['분양가격'] == '']
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>광주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td></td>
    </tr>
    <tr>
      <th>29</th>
      <td>광주</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td>대전</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td></td>
    </tr>
    <tr>
      <th>81</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>광주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>11</td>
      <td></td>
    </tr>
    <tr>
      <th>114</th>
      <td>광주</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>11</td>
      <td></td>
    </tr>
    <tr>
      <th>119</th>
      <td>대전</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>11</td>
      <td></td>
    </tr>
    <tr>
      <th>166</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>11</td>
      <td></td>
    </tr>
    <tr>
      <th>198</th>
      <td>광주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>199</th>
      <td>광주</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>204</th>
      <td>대전</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>251</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>12</td>
      <td></td>
    </tr>
    <tr>
      <th>283</th>
      <td>광주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2016</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>284</th>
      <td>광주</td>
      <td>전용면적 102㎡초과</td>
      <td>2016</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>289</th>
      <td>대전</td>
      <td>전용면적 102㎡초과</td>
      <td>2016</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>336</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2016</td>
      <td>1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


### Insert 0 for data with blank spaces



```python
df.loc[df['분양가격'] == '', '분양가격'] = 0
```

### Change NaN data by use fillna



```python
df['분양가격'] = df['분양가격'].fillna(0)
```


```python
df.loc[df['분양가격'] == '6,657']
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2125</th>
      <td>서울</td>
      <td>전체</td>
      <td>2017</td>
      <td>11</td>
      <td>6,657</td>
    </tr>
  </tbody>
</table>
</div>


### Remove comma



```python
df['분양가격'] = df['분양가격'].str.replace(',', '')
```


```python
df.iloc[2125]
```

<pre>
지역명       서울
규모구분      전체
연도      2017
월         11
분양가격    6657
Name: 2125, dtype: object
</pre>

```python
df['분양가격'] = df['분양가격'].fillna(0)
```


```python
df.loc[df['분양가격'] == '-']
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3683</th>
      <td>광주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2019</td>
      <td>5</td>
      <td>-</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>대전</td>
      <td>전용면적 60㎡이하</td>
      <td>2019</td>
      <td>5</td>
      <td>-</td>
    </tr>
    <tr>
      <th>3688</th>
      <td>대전</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2019</td>
      <td>5</td>
      <td>-</td>
    </tr>
    <tr>
      <th>3690</th>
      <td>울산</td>
      <td>전체</td>
      <td>2019</td>
      <td>5</td>
      <td>-</td>
    </tr>
    <tr>
      <th>3691</th>
      <td>울산</td>
      <td>전용면적 60㎡이하</td>
      <td>2019</td>
      <td>5</td>
      <td>-</td>
    </tr>
    <tr>
      <th>3692</th>
      <td>울산</td>
      <td>전용면적 60㎡초과 85㎡이하</td>
      <td>2019</td>
      <td>5</td>
      <td>-</td>
    </tr>
    <tr>
      <th>3693</th>
      <td>울산</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2019</td>
      <td>5</td>
      <td>-</td>
    </tr>
    <tr>
      <th>3694</th>
      <td>울산</td>
      <td>전용면적 102㎡초과</td>
      <td>2019</td>
      <td>5</td>
      <td>-</td>
    </tr>
    <tr>
      <th>3696</th>
      <td>세종</td>
      <td>전용면적 60㎡이하</td>
      <td>2019</td>
      <td>5</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



```python
df['분양가격'] = df['분양가격'].str.replace('-', '')
```


```python
df['분양가격'] = df['분양가격'].fillna(0)
```


```python
df.loc[df['분양가격'] == '', ['분양가격']] = 0
```


```python
df['분양가격'] = df['분양가격'].astype(int)
```


```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4505 entries, 0 to 4504
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   지역명     4505 non-null   object
 1   규모구분    4505 non-null   object
 2   연도      4505 non-null   int64 
 3   월       4505 non-null   int64 
 4   분양가격    4505 non-null   int64 
dtypes: int64(3), object(2)
memory usage: 176.1+ KB
</pre>
## Remove 'unnecessary dedicated area' in size classification column



```python
df
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>전체</td>
      <td>2015</td>
      <td>10</td>
      <td>5841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>서울</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5652</td>
    </tr>
    <tr>
      <th>2</th>
      <td>서울</td>
      <td>전용면적 60㎡초과 85㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5882</td>
    </tr>
    <tr>
      <th>3</th>
      <td>서울</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5721</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서울</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td>5879</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>제주</td>
      <td>전체</td>
      <td>2020</td>
      <td>2</td>
      <td>3955</td>
    </tr>
    <tr>
      <th>4501</th>
      <td>제주</td>
      <td>전용면적 60㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>4039</td>
    </tr>
    <tr>
      <th>4502</th>
      <td>제주</td>
      <td>전용면적 60㎡초과 85㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>3962</td>
    </tr>
    <tr>
      <th>4503</th>
      <td>제주</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4504</th>
      <td>제주</td>
      <td>전용면적 102㎡초과</td>
      <td>2020</td>
      <td>2</td>
      <td>3601</td>
    </tr>
  </tbody>
</table>
<p>4505 rows × 5 columns</p>
</div>



```python
df['규모구분'] = df['규모구분'].str.replace('전용면적', '')
```


```python
df['규모구분'].value_counts()
```

<pre>
전체               901
 60㎡이하           901
 60㎡초과 85㎡이하     901
 85㎡초과 102㎡이하    901
 102㎡초과          901
Name: 규모구분, dtype: int64
</pre>
### Check the average sale price by area name



```python
df.groupby('지역명')['분양가격'].mean()
```

<pre>
지역명
강원    2339.807547
경기    4072.667925
경남    2761.275472
경북    2432.128302
광주    2450.728302
대구    3538.920755
대전    2479.135849
부산    3679.920755
서울    7225.762264
세종    2815.098113
울산    1826.101887
인천    3578.433962
전남    2270.177358
전북    2322.060377
제주    2979.407547
충남    2388.324528
충북    2316.871698
Name: 분양가격, dtype: float64
</pre>

```python
df.loc[df['분양가격'] < 100]
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>광주</td>
      <td>85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>광주</td>
      <td>102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>대전</td>
      <td>102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>제주</td>
      <td>60㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>광주</td>
      <td>85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4461</th>
      <td>세종</td>
      <td>60㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4488</th>
      <td>전남</td>
      <td>85㎡초과 102㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4493</th>
      <td>경북</td>
      <td>85㎡초과 102㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4499</th>
      <td>경남</td>
      <td>102㎡초과</td>
      <td>2020</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4503</th>
      <td>제주</td>
      <td>85㎡초과 102㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>320 rows × 5 columns</p>
</div>



```python
df.loc[df['분양가격'] < 100].index
```

<pre>
Int64Index([  28,   29,   34,   81,  113,  114,  119,  166,  198,  199,
            ...
            4418, 4448, 4453, 4458, 4459, 4461, 4488, 4493, 4499, 4503],
           dtype='int64', length=320)
</pre>

```python
idx = df.loc[df['분양가격'] < 100].index
idx
```

<pre>
Int64Index([  28,   29,   34,   81,  113,  114,  119,  166,  198,  199,
            ...
            4418, 4448, 4453, 4458, 4459, 4461, 4488, 4493, 4499, 4503],
           dtype='int64', length=320)
</pre>

```python
df
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>전체</td>
      <td>2015</td>
      <td>10</td>
      <td>5841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>서울</td>
      <td>60㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5652</td>
    </tr>
    <tr>
      <th>2</th>
      <td>서울</td>
      <td>60㎡초과 85㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5882</td>
    </tr>
    <tr>
      <th>3</th>
      <td>서울</td>
      <td>85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5721</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서울</td>
      <td>102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td>5879</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>제주</td>
      <td>전체</td>
      <td>2020</td>
      <td>2</td>
      <td>3955</td>
    </tr>
    <tr>
      <th>4501</th>
      <td>제주</td>
      <td>60㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>4039</td>
    </tr>
    <tr>
      <th>4502</th>
      <td>제주</td>
      <td>60㎡초과 85㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>3962</td>
    </tr>
    <tr>
      <th>4503</th>
      <td>제주</td>
      <td>85㎡초과 102㎡이하</td>
      <td>2020</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4504</th>
      <td>제주</td>
      <td>102㎡초과</td>
      <td>2020</td>
      <td>2</td>
      <td>3601</td>
    </tr>
  </tbody>
</table>
<p>4505 rows × 5 columns</p>
</div>



```python
df = df.drop(idx, axis = 0)
```


```python
df.count()
```

<pre>
지역명     4185
규모구분    4185
연도      4185
월       4185
분양가격    4185
dtype: int64
</pre>

```python
df.groupby('지역명')['분양가격'].mean()
```

<pre>
지역명
강원    2412.642023
경기    4072.667925
경남    2814.376923
경북    2547.486166
광주    3049.028169
대구    3663.335938
대전    3128.433333
부산    3679.920755
서울    7225.762264
세종    2984.004000
울산    3043.503145
인천    3633.275862
전남    2304.969349
전북    2348.648855
제주    3432.795652
충남    2501.604743
충북    2316.871698
Name: 분양가격, dtype: float64
</pre>

```python
df.groupby('지역명')['분양가격'].count()
```

<pre>
지역명
강원    257
경기    265
경남    260
경북    253
광주    213
대구    256
대전    210
부산    265
서울    265
세종    250
울산    159
인천    261
전남    261
전북    262
제주    230
충남    253
충북    265
Name: 분양가격, dtype: int64
</pre>

```python
df.groupby('지역명')['분양가격'].max()
```

<pre>
지역명
강원     3906
경기     5670
경남     4303
경북     3457
광주     4881
대구     5158
대전     4877
부산     4623
서울    13835
세종     3931
울산     3594
인천     5188
전남     3053
전북     3052
제주     5462
충남     3201
충북     2855
Name: 분양가격, dtype: int64
</pre>

```python
df.groupby('지역명')['분양가격'].min()
```

<pre>
지역명
강원    2012
경기    3079
경남    2200
경북    2106
광주    2251
대구    2503
대전    2425
부산    2930
서울    5061
세종    2572
울산    2422
인천    2890
전남    1906
전북    1900
제주    2225
충남    2099
충북    1868
Name: 분양가격, dtype: int64
</pre>

```python
df.groupby('연도')['분양가격'].mean()
```

<pre>
연도
2015    2788.707819
2016    2934.250000
2017    3143.311795
2018    3326.951034
2019    3693.422149
2020    3853.960526
Name: 분양가격, dtype: float64
</pre>
### make pivot tale



```python
pd.pivot_table(df, index = '연도', columns = '규모구분', values = '분양가격')
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
      <th>규모구분</th>
      <th>102㎡초과</th>
      <th>60㎡이하</th>
      <th>60㎡초과 85㎡이하</th>
      <th>85㎡초과 102㎡이하</th>
      <th>전체</th>
    </tr>
    <tr>
      <th>연도</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015</th>
      <td>2980.977778</td>
      <td>2712.583333</td>
      <td>2694.490196</td>
      <td>2884.395833</td>
      <td>2694.862745</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>3148.099476</td>
      <td>2848.144279</td>
      <td>2816.965686</td>
      <td>3067.380435</td>
      <td>2816.073529</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>3427.649746</td>
      <td>3112.538071</td>
      <td>2981.950980</td>
      <td>3204.075145</td>
      <td>3008.279412</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>3468.355932</td>
      <td>3286.184783</td>
      <td>3227.458128</td>
      <td>3467.184211</td>
      <td>3235.098522</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>4039.854839</td>
      <td>3486.910112</td>
      <td>3538.545918</td>
      <td>3933.538462</td>
      <td>3515.974490</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>4187.566667</td>
      <td>3615.968750</td>
      <td>3594.852941</td>
      <td>4532.090909</td>
      <td>3603.911765</td>
    </tr>
  </tbody>
</table>
</div>


### Find out prices by year and size



```python
df.groupby(['연도', '규모구분'])['분양가격'].mean()
```

<pre>
연도    규모구분         
2015   102㎡초과          2980.977778
       60㎡이하           2712.583333
       60㎡초과 85㎡이하     2694.490196
       85㎡초과 102㎡이하    2884.395833
      전체               2694.862745
2016   102㎡초과          3148.099476
       60㎡이하           2848.144279
       60㎡초과 85㎡이하     2816.965686
       85㎡초과 102㎡이하    3067.380435
      전체               2816.073529
2017   102㎡초과          3427.649746
       60㎡이하           3112.538071
       60㎡초과 85㎡이하     2981.950980
       85㎡초과 102㎡이하    3204.075145
      전체               3008.279412
2018   102㎡초과          3468.355932
       60㎡이하           3286.184783
       60㎡초과 85㎡이하     3227.458128
       85㎡초과 102㎡이하    3467.184211
      전체               3235.098522
2019   102㎡초과          4039.854839
       60㎡이하           3486.910112
       60㎡초과 85㎡이하     3538.545918
       85㎡초과 102㎡이하    3933.538462
      전체               3515.974490
2020   102㎡초과          4187.566667
       60㎡이하           3615.968750
       60㎡초과 85㎡이하     3594.852941
       85㎡초과 102㎡이하    4532.090909
      전체               3603.911765
Name: 분양가격, dtype: float64
</pre>

```python
pd.DataFrame(df.groupby(['연도', '규모구분'])['분양가격'].mean())
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
      <th></th>
      <th>분양가격</th>
    </tr>
    <tr>
      <th>연도</th>
      <th>규모구분</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2015</th>
      <th>102㎡초과</th>
      <td>2980.977778</td>
    </tr>
    <tr>
      <th>60㎡이하</th>
      <td>2712.583333</td>
    </tr>
    <tr>
      <th>60㎡초과 85㎡이하</th>
      <td>2694.490196</td>
    </tr>
    <tr>
      <th>85㎡초과 102㎡이하</th>
      <td>2884.395833</td>
    </tr>
    <tr>
      <th>전체</th>
      <td>2694.862745</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2016</th>
      <th>102㎡초과</th>
      <td>3148.099476</td>
    </tr>
    <tr>
      <th>60㎡이하</th>
      <td>2848.144279</td>
    </tr>
    <tr>
      <th>60㎡초과 85㎡이하</th>
      <td>2816.965686</td>
    </tr>
    <tr>
      <th>85㎡초과 102㎡이하</th>
      <td>3067.380435</td>
    </tr>
    <tr>
      <th>전체</th>
      <td>2816.073529</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2017</th>
      <th>102㎡초과</th>
      <td>3427.649746</td>
    </tr>
    <tr>
      <th>60㎡이하</th>
      <td>3112.538071</td>
    </tr>
    <tr>
      <th>60㎡초과 85㎡이하</th>
      <td>2981.950980</td>
    </tr>
    <tr>
      <th>85㎡초과 102㎡이하</th>
      <td>3204.075145</td>
    </tr>
    <tr>
      <th>전체</th>
      <td>3008.279412</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2018</th>
      <th>102㎡초과</th>
      <td>3468.355932</td>
    </tr>
    <tr>
      <th>60㎡이하</th>
      <td>3286.184783</td>
    </tr>
    <tr>
      <th>60㎡초과 85㎡이하</th>
      <td>3227.458128</td>
    </tr>
    <tr>
      <th>85㎡초과 102㎡이하</th>
      <td>3467.184211</td>
    </tr>
    <tr>
      <th>전체</th>
      <td>3235.098522</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2019</th>
      <th>102㎡초과</th>
      <td>4039.854839</td>
    </tr>
    <tr>
      <th>60㎡이하</th>
      <td>3486.910112</td>
    </tr>
    <tr>
      <th>60㎡초과 85㎡이하</th>
      <td>3538.545918</td>
    </tr>
    <tr>
      <th>85㎡초과 102㎡이하</th>
      <td>3933.538462</td>
    </tr>
    <tr>
      <th>전체</th>
      <td>3515.974490</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2020</th>
      <th>102㎡초과</th>
      <td>4187.566667</td>
    </tr>
    <tr>
      <th>60㎡이하</th>
      <td>3615.968750</td>
    </tr>
    <tr>
      <th>60㎡초과 85㎡이하</th>
      <td>3594.852941</td>
    </tr>
    <tr>
      <th>85㎡초과 102㎡이하</th>
      <td>4532.090909</td>
    </tr>
    <tr>
      <th>전체</th>
      <td>3603.911765</td>
    </tr>
  </tbody>
</table>
</div>



```python
```
