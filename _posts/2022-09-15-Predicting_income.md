---
layout: single
title:  "Predicting_income"
categories: Kaggle
tag: [python, blog, jekyll, matplotlib, seaborn, ML, Mathmatics, XAI]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

**[Notice]** [start to stduy XAI]
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


# Predicting the income range with financial data


## Introduction to Financial Data and Overview of Predictive Models

The problem of predicting customer income ranges is one of the most important problems in financial data analysis.

Before we get into the analysis, let's point out two things.


### <b> Properties of financial data</b>

Financial data mainly has the following characteristics:

- 1) <b>Combination of heterogeneous data</b>: Data source, form, scale, etc. have different characteristics

- 2) <b>skewedness of distribution</b>: If the predicted value and the correct answer are far apart, the bias of the learning result may be high.

- 3) <b>Unclearness of classification label</b>: Income section, credit rating, product type, etc. include business logic, so classification is arbitrary → Analyst’s interpretation is important

- 4) <b>multicollinearity of variables</b>: Interdependence or correlation between variables may be strong

- 5) <b>Nonlinearity of variables</b>: The influence of variables may not be linear, e.g.) What is the effect of age on income?

- Data may be incomplete (missing, truncated, censored) due to other practical limitations such as regulation, collection, and storage


### <b>Multi-classification and prediction of income brackets</b>

When there are more than 3 classes (also called labels or levels) to predict, it is called a multiclassification problem. In simple terms, it is called multiclass classification or multinomial logistic regression if you use a regression method. It is assumed that the hierarchical relationship (inclusion relationship) between classes is equivalent.



Forecasting income brackets is a classic multiple classification problem. Before analyzing, let's consider the following:

- 1) <b> In case the division between classes is not clear</b>: How should the division of income be established and how many classes should be decided?

- 2) <b>If there is an order in the divisions between classes</b>: To be precise, each income level should be viewed as an ordinal class.

- 3) <b>Insufficient value for a specific class</b>: How do you solve the difference between the number of customers in the high-income bracket and the number of customers in the middle-income bracket?



The multiclass classification problem has the following additional considerations compared to the binary classification problem.

- 1) <b>Cautions when implementing the model</b>: One-hot-encoding of variables, determination of objective function, etc.

- 2) <b>Cautions when interpreting results</b>: Accuracy, F1 score, Confusion Matrix, etc.


-----



```python
```


```python
```

## Load data to predict


### Introduction to data

 

- This topic uses data collected by the US Census Bureau and distributed by UCI to the US Adult Income dataset, with simulated variables added and modified by the instructor.

- The first data to be used is the US Adult Income dataset, and the columns are as follows.

 

 

- `age` : 나이

- `workclass`: 직업구분

- `education`: 교육수준

- `education.num`: 교육수준(numerically coded)

- `marital.status`: 혼인상태

- `occupation` : 직업

- `relationship`: 가족관계

- `race`: 인종

- `sex`: 성별

- `capital.gain`: 자본이득

- `capital.loss`: 자본손실

- `hours.per.week`: 주당 근로시간

- `income` : 소득 구분

 

Data from: https://archive.ics.uci.edu/ml/datasets/adult


--------------



```python
```


```python
```

### Import data



```python
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
```


```python
datapath = 'https://github.com/mchoimis/financialML/raw/main/income/'
df = pd.io.parsers.read_csv(datapath + 'income.csv')
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>?</td>
      <td>77053</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>Private</td>
      <td>132870</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>?</td>
      <td>186061</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>Private</td>
      <td>140359</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Divorced</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>Private</td>
      <td>264663</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Separated</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>


-----



```python
```


```python
```

### Data preview



```python
print(df.shape)
print(df.columns)
```




```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32561 entries, 0 to 32560
Data columns (total 15 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   age             32561 non-null  int64 
 1   workclass       32561 non-null  object
 2   fnlwgt          32561 non-null  int64 
 3   education       32561 non-null  object
 4   education.num   32561 non-null  int64 
 5   marital.status  32561 non-null  object
 6   occupation      32561 non-null  object
 7   relationship    32561 non-null  object
 8   race            32561 non-null  object
 9   sex             32561 non-null  object
 10  capital.gain    32561 non-null  int64 
 11  capital.loss    32561 non-null  int64 
 12  hours.per.week  32561 non-null  int64 
 13  native.country  32561 non-null  object
 14  income          32561 non-null  object
dtypes: int64(6), object(9)
memory usage: 3.7+ MB
</pre>
-----------------------



```python
```


```python
```

### Check Data



```python
# Replace missing values ​​with NaN
df[df =='?'] = np.nan
```


```python
# Filling out Missing Values ​​with Mode
for col in ['workclass', 'occupation', 'native.country']:
    df[col].fillna(df[col].mode()[0], inplace = True)
```


```python
# result
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>Private</td>
      <td>77053</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>Private</td>
      <td>132870</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>Private</td>
      <td>186061</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Widowed</td>
      <td>Prof-specialty</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>Private</td>
      <td>140359</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Divorced</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>Private</td>
      <td>264663</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Separated</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.isnull().sum()
```

<pre>
age               0
workclass         0
fnlwgt            0
education         0
education.num     0
marital.status    0
occupation        0
relationship      0
race              0
sex               0
capital.gain      0
capital.loss      0
hours.per.week    0
native.country    0
income            0
dtype: int64
</pre>
-----



```python
```


```python
```

## Feature Engineering


### Creating input features and target values



```python
X = df.drop(['income','education','fnlwgt'], axis =1)
y = df['income']
```




```python
X.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>Private</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>Private</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>Private</td>
      <td>10</td>
      <td>Widowed</td>
      <td>Prof-specialty</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>Private</td>
      <td>4</td>
      <td>Divorced</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>Private</td>
      <td>10</td>
      <td>Separated</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>


-----------



```python
y.head()
```

<pre>
0    <=50K
1    <=50K
2    <=50K
3    <=50K
4    <=50K
Name: income, dtype: object
</pre>

```python
```


```python
```

------


### Divide the raw data into training set and test set



```python
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
X_train.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32098</th>
      <td>40</td>
      <td>State-gov</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>25206</th>
      <td>39</td>
      <td>Local-gov</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>23491</th>
      <td>42</td>
      <td>Private</td>
      <td>10</td>
      <td>Never-married</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>12367</th>
      <td>27</td>
      <td>Local-gov</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Farming-fishing</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>7054</th>
      <td>38</td>
      <td>Federal-gov</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



```python
```


```python
```

----


### Handling categorical variables



```python
from sklearn.preprocessing import LabelEncoder

categorical = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
        le = LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])  
        X_test[feature] = le.transform(X_test[feature])    
```

### Check the result of categorical variable processing



```python
# Check the transformed categorical variable column (X_train)
X_train[categorical].head(3)
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
      <th>workclass</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>native.country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32098</th>
      <td>6</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>25206</th>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>38</td>
    </tr>
    <tr>
      <th>23491</th>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Checking the converted categorical variable column (X_test)

X_test[categorical].head(3)
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
      <th>workclass</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>native.country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22278</th>
      <td>3</td>
      <td>6</td>
      <td>11</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>8950</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>7838</th>
      <td>3</td>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>



```python
X_train[categorical].nunique()
```

<pre>
workclass          8
marital.status     7
occupation        14
relationship       6
race               5
sex                2
native.country    41
dtype: int64
</pre>

```python
X_test[categorical].nunique()
```

<pre>
workclass          8
marital.status     7
occupation        14
relationship       6
race               5
sex                2
native.country    40
dtype: int64
</pre>
----------


### Note: Handling of categorical variables

Categorical variables can be roughly divided into two methods.



- Convert class to number

- One-hot-encoding (dummy encoding)



In the case of financial data, categorical variables occupy most of the data, so when one-hot-encoding is performed, the majority of the entire dataset may have a value of 0. When there are many meaningless values ​​in a high-dimensional dataset, it is said that the features are sparse, and the learning efficiency may not be high.


### Scaling Features



```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()   
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns) 
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X.columns)
```


```python
# Check the scaled X_train data
X_train.head()
 
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
      <th>age</th>
      <th>workclass</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32098</th>
      <td>40</td>
      <td>6</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>38</td>
    </tr>
    <tr>
      <th>25206</th>
      <td>39</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>38</td>
    </tr>
    <tr>
      <th>23491</th>
      <td>42</td>
      <td>3</td>
      <td>10</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>38</td>
    </tr>
    <tr>
      <th>12367</th>
      <td>27</td>
      <td>1</td>
      <td>9</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>38</td>
    </tr>
    <tr>
      <th>7054</th>
      <td>38</td>
      <td>0</td>
      <td>14</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(min(X_train['age']))
print(max(X_train['age']))
print(np.mean(X_train['age']))
print(np.var(X_train['age']))
print('\n')
print(min(X_test['age']))
print(max(X_test['age']))
print(np.mean(X_test['age']))
print(np.var(X_test['age']))
```

<pre>
17
90
38.61429448929449
186.44402697680712


17
90
38.505476507319074
185.14136114309127
</pre>

```python
print(min(X_train_scaled['age']))
print(max(X_train_scaled['age']))
print(np.mean(X_train_scaled['age']))
print(np.var(X_train_scaled['age']))
print('\n')
print(min(X_test_scaled['age']))
print(max(X_test_scaled['age']))
print(np.mean(X_test_scaled['age']))
print(np.var(X_test_scaled['age']))
```

<pre>
-1.5829486507307393
3.7632934651328265
1.7567165303651125e-16
1.0


-1.5829486507307393
3.7632934651328265
-0.007969414769866482
0.9930130996694361
</pre>
### Note: feature scaler provided by scikit-learn



- `StandardScaler`: default scale, converts the mean of each feature to 0 and standard deviation to 1

- `RobustScaler`: Similar to the above, but uses the median, quartile, and quartile values ​​instead of the mean to minimize the influence of outliers

- `MinMaxScaler`: scale so that the maximum and minimum values ​​of all features are 1 and 0 respectively

- `Normalizer`: Normalized per row, not feature (column), and adjusts the data so that the Euclidean distance is 1.



<p> The reason for scaling is that training may not work properly when the values ​​of the data are too large or too small. Also, for classifiers where the effect of scale is absolute (e.g. distance-based algorithms such as knn), it is essential to consider scaling.

​    

<p> On the other hand, some items may be better to keep the distribution of the original data. For example, when data is standardized on features that are concentrated in almost one place to make the distributions the same, small changes may be learned as large differences. You can also omit it if you use a classifier that is not significantly affected by scale (e.g., a tree-based ensemble algorithm), if the performance is acceptable or if you are less concerned about overfitting.

​    

<p> One thing to keep in mind when scaling is that the original data may lose its meaning. It may be difficult to improve the model if the explanatory power of the original feature is lost when the purpose of finding an answer is not the ultimate goal, but the interpretation of the model or its application to other datasets in the future is more important. Please consider this together.


## Implementing a linear classification model


### Running a Logistic Regression Model Using Raw Data



```python
# Feature scaling 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression() 
logreg.fit(X_train, y_train )
```




### Checking the accuracy of the Logistic Regression model using raw data



```python
y_pred = logreg.predict(X_test)   
logreg_score = accuracy_score(y_test, y_pred)  

print('Logistic Regression accuracy score: {0:0.4f}'. format( logreg_score ))
```

<pre>
Logistic Regression accuracy score: 0.8057
</pre>
### Running a Logistic Regression Model Using Scaled Data



```python
# Feature scaling 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()  
logreg.fit( X_train_scaled, y_train ) 

```


### Checking the accuracy of a logistic regression model with scaled data





```python
y_pred = logreg.predict(X_test_scaled )  
logreg_score = accuracy_score(y_test, y_pred)  
print('Logistic Regression (scaled data) accuracy score: {0:0.4f}'. format(logreg_score))
```

<pre>
Logistic Regression (scaled data) accuracy score: 0.8222
</pre>

```python
y_pred
```

<pre>
array(['<=50K', '<=50K', '<=50K', ..., '<=50K', '<=50K', '<=50K'],
      dtype=object)
</pre>
### Check the classification result of Logistic Regression model using scaled data



```python
from sklearn.metrics import classification_report

cm_logreg = classification_report(y_test,y_pred)  
print(cm_logreg)
```

<pre>
              precision    recall  f1-score   support

       <=50K       0.84      0.94      0.89      7410
        >50K       0.71      0.44      0.55      2359
    
    accuracy                           0.82      9769
   macro avg       0.78      0.69      0.72      9769
weighted avg       0.81      0.82      0.81      9769

</pre>
-----------



```python
```


```python
```

## Implementation of tree-based classification model


### Implementing a Random Forest Model and Checking for Accuracy



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier()  
rfc.fit(X_train_scaled, y_train)
```

<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier()</pre></div></div></div></div></div>



```python
y_pred = rfc.predict(X_test_scaled ) 
rfc_score =  accuracy_score(y_test, y_pred )
print('Random Forest Model accuracy score : {0:0.4f}'. format(rfc_score))
```

<pre>
Random Forest Model accuracy score : 0.8503
</pre>
### Checking the Confusion Matrix of a Random Forest Model



```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)   
print('Confusion Matrix for Binary Labels \n')
# print('Confusion Matrix for Binary Labels\n')
# print('Actual class')
# print('Predicted', '[[True Positive', 'False Positive]')
# print('         ', '[False Negative', 'True Negative]]')
print(cm)
```

<pre>
Confusion Matrix for Binary Labels 

[[6858  552]
 [ 910 1449]]
</pre>

```python
# Calculating Recall and Precision in Confusion Matrix

print('\nRecall for Class [<=50K] = ', cm[0,0], '/' , cm[0,0] + cm[0,1])
print('\nPrecision for Class [<=50K] = ', cm[0,0], '/' , cm[0,0] + cm[1,0])
print('\nRecall for Class [>50K] = ', cm[1,1], '/' , cm[1,0] + cm[1,1])
print('\nPrecision for Class [>50K] = ', cm[1,1], '/' , cm[0,1] + cm[1,1])
```

### Check the classification result of the Random Forest model



```python
from sklearn.metrics import classification_report

cm_rfc =  classification_report(y_test, y_pred) 
print(cm_rfc)
```

<pre>
              precision    recall  f1-score   support

       <=50K       0.89      0.94      0.92      7410
        >50K       0.77      0.65      0.71      2359
    
    accuracy                           0.87      9769
   macro avg       0.83      0.80      0.81      9769
weighted avg       0.87      0.87      0.87      9769

</pre>
------



```python
```


```python
```

## Implementation of a boosting-based classification model


### Boosting





- Boosting is one of the ensemble algorithms that sums the fitted results of several trees, in which the concept of sequential is added. In other words, it means to catch the continuous weak learner, the current weak learner that reflects the error of the weak learner just before. This idea extends to the concept of catching weak learners in a gradient boosting model (GBM) in the direction of continuously reducing the loss



![boost](https://pluralsight2.imgix.net/guides/81232a78-2e99-4ccc-ba8e-8cd873625fdf_2.jpg)





- Boosting models include AdaBoost, Gradient Boosting Model (GBM), XGBoost, LightGBM, etc.





----



```python
```


```python
```

### Implement Gradient Boosting Model and Check Accuracy



```python
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state= 0) 
gbc.fit( X_train, y_train)
```

<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GradientBoostingClassifier(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">GradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>GradientBoostingClassifier(random_state=0)</pre></div></div></div></div></div>



```python
y_pred = gbc.predict(X_test)  
gbc_score = accuracy_score(y_pred, y_test)  
print('Gradient Boosting accuracy score : {0:0.4f}'.format(gbc_score))
```

<pre>
Gradient Boosting accuracy score : 0.8621
</pre>
### Check the classification result of the gradient boosting model



```python
from sklearn.metrics import classification_report

cm_gbc = classification_report(y_test, y_pred)   
print(cm_gbc)
```

<pre>
              precision    recall  f1-score   support

       <=50K       0.88      0.95      0.91      7410
        >50K       0.78      0.59      0.68      2359
    
    accuracy                           0.86      9769
   macro avg       0.83      0.77      0.79      9769
weighted avg       0.86      0.86      0.86      9769

</pre>
### Implement Light GBM and check accuracy



```python
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

lgbm = LGBMClassifier(random_state = 0)  
lgbm.fit(X_train, y_train )
y_pred = lgbm.predict(X_test)    

lgbm_score = accuracy_score(y_pred, y_test)  
print('LGBM Model accuracy score : {0:0.4f}'.format(lgbm_score))
```

<pre>
LGBM Model accuracy score : 0.8696
</pre>
###  Check the Light GBM classification result



```python
from sklearn.metrics import classification_report

cm_lgbm = classification_report(y_test, y_pred)  
print( cm_lgbm)
```

<pre>
              precision    recall  f1-score   support

       <=50K       0.89      0.94      0.92      7410
        >50K       0.77      0.65      0.71      2359
    
    accuracy                           0.87      9769
   macro avg       0.83      0.80      0.81      9769
weighted avg       0.87      0.87      0.87      9769

</pre>
### Comparing the Accuracy of the Income Binary Classification Prediction Model



```python
print ('Accuracy Comparisons for Binary Models\n')
print ('logreg_score:', '{0:0.5f}'.format(logreg_score))
print ('rfc_score   :', '{0:0.5f}'.format(rfc_score))
print ('gbc_score   :', '{0:0.5f}'.format(gbc_score ))
print ('lgbm_score  :', '{0:0.5f}'.format(lgbm_score))
```

<pre>
Accuracy Comparisons for Binary Models

logreg_score: 0.82219
rfc_score   : 0.85034
gbc_score   : 0.86211
lgbm_score  : 0.86959
</pre>
### Comparing the final binary classification model



```python
print ('Classification Comparions for Binary Models\n')
print ('logreg_score:', '{0:0.4f}'.format(logreg_score))
print (cm_logreg)
print ('rfc_score   :', '{0:0.4f}'.format(rfc_score))
print (cm_rfc)
print ('gbc_score   :', '{0:0.4f}'.format(gbc_score ))
print (cm_gbc)
print ('lgbm_score  :', '{0:0.4f}'.format(lgbm_score))
print (cm_lgbm)
```

<pre>
Classification Comparions for Binary Models

logreg_score: 0.8222
              precision    recall  f1-score   support

       <=50K       0.84      0.94      0.89      7410
        >50K       0.71      0.44      0.55      2359
    
    accuracy                           0.82      9769
   macro avg       0.78      0.69      0.72      9769
weighted avg       0.81      0.82      0.81      9769

rfc_score   : 0.8503
              precision    recall  f1-score   support

       <=50K       0.89      0.94      0.92      7410
        >50K       0.77      0.65      0.71      2359
    
    accuracy                           0.87      9769
   macro avg       0.83      0.80      0.81      9769
weighted avg       0.87      0.87      0.87      9769

gbc_score   : 0.8621
              precision    recall  f1-score   support

       <=50K       0.88      0.95      0.91      7410
        >50K       0.78      0.59      0.68      2359
    
    accuracy                           0.86      9769
   macro avg       0.83      0.77      0.79      9769
weighted avg       0.86      0.86      0.86      9769

lgbm_score  : 0.8696
              precision    recall  f1-score   support

       <=50K       0.89      0.94      0.92      7410
        >50K       0.77      0.65      0.71      2359
    
    accuracy                           0.87      9769
   macro avg       0.83      0.80      0.81      9769
weighted avg       0.87      0.87      0.87      9769

</pre>
-----



```python
```


```python
```

## Implementation of multi-classification model


### 

Introduction to data


- The second data to be written is the modified (+) and added (++) of the financial variables created by simulation to the US Adult data used earlier, and the columns are as follows.



income_ext.csv

- `age` : 나이

- `workclass`: 직업구분

- `education`: 교육수준

- `education.num`: 교육수준(numerically coded)

- `marital.status`: 혼인상태

- `occupation` : 직업

- `relationship`: 가족관계

- `sex`: 성별

- `capital.gain`: 자본이득

- `capital.loss`: 자본손실

- `hours.per.week`: 주당 근로시간

- `spending.groc`: 식료품 소비 금액(continuous) ++

- `spending.med`: 병의원 소비 금액(continuous) ++

- `spending.trav`: 여행 레저 소비 금액(continuous) ++

- `income` : 소득 이진구분( <=50K: 0, >50K: 1 ) +

- `income.num` : 소득금액(continuous) ++


### Data load



```python
data = pd.read_csv(datapath + 'income_ext.csv')
print(data.shape)
print(data.columns)
```

<pre>
(32561, 19)
Index(['age', 'workclass', 'education', 'education.num', 'marital.status',
       'occupation', 'relationship', 'sex', 'capital.gain', 'capital.loss',
       'hours.per.week', 'spend.groc', 'spend.med', 'spend.trav', 'asset',
       'balance', 'native.country', 'income', 'income.num'],
      dtype='object')
</pre>

```python
data.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>spend.groc</th>
      <th>spend.med</th>
      <th>spend.trav</th>
      <th>asset</th>
      <th>balance</th>
      <th>native.country</th>
      <th>income</th>
      <th>income.num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>?</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>77053</td>
      <td>13687</td>
      <td>3478</td>
      <td>345498</td>
      <td>-36</td>
      <td>United-States</td>
      <td>0</td>
      <td>43713</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>132870</td>
      <td>27294</td>
      <td>27567</td>
      <td>242642</td>
      <td>57638</td>
      <td>United-States</td>
      <td>0</td>
      <td>33113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>?</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Unmarried</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>186061</td>
      <td>32029</td>
      <td>24506</td>
      <td>578403</td>
      <td>46028</td>
      <td>United-States</td>
      <td>0</td>
      <td>31652</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>Private</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Divorced</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>140359</td>
      <td>22449</td>
      <td>23655</td>
      <td>229303</td>
      <td>7899</td>
      <td>United-States</td>
      <td>0</td>
      <td>55087</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>Private</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Separated</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>264663</td>
      <td>40863</td>
      <td>71439</td>
      <td>672614</td>
      <td>57675</td>
      <td>United-States</td>
      <td>0</td>
      <td>85583</td>
    </tr>
  </tbody>
</table>
</div>



```python
data['income'].value_counts()
```

<pre>
0    24720
1     7841
Name: income, dtype: int64
</pre>
------



```python
```


```python
```

### Light GBM 





- Gradient Boosting Decision Tree (GBDT), an ensemble model of Decision Tree, is also known in practice as XGboost (eXtreme Gradient Boosting). At each iteration, GBDT trains a decision tree by fitting a negative gradient (residual error).





- However, with high-dimensional large data, it was too time-consuming because it was necessary to scan the entire data to evaluate the information gain for all possible split points.





- Light GBM is a methodology that dramatically shortens the analysis time by <b>reducing the amount of data to be scanned</b> by using techniques such as sampling to overcome the shortcomings of the gradient boosting model.





- LGBM is a useful algorithm for <b>structured data with many categorical variables</b> and <b>multi-classification</b>, and it will be helpful if you know a simple principle.





- link: Light GBM: A Highly Efficient Gradient Boosting Decision Tree (NIPS 2017)

[https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html ] 


### Light GBM Parameter Description



- `objective` : objective function, regression, binary, multiclass available

- `categorical_feature` : Declare a categorical variable

- `max_cat_group` : Finds a branching point that prevents overfitting when there are many categorical variables, merges the category groups into a max_cat_group group, and finds a branching point at the group boundary, the default is 64

- `boosting`: boosting method, default is gbdt (gradient boosting decision tree), goss (Gradient-based One-Side Sampling) using sampling, dart (Dropouts meet Multiple Additive Regression Trees) like dropout in deep learning, rf (Random Forest), etc., can be selected

- `learning_rate` : learning rate, how much weight to give each predictor to learn

- `early_stopping_round` : Unit to stop when there is no more performance improvement in validation data

- `metric`: loss metric, binary_logloss, multi_logloss, mae, rmse, auc, cross_entropy, etc.


### Check Data



```python
# Create raw dataset for input
X = data.drop(['education', 'income', 'income.num'], axis = 1)                              # Drop columns
y = data['income.num']                              # Choose column
```


```python
# X raw 
X.head()
 
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
      <th>age</th>
      <th>workclass</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>spend.groc</th>
      <th>spend.med</th>
      <th>spend.trav</th>
      <th>asset</th>
      <th>balance</th>
      <th>native.country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>?</td>
      <td>9</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>77053</td>
      <td>13687</td>
      <td>3478</td>
      <td>345498</td>
      <td>-36</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>Private</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>132870</td>
      <td>27294</td>
      <td>27567</td>
      <td>242642</td>
      <td>57638</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>?</td>
      <td>10</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Unmarried</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>186061</td>
      <td>32029</td>
      <td>24506</td>
      <td>578403</td>
      <td>46028</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>Private</td>
      <td>4</td>
      <td>Divorced</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>140359</td>
      <td>22449</td>
      <td>23655</td>
      <td>229303</td>
      <td>7899</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>Private</td>
      <td>10</td>
      <td>Separated</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>264663</td>
      <td>40863</td>
      <td>71439</td>
      <td>672614</td>
      <td>57675</td>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



```python
# y raw 
y.head()
```

<pre>
0    43713
1    33113
2    31652
3    55087
4    85583
Name: income.num, dtype: int64
</pre>
### Creating Multiclassification Classes



```python
def value_change(x):
    if x <= 10000: return 0
    if x >  10000 and x <= 20000 : return 1
    if x >  20000 and x <= 30000 : return 2
    if x >  30000 and x <= 40000 : return 3
    if x >  40000 and x <= 50000 : return 4
    if x >  50000 and x <= 60000 : return 5
    if x >  60000 and x <= 70000 : return 6
    if x >  70000 and x <= 80000 : return 7
    if x >  80000 and x <= 90000 : return 8
    if x >  90000 and x <= 100000 : return 9
    return 10                          
                                       ## Define function

y = y.apply(lambda x: value_change(x))                                  ## Apply Lambda function
```


```python
# Check the label created

y.unique()
```

<pre>
array([ 4,  3,  5,  8,  7,  9,  2,  6,  1,  0, 10], dtype=int64)
</pre>

```python
y.value_counts()
```

<pre>
6     3318
9     3310
7     3276
4     3273
8     3257
1     3250
5     3246
3     3233
2     3180
0     3176
10      42
Name: income.num, dtype: int64
</pre>

```python
sns.set(font_scale=1.4)
y.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Income Labels", labelpad=14)
plt.ylabel("Counts", labelpad=14)
plt.title("Counts of 11 Income Labels\n", y=1.02);
```

<img src="data:image/png;base64,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5uaiW7duuHXrFhYvXow7d+4gKSkJgYGBcHFxQZcuXQAAfn5+2Lp1K/bu3YvU1FTMmTMHhYWFGDBgAADAx8cHlpaWCA4OxrVr13Dy5EmsXLkSfn5+Fe4FExERqYMk7+keP34chYWFSExMRMeOHZXmWrdujR07dmDjxo2IiIhAnz59oK+vD29vb4SEhEBLSwsAMHDgQOTn52PNmjXIzc1Fq1atsGXLFsVlaX19fURFRWHBggUYOHAgTE1N4evri4CAANGPl4iI3g+SLN3hw4dj+PDhVa7p2LFjhUL+O39/f/j7+790vlmzZoiJiXmjjERERK9LkpeXiYiIaiKWLhERkUhYukRERCJh6RIREYmEpUtERCQSli4REZFIWLpEREQiYekSERGJhKVLREQkEpYuERGRSFi6REREImHpEhERiYSlS0REJBKWLhERkUhYukRERCJh6RIREYmEpUtERCQSli4REZFIWLpEREQiYekSERGJhKVLREQkEpYuERGRSFi6REREImHpEhERiYSlS0REJBKWLhERkUhYukRERCJh6RIREYmEpUtERCQSli4REZFIWLpEREQiYekSERGJhKVLREQkEpYuERGRSFi6REREIpFs6RYUFGDJkiXo2rUrXF1d0bdvX5w8eVIxn56ejrFjx6J169bw9PREeHg4SktLlbYRHx8Pb29vODk5YdCgQbh8+bLSvCrbICIiUhfJlu6sWbPw448/YtGiRdi/fz+8vb0xceJE/Pbbb3j27Bn8/f0hCAJ27NiBhQsXYufOnYiIiFA8f9++fVi+fDmCgoKwd+9eNG/eHH5+fnj06BEAqLQNIiIidZJk6WZlZeH48eOYPXs2PD090axZM0yYMAFt27bF7t27cezYMdy7dw8rVqyAjY0NvL29MW3aNGzduhVFRUUAgI0bN8LX1xe9e/dGixYtEBoaCiMjI+zcuRMAVNoGERGROkmydGvXro3NmzfD3d1daVxLSwuPHz9GcnIy7OzsYGpqqpjz8PDAkydPcPnyZWRnZ+POnTto166dYr5WrVpwc3NDUlISALxyG0REROqmo+kAlTE2NkanTp2UxlJSUpCYmIg5c+bgzJkzsLKyUppv0KABACAjIwMGBgYAUOmaS5cuKdZVtY3XYW5u/Frr35SFhYko+1GVlPJIKQsgrTxSygIwT1WklAWQVh4xs7zNfb12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"/>

-----



```python
```


```python
```

### Feature Engineering



```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
```


```python
X.columns
```

<pre>
Index(['age', 'workclass', 'education.num', 'marital.status', 'occupation',
       'relationship', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week',
       'spend.groc', 'spend.med', 'spend.trav', 'asset', 'balance',
       'native.country'],
      dtype='object')
</pre>

```python
from sklearn.preprocessing import LabelEncoder

categorical = ['workclass','occupation', 'marital.status','relationship','sex', 'native.country' ]  
for feature in categorical:
        le = LabelEncoder()  
        x_train[feature] =le.fit_transform(x_train[feature])
        x_test[feature] = le.transform(x_test[feature]) 
```


```python
x_train[categorical].head()
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
      <th>workclass</th>
      <th>occupation</th>
      <th>marital.status</th>
      <th>relationship</th>
      <th>sex</th>
      <th>native.country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32098</th>
      <td>7</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>25206</th>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>39</td>
    </tr>
    <tr>
      <th>23491</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>12367</th>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>39</td>
    </tr>
    <tr>
      <th>7054</th>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>


-----



```python
```


```python
```

### Implementing multi-classification using Light GBM



```python
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

lgbm = LGBMClassifier() 
lgbm.fit(x_train, y_train)
```

<pre>
foo
</pre>

```python
"""
import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=y_train)
d_val = lgb.Dataset(x_test, label=y_test)

params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'multiclass' ###customize
params['metric'] = 'multi_logloss' ###customize
params['num_class'] = y_train.nunique() ###customize
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
"""
```

<pre>
"\nimport lightgbm as lgb\nd_train = lgb.Dataset(x_train, label=y_train)\nd_val = lgb.Dataset(x_test, label=y_test)\n\nparams = {}\nparams['learning_rate'] = 0.05\nparams['boosting_type'] = 'gbdt'\nparams['objective'] = 'multiclass' ###customize\nparams['metric'] = 'multi_logloss' ###customize\nparams['num_class'] = y_train.nunique() ###customize\nparams['num_leaves'] = 10\nparams['min_data'] = 50\nparams['max_depth'] = 10\n"
</pre>
------



```python
```


```python
```

### Check the Light GBM multi-classification results: Accuracy, Confusion Matrix, Heatmap



```python
## Finding Accuracy
y_pred1 = lgbm.predict(x_test)   

lgbm_score = accuracy_score(y_test, y_pred1)  
print('LGBM Model accuracy score : {0:0.4f}'.format(lgbm_score))
```

<pre>
LGBM Model accuracy score : 0.8151
</pre>

```python
pd.DataFrame(y_test).head(10)
```


```python
pd.DataFrame(y_pred1).head(10)
```


```python
## check out Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)  
print('LGBM Confusion Matrix for 11-class Labels\n')
print(cm1)
```

<pre>
LGBM Confusion Matrix for 11-class Labels

[[917  46   5   8   1   0   4   6   0   1   0]
 [ 36 862  50  11   7   1   1   1   0   0   0]
 [ 18  36 810  45  10   1   3   3   0   4   0]
 [ 13   7  66 847  60  11   5   4   3   3   0]
 [  8  10  11  65 773  64  20   6   7   5   0]
 [  8   6  17  15  81 755  88  18   9   5   0]
 [  8   8   9   8  12  83 714  79  13  17   0]
 [ 11   6  12   3  10   9  80 736  93  22   0]
 [  7   7   8   7   9  12   8 106 737  77   0]
 [  7   6   6  10  14   8   9  28  67 812   2]
 [  0   0   0   0   0   0   0   0   0  12   0]]
</pre>

```python
```


```python
## Visualize with heatmaps
plt.figure(figsize=[8,7])
sns.heatmap(cm1, annot=True, fmt = '.0f')
plt.title('LGBM Heatmap for 11-class Labels\n')
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
print('LGBM Model accuracy score : {0:0.4f}'.format(lgbm_score))
print('\n')
print(classification_report(y_test, y_pred1))
```

<pre>
LGBM Model accuracy score : 0.8151


              precision    recall  f1-score   support
    
           0       0.89      0.93      0.91       988
           1       0.87      0.89      0.88       969
           2       0.81      0.87      0.84       930
           3       0.83      0.83      0.83      1019
           4       0.79      0.80      0.79       969
           5       0.80      0.75      0.78      1002
           6       0.77      0.75      0.76       951
           7       0.75      0.75      0.75       982
           8       0.79      0.75      0.77       978
           9       0.85      0.84      0.84       969
          10       0.00      0.00      0.00        12
    
    accuracy                           0.82      9769
   macro avg       0.74      0.74      0.74      9769
weighted avg       0.81      0.82      0.81      9769

</pre>
--------------



```python
```


```python
```

## Evaluation and improvement of multi-classification models


### Improving the model by converting it to an appropriate number of classes



```python
## 인접 정확도의 계산
precise_accuracy =  np.sum(y_pred==y_test) / len(y_pred)
adjacent_accuracy =  np.sum(np.abs(y_pred - y_test) <= 1) / len(y_pred)

print('precise accuracy: {0:0.4f}'. format(precise_accuracy))
print('adjacent accuracy: {0:0.4f}'. format(adjacent_accuracy))
```

<pre>
precise accuracy: 0.7659
adjacent accuracy: 0.9335
</pre>

```python
def value_change(x):
    if x <= 20000: return 0
    if x >  20000 and x <= 50000 : return 1
    if x >  50000 and x <= 70000 : return 2
    if x >  70000 and x <= 90000 : return 3 
    return 4
                                       ## Define new function
y = data['income.num'].apply(lambda x: value_change(x))      ## Apply Lambda function
```


```python
y.value_counts()
```

<pre>
1    9686
2    6564
3    6533
0    6426
4    3352
Name: income.num, dtype: int64
</pre>

```python
y.value_counts().sum()
```

<pre>
32561
</pre>

```python
sns.set(font_scale=1.4)
y.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Income Labels", labelpad=14)
plt.ylabel("Counts", labelpad=14)
plt.title("Counts of 5 Income Labels\n", y=1.02);
```

<img src="data:image/png;base64,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Yfbxxx8Xqc5Tp07p6tWrGjRokPlnonnz5lq6dKlycnKKtE/4dXDEjMdauXLlJEn5+flWjT9y5Ihyc3PNv0jvaNWqlZ588kkdOnTooWto0aKFxWtPT0/l5OSooKDgnuMDAwOVnZ1tDp6vv/5a3t7eGj169D1DWZIOHTokT09P8y/gO377298qNzdX//jHP6yu95VXXlFcXJzmz5+vZ599Vp06ddKyZctUv359zZ8/3+p57ihs/0+fPq20tDQ9//zzFmP69++vHTt2yMXFRYcOHdJzzz1nDmVJKl++vLp166Zjx45ZXILw9/c3/+8aNWpIknx8fMxtdnZ2qlKlijIyMiTJfMTduXNn5eXlmf917txZGRkZ+u677x56f+9wdHTU8uXL1b17d126dEkHDhzQhg0btH//fkk/B/cd5cqVU9euXc2vnZyc9Nxzz5lP1xelzoYNG6pGjRoaPny4oqKitGvXLjk4OGjSpElq0KBBkfcLxY8jZjzWqlSpIldXV/3000/3HZOdna1bt26pevXqun79uqT//lL/pRo1aph/oT+M/z2FfucGs4KCgnvebObr66uEhAQlJCRozZo1io+PV9WqVdW/f39FRETIzs7urm2uX79+35olWYTXg9SuXVu1a9e2aHNwcFDbtm21bt06q+e5o7D9v3btmiTJzc3tvts/zL7d6w+X+13CkGR+/4CAgHv2X7p06b7bWuPLL7/UzJkz9Z///Eeurq7y8vKSq6urJMvPrletWlXly1v+OnZzczP/vBWlThcXF61bt05Lly7Vn//8Z23atEkODg7q2rWrpk6dqsqVKz/SvqH4EMx47LVr104HDx5UTk6O+dTwL+3YsUNTpkxRfHy8+Trz5cuX1bBhQ4txaWlp5qOvO+H4y6PeW7du2azmoKAgBQUFKScnR0lJSdq4caMWL16shg0b6qWXXrprfJUqVfSf//znrva0tDRJUrVq1ax+788//1yS1KVLF4v2nJych5rHGnfC4erVqxbtmZmZ+vbbb+Xn56cqVarcdbOY9N99q1q16l3bW6tSpUpydnbWmjVr7tlv7SWQe/nxxx81cuRI8xmH2rVry87OTmvXrr3rBqyMjIy7/lC7fPmyqlev/kh11q1bVx9++KEKCgp09OhR/d///Z8SEhJUuXJli0s1MBZOZeOxFxoaqvT09Ht+GcS1a9e0bNky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"/>

-----



```python
```


```python
```

### Feature Engineering



```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
```


```python
from sklearn.preprocessing import LabelEncoder

categorical = ['workclass', 'marital.status', 'occupation', 'relationship', 'sex', 'native.country']
for feature in categorical:
        le = LabelEncoder()
        x_train[feature] = le.fit_transform(x_train[feature])
        x_test[feature] = le.transform(x_test[feature])
```

-----



```python
```


```python
```

### Checking the improvement result of the Random Forest multi-classification model using the changed class



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier(random_state=0)
rfc.fit(x_train, y_train)
```



```python
## Check out the Confusion Matrix
y_pred = rfc.predict(x_test)   
cm3 = confusion_matrix(y_test, y_pred)  
print('Random Forest Confusion Matrix for 5-class Labels\n')
print(cm3)
```

<pre>
Random Forest Confusion Matrix for 5-class Labels

[[1811  144    2    0    0]
 [ 105 2721   84    8    0]
 [  31  194 1609  118    1]
 [  26   81  134 1680   39]
 [  13   56   29  253  630]]
</pre>

```python
# Draw a Heatmap

plt.figure(figsize=[8,7])
sns.heatmap(cm3, cmap="Reds", annot=True, fmt='.0f')
plt.title('Random Forest Heatmap for 5-class Labels\n', fontsize=14)
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
## Accruacy Evaluation
precise_accuracy =  np.sum(y_pred==y_test) / len(y_pred) # exact_match
adjacent_accuracy =  np.sum(np.abs(y_pred - y_test) <=1) / len(y_pred)

print('precise accuracy: {0:0.4f}'. format(precise_accuracy))
print('adjacent accuracy: {0:0.4f}'. format(adjacent_accuracy))
```

<pre>
precise accuracy: 0.8651
adjacent accuracy: 0.9747
</pre>

```python
## Comparison of two Accruacy scores
print('Random Forest precise  accuracy for 5 labels: {0:0.4f}'. format(precise_accuracy))
print('Random Forest adjacent accuracy for 5 labels: {0:0.4f}'. format(adjacent_accuracy))
print('\n')
print(classification_report(y_test, y_pred))
```

<pre>
Random Forest precise  accuracy for 5 labels: 0.8651
Random Forest adjacent accuracy for 5 labels: 0.9747


              precision    recall  f1-score   support
    
           0       0.91      0.93      0.92      1957
           1       0.85      0.93      0.89      2918
           2       0.87      0.82      0.84      1953
           3       0.82      0.86      0.84      1960
           4       0.94      0.64      0.76       981
    
    accuracy                           0.87      9769
   macro avg       0.88      0.84      0.85      9769
weighted avg       0.87      0.87      0.86      9769

</pre>
------



```python
```


```python
```

### Checking the improvement results of the Ligth GBM multi-classification model using the changed class



```python
lgbm = LGBMClassifier(random_state=0)  
lgbm.fit(x_train, y_train)
y_pred1 = lgbm.predict(x_test) 

lgbm_score1 = accuracy_score(y_test, y_pred1)
print('LGBM Model accuracy score : {0:0.4f}'.format( lgbm_score1 ))
```

<pre>
LGBM Model accuracy score : 0.9030
</pre>

```python
## Confusion Matrix 

cm4 = confusion_matrix(y_test, y_pred1)   
print('LGBM Confusion Matrix for 5-class Labels\n')
print(cm4)
```

<pre>
LGBM Confusion Matrix for 5-class Labels

[[1858   93    2    4    0]
 [  75 2756   70   13    4]
 [  26  129 1700   95    3]
 [  24   54  106 1717   59]
 [  13   35   24  119  790]]
</pre>

```python
# heatmap using seaborn

plt.figure(figsize=[8, 7])
sns.heatmap(cm3, cmap='Reds', annot=True, fmt='.0f')
plt.title('Random Forest Classifier Heatmap for 5-class Labels\n', fontsize=14)
plt.show()
```

<img src="data:image/png;base64,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c/jwYcXDw0M5c+aM8vjxY2XKlClK9erVlaSkpFT7zpw5o7i6uioJCQnKixcvNHG/ePFC6/UVRVF8fHyUXr16Kbdv31auX7+udOrUSfHx8dG8N1xdXZWRI0cqjx49Up48eaKzP35+fkr79u1T3NcaNWpo3ifHjh1Tqlatqhw5ckR5+PChsnTpUqV8+fJKaGiooiiK0q1bN6Vfv35KQECAcv/+faVPnz5Ko0aNFEVRlL/++kupWbOmsnbtWiUiIkLznu3Vq5cSEBCgrFixQilRooTSokUL5cqVK8rhw4eVkiVLKseOHVMURVHGjRuntGvXTrl165by8OFDZfz48YqHh4cSFxenKErye7RatWrK8ePHlWvXrilffPGFMnjwYJ39SUxMVMLCwhRXV1flwIEDSkREhBIQEKCUKVNGWbBggfLgwQNl9+7dSrly5ZQDBw4oiqIoo0ePVooXL64cPnxYuXr1qpKUlKQ1ZlxcnFKyZEll2LBhSr169ZS6desq8+bN08SWku+++05xd3dXfvjhByUoKEgZOXKk0qBBA0WtViujR49Whg8friiKouzdu1epXLmycurUKeXJkyfKli1bFFdXV+Xq1auKoihK06ZNlQkTJiiPHz9Wzp07p1SpUkXZunXrB/ve1blzZ2XhwoUp9kVGRioeHh7KkiVLlMePHysXLlxQGjRooEyePFlRlP//G+3du7dy7949Zffu3UqpUqWU33///aPeN2//DfwnMYv045NNvqVLl1bKly+vlC9fXildurTi7u6ujBgxQpMEoqOjlVWrVilRUVGa523dulWpXbu2oij//wN7584dTf+AAQOUMWPGKIqiKAMHDlRGjx6t6Xvx4oVSunRpZdeuXYparVY8PT01iVpRFCU+Pl6pVauW8u233yqKkvyH7c3/K4qitGjRQhk1apTm5+nTpyu9evVKcf927dqluLm5afbvzWPGjBmKoijK0aNHlbJly2olw19++UUpUaKEEhERofmH/fa+16xZUzl8+LDmZ7VarXh5eSl79uxRbt68qbi6uionTpzQ9J86dUp5+fKloiip/5HZt2+fJuGl5u3k96Hfzbp165SqVasqjx8/VhQl+Q/ZqVOnlISEhFT73k6+78b99us/evRIcXNz0zp+wcHBiqurq/Ls2TPNOHfv3n3v/vj5+SnFixfX+R2VL19ecXNz0yTf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"/>


```python
# heatmap using seaborn

plt.figure(figsize=[8, 7])
sns.heatmap(cm4, cmap='Reds', annot=True, fmt='.0f')
plt.title('LGBM Heatmap for 5-class Labels\n', fontsize=14)
plt.show()
```

<img src="data:image/png;base64,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6dSpU6qfXVRUFMHBwbi6ujJjxgxevXrFjBkzGDJkCMuXL091nS1btvDll19qKhxDhgwhOTmZ8PBwreUsLCyYNm0aDRo0ACBfvnysWbOGu3fvAimT9lxcXMiXLx9mZmb4+/trqilp9f0TMTExtG7dmvbt22sqKM2bN2fZsmVay02dOhUXFxdcXV3p2rUrW7ZsoXXr1h/83rwtvWIWQlck+X7CbG1tFaOili1bUqdOHQDWrl2r+eMKkCtXLjZt2gRAYmIiT548Yd68eXh7e2sS9Bt169bF29ubxMRETp8+TbFixciZM+cHY3r7Pd42ePBgzf8HBgZy+/ZtrZnWCQkJlCtXDoBmzZpx5MgRduzYwb1797h27RqApnybmvz582v+39zcnLx582p+trCwID4+HoB69erx22+/MWPGDO7du8f169cV2357RFqgQAHs7e0JDAxUJN93vVnm7WqDm5sbL1++5MWLF0BK0nsfKysrzp07h5WVFcbGxgDMmDGDli1b8uTJEwICArQOfr7//nsCAwNp3bq1ps3IyIjhw4crtl2mTBksLCzw8/Pj7t273Lp1iwcPHlClShUAevXqxciRIzl69Cg1atSgQYMGfPXVVx/s+ycsLS1p3749+/bt4+rVq5rP397eXrOMubm55qALwNXVlVWrVgEf/t68Lb1iFkJXJPl+wsqXL09MTAw3b96kZMmSQEpCtrW11fz/24yMjHB2dtb8XLRoUaytrWnXrh137tyhePHimr6KFStibGzMhQsXOHr0KPXr1/+omN59jzfMzc01/5+UlET37t1p0aKF1jJmZmYADB8+nD/++IOmTZvSvn17cufOTdu2bdN833cnO71dDXjb/Pnz2bZtGy1btqRp06ZMmDBBc7DyxpvE93a879ve2ywsLBRtb5L6m/++/Tmk5t1/s6JFiwIQEhLCwIED6d69u6bPwcEhzdMKbztx4gTe3t40a9ZMMzt70qRJmv7GjRtTrVo1jhw5wq+//sqoUaM4efIkM2bMSLPvn4iKiqJVq1bY2dlRr149GjduzL1797RG9e+OTtVqtebf9kPfm7elV8xC6Iqc8/2Eubq6Uq5cORYtWkRqT4YMDQ394DberJeUlKTVbmRkRK1atTh27BjHjx9Pt/O9AIULF+bRo0c4OztrXvv27ePw4cNERkZy4MAB5s6dy8CBA6lfvz6v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"/>


```python
## Accuracy Evalaution
precise_accuracy1 =  np.sum(y_pred1 == y_test) / len(y_pred1) # exact_match
adjacent_accuracy1 =  np.sum(np.abs(y_pred1 - y_test) <= 1) / len(y_pred1)   

print('LGBM precise  accuracy for 5 labels: {0:0.4f}'. format(precise_accuracy1))
print('LGBM adjacent accuracy for 5 labels: {0:0.4f}'. format(adjacent_accuracy1))
print('\n')
print(classification_report(y_test, y_pred1))
```

<pre>
LGBM precise  accuracy for 5 labels: 0.9030
LGBM adjacent accuracy for 5 labels: 0.9793


              precision    recall  f1-score   support
    
           0       0.93      0.95      0.94      1957
           1       0.90      0.94      0.92      2918
           2       0.89      0.87      0.88      1953
           3       0.88      0.88      0.88      1960
           4       0.92      0.81      0.86       981
    
    accuracy                           0.90      9769
   macro avg       0.91      0.89      0.90      9769
weighted avg       0.90      0.90      0.90      9769

</pre>
-----------



```python
```


```python
```

### Comparing the results of the income multi-class prediction model



```python
print ('Accuracy Comparisons for Multiclass Models\n')
print ('rfc_score  (11 labels)  :', '{0:0.5f}'.format(rfc_score)) 
print ('rfc_score  ( 5 labels)  :', '{0:0.5f}'.format(precise_accuracy)) 
print ('lbgm_score (11 labels)  :', '{0:0.5f}'.format(lgbm_score)) 
print ('lgbm_score ( 5 labels)  :', '{0:0.5f}'.format(precise_accuracy1)) 
```

<pre>
Accuracy Comparisons for Multiclass Models

rfc_score  (11 labels)  : 0.76589
rfc_score  ( 5 labels)  : 0.86508
lbgm_score (11 labels)  : 0.81513
lgbm_score ( 5 labels)  : 0.90296
</pre>

```python
print ('Classification Comparions for Multiclass Models\n')
print ('rfc_score   :', '{0:0.4f}'.format(rfc_score))
print (classification_report(y_test, y_pred)) 
print ('lgbm_score  :', '{0:0.4f}'.format(lgbm_score))
print (classification_report(y_test, y_pred1))
```

<pre>
Classification Comparions for Multiclass Models

rfc_score   : 0.7659
              precision    recall  f1-score   support

           0       0.91      0.93      0.92      1957
           1       0.85      0.93      0.89      2918
           2       0.87      0.82      0.84      1953
           3       0.82      0.86      0.84      1960
           4       0.94      0.64      0.76       981
    
    accuracy                           0.87      9769
   macro avg       0.88      0.84      0.85      9769
weighted avg       0.87      0.87      0.86      9769

lgbm_score  : 0.8151
              precision    recall  f1-score   support

           0       0.93      0.95      0.94      1957
           1       0.90      0.94      0.92      2918
           2       0.89      0.87      0.88      1953
           3       0.88      0.88      0.88      1960
           4       0.92      0.81      0.86       981
    
    accuracy                           0.90      9769
   macro avg       0.91      0.89      0.90      9769
weighted avg       0.90      0.90      0.90      9769

</pre>
----------



```python
```


```python
```

## conclusion



- 1) Understanding binary and multi-classification models <p>: A modeling method that varies depending on the number of classification items (parameter processing)



- 2) Understanding of linear and tree-based classification models <p>: Logistic Regression, Random Forest, Gradient Boosting, <strong>Light GBM</strong>, etc.





- 3) Understanding how to process variables<p>: Categorical variable handling, scaling, 3 or more classes





- 4) Learning how to interpret classification model results <p>: simple accuracy, <b>proximity accuracy</b>, Precision, Recall, etc.





- 5) Learn how to improve the model based on the evaluation results<p>: Comprehensively use the F1 score, Confusion Matrix, Classification Report, etc.



