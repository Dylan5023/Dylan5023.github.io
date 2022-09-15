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


