---
layout: single
title:  "Pokemon_classification"
categories: ML
tag: [python, blog, jekyll, matplotlib, seaborn, ML]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

**[Notice]** [ML_practical practice_2]
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
df = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/Pokemon.csv")
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
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


-----


## 2) EDA 



```python
df.shape
```

<pre>
(800, 13)
</pre>

```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 800 entries, 0 to 799
Data columns (total 13 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   #           800 non-null    int64 
 1   Name        800 non-null    object
 2   Type 1      800 non-null    object
 3   Type 2      414 non-null    object
 4   Total       800 non-null    int64 
 5   HP          800 non-null    int64 
 6   Attack      800 non-null    int64 
 7   Defense     800 non-null    int64 
 8   Sp. Atk     800 non-null    int64 
 9   Sp. Def     800 non-null    int64 
 10  Speed       800 non-null    int64 
 11  Generation  800 non-null    int64 
 12  Legendary   800 non-null    bool  
dtypes: bool(1), int64(9), object(3)
memory usage: 75.9+ KB
</pre>

```python
df.isnull().sum()
```

<pre>
#               0
Name            0
Type 1          0
Type 2        386
Total           0
HP              0
Attack          0
Defense         0
Sp. Atk         0
Sp. Def         0
Speed           0
Generation      0
Legendary       0
dtype: int64
</pre>

```python
df['Legendary'].value_counts()
```

<pre>
False    735
True      65
Name: Legendary, dtype: int64
</pre>

```python
df['Generation'].value_counts()
```

<pre>
1    166
5    165
3    160
4    121
2    106
6     82
Name: Generation, dtype: int64
</pre>

```python
df['Generation'].value_counts().sort_index().plot()
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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"/>


```python
df['Type 1'].unique()
```

<pre>
array(['Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Poison', 'Electric',
       'Ground', 'Fairy', 'Fighting', 'Psychic', 'Rock', 'Ghost', 'Ice',
       'Dragon', 'Dark', 'Steel', 'Flying'], dtype=object)
</pre>

```python
df['Type 2'].unique()
```

<pre>
array(['Poison', nan, 'Flying', 'Dragon', 'Ground', 'Fairy', 'Grass',
       'Fighting', 'Psychic', 'Steel', 'Ice', 'Rock', 'Dark', 'Water',
       'Electric', 'Fire', 'Ghost', 'Bug', 'Normal'], dtype=object)
</pre>

```python
len(df[df['Type 2'].notnull()]['Type 2'].unique())
```

<pre>
18
</pre>
-----


### 2-2) Explore data features



```python
fig = plt.figure(figsize = (12, 12))
ax = fig.gca()
sns.boxplot(data = df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']], ax = ax)
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
fig = plt.figure(figsize = (12, 12))
ax = fig.gca()
sns.boxplot(data =df[df['Legendary'] == 1][['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']], ax = ax)
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAsYAAAKrCAYAAADyAksxAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAArt0lEQVR4nO3df5Sd913Y+fdXkUJMDCTYTkgjggDZUKA5oWi9bTlwEooNAhbKtl2c023Vkt1AD1gtsGcL1EmziWlp+bU7ocBJNynKbkmgDZymgMAOJFB2ASMT4/y2JiBYhRBbDgEbm0S2vvvHXIFsJOvHzOh6pNfrHJ2Z+9x7n/nMPLp33vOdZ2bGnDMAALjcbVv2AAAA8GQgjAEAIGEMAACVMAYAgEoYAwBAVduXPUDV1VdfPXft2rXsMQAAuMTdeeedx+ac15zuuidFGO/atatDhw4tewwAAC5xY4zfPdN1TqUAAICEMQAAVMIYAAAqYQwAAJUwBgCAShgDAEAljAEAoBLGAABQCWMAAKiEMQAAVMIYAAAqYQwAAJUwBgCAShgDAEAljAEAoBLGAABQCWMAAKiEMQAAVMIYAAAqYQwAAJUwBgCAShgDAEAljAEAoBLGAABQCWPgEnXs2LFuvvnm7r///mWPApcdjz+2qrOG8RjjU8cYbx1jvGeM8a4xxj9ZbP/kMcbtY4zDi5fPPOU+3zHGWB1jvG+M8WWb+Q4AnM6BAwe6++67O3DgwLJHgcuOxx9b1bmsGD9Sfduc8y9Xf636pjHG51TfXv3CnPPa6hcWl1tcd1P1udWXVz80xnjKZgwPcDrHjh3r4MGDzTk7ePCgVSu4iDz+2MrOGsZzzg/OOX9z8foD1Xuq51ZfU538UvBA9bcWr39N9cY550fnnL9TrVbXb/DcAGd04MCB5pxVnThxwqoVXEQef2xl53WO8RhjV/X51a9Xz55zfrDW4rl61uJmz63+v1PudnSx7fH7eukY49AY49B99913AaMDnN7tt9/e8ePHqzp+/Hi33XbbkieCy4fHH1vZOYfxGOPK6k3VP51z/vET3fQ02+Zf2DDna+ace+ace6655ppzHQPgrG644YZ27NhR1Y4dO7rxxhuXPBFcPjz+2MrOKYzHGDtai+L/MOf8ycXmD40xnrO4/jnVvYvtR6tPPeXuO6vf35hxAc5u3759jbH2Nfq2bdvat2/fkieCy4fHH1vZufxWilG9tnrPnPP7T7nqzdXJ/+37qv98yvabxhgfN8b49Ora6o6NGxngiV199dXt3bu3MUZ79+7tqquuWvZIcNnw+GMr234Ot/nC6u9X7xhj3LXY9p3Vd1c/McZ4SfV71d+tmnO+a4zxE9W7W/uNFt8053x0owcHeCL79u3ryJEjVqtgCTz+2KrGyZ8cXaY9e/bMQ4cOLXsMAAAucWOMO+ece053nb98BwAACWMAAKiEMQAAVMIYAAAqYQwAAJUwBgCAShgDAEAljAEAoBLGAABQCWMAAKiEMQAAVMIYAAAqYQwAAFVtX/YAAGxdKysrra6ubvh+jx49WtXOnTs3fN+7d+9u//79G75fYOsTxgA86Tz88MPLHgG4DAljAC7YZq28ntzvysrKpuwf4HScYwwAAAljAACohDEAAFTCGAAAKmEMAACVMAYAgEoYAwBAJYzhCR07dqybb765+++/f9mjAACbTBjDEzhw4EB33313Bw4cWPYoAMAmE8ZwBseOHevgwYPNOTt48KBVYwC4xAljOIMDBw4056zqxIkTVo0B4BInjOEMbr/99o4fP17V8ePHu+2225Y8EQCwmYQxnMENN9zQjh07qtqxY0c33njjkicCADaTMIYz2LdvX2OMqrZt29a+ffuWPBEAsJmEMZzB1Vdf3d69extjtHfv3q666qpljwQAbKLtyx4Ansz27dvXkSNHrBYDwGVAGMMTuPrqq3v1q1+97DEAgIvAqRQAAJAwBgCAShgDAEAljAEAoBLGAABQCWMAAKiEMQAAVMIYAAAqYQwAAJUwBgCAShgDAEAljAEAoBLGAABQCWMAAKiEMQAAVMIYAAAqYQwAAJUwBgCAShgDAEAljAEAoBLGAABQCWMAAKiEMQAAVMIYntAdd9zRC1/4wu68885ljwIAm+7YsWPdfPPN3X///cseZSmEMTyBV7ziFZ04caKXvexlyx4FADbdgQMHuvvuuztw4MCyR1kKYQxncMcdd/Tggw9W9eCDD1o1BuCSduzYsQ4ePNics4MHD16Wq8bCGM7gFa94xWMuWzUG4FJ24MCB5pxVnThx4rJcNRbGcAYnV4vPdBkALiW33357x48fr+r48ePddtttS57o4hPGcAZXXnnlE14GgEvJDTfc0I4dO6rasWNHN95445InuviEMZzB40+leNWr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"/>


```python
df['Total'].hist(bins = 50)
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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"/>


```python
df['Type 1'].value_counts(sort = False).sort_index().plot.barh()
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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"/>


```python
df[df['Legendary'] == 1]['Type 1'].value_counts(sort = False).sort_index().plot.barh()
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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"/>


```python
df['Type 2'].value_counts(sort = False).sort_index().plot.barh()
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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"/>


```python
df[df['Legendary'] == 1]['Type 2'].value_counts(sort = False).sort_index().plot.barh()
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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"/>


```python
df['Generation'].value_counts(sort = False).sort_index().plot.barh()
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAWoAAAD4CAYAAADFAawfAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAANG0lEQVR4nO3dW6xcZ3nG8f9T5wAO4ECdUNeOukkVIqFGELSLSNOiEihNIUpVqRepmgpaKt8h6Ik6QqrEXaEVoldUFoeiEoKikNAoEYe0kCKkNul2DjjBSTkZYiepiapuDpYKMW8vZplszBjPLrNmXuL/TxrtmbWWlh/PeD37mzVr/KWqkCT19TPLDiBJ+vEsaklqzqKWpOYsaklqzqKWpObOGGOn27dvr5WVlTF2LUlPS/v27Xuiqs6btm6Uol5ZWWFtbW2MXUvS01KSr51snac+JKk5i1qSmrOoJak5i1qSmrOoJak5i1qSmrOoJam5Ua6j3n94nZU9t4+xa2mqg3/9umVHkEbjiFqSmrOoJak5i1qSmrOoJak5i1qSmpupqJOcm+SmJA8lOZDksrGDSZImZr087++AT1TV7yY5C9g6YiZJ0ganLOokzwFeAbwBoKq+C3x33FiSpONmOfVxIfAN4ANJ7k3y3iTnnLhRkt1J1pKsHTu6PvegknS6mqWozwBeCrynqi4FvgPsOXGjqtpbVatVtbpl67Y5x5Sk09csRX0IOFRVdw2Pb2JS3JKkBThlUVfV48AjSS4eFr0K+MKoqSRJPzDrVR9vAq4frvj4CvCH40WSJG00U1FX1X3A6rhRJEnT+M1ESWrOopak5ixqSWrOopak5kaZiuuSndtYc2okSZoLR9SS1JxFLUnNWdSS1JxFLUnNWdSS1JxFLUnNWdSS1JxFLUnNWdSS1JxFLUnNWdSS1JxFLUnNWdSS1JxFLUnNWdSS1JxFLUnNWdSS1NwoM7zsP7zOyp7bx9i1JLVzcOQZrRxRS1JzFrUkNWdRS1JzFrUkNWdRS1JzM131keQg8C3gGPBkVa2OGUqS9JTNXJ73yqp6YrQkkqSpPPUhSc3NWtQFfCrJviS7p22QZHeStSRrx46uzy+hJJ3mZj31cXlVPZrkfOCOJA9V1Wc3blBVe4G9AGfvuKjmnFOSTlszjair6tHh5xHgFuBlY4aSJD3llEWd5Jwkzz5+H3gN8MDYwSRJE7Oc+ng+cEuS49t/uKo+MWoqSdIPnLKoq+orwIsXkEWSNIWX50lScxa1JDVnUUtSc6PM8HLJzm2sjTzjgSSdLhxRS1JzFrUkNWdRS1JzFrUkNWdRS1JzFrUkNWdRS1JzFrUkNWdRS1JzFrUkNWdRS1JzFrUkNWdRS1JzFrUkNWdRS1JzFrUkNWdRS1Jzo8zwsv/wOit7bh9j15J+QgedfemnjiNqSWrOopak5ixqSWrOopak5ixqSWpu5qJOsiXJvUluGzOQJOmHbWZE/WbgwFhBJEnTzVTUSXYBrwPeO24cSdKJZh1Rvxt4K/D9k22QZHeStSRrx46uzyObJIkZijrJVcCRqtr347arqr1VtVpVq1u2bptbQEk63c0yor4cuDrJQeAjwBVJPjRqKknSD5yyqKvquqraVVUrwDXAp6vq2tGTSZIAr6OWpPY29b/nVdWdwJ2jJJEkTeWIWpKas6glqTmLWpKaG2WGl0t2bmPNWSQkaS4cUUtScxa1JDVnUUtScxa1JDVnUUtScxa1JDVnUUtScxa1JDVnUUtScxa1JDVnUUtScxa1JDVnUUtScxa1JDVnUUtScxa1JDVnUUtSc6PM8LL/8Dore24fY9eStDAHm8xU5YhakpqzqCWpOYtakpqzqCWpOYtakpo7ZVEneUaSu5Pcn+TBJG9fRDBJ0sQsl+f9L3BFVX07yZnA55J8vKr+feRskiRmKOqqKuDbw8Mzh1uNGUqS9JSZzlEn2ZLkPuAIcEdV3TVlm91J1pKsHTu6PueYknT6mqmoq+pYVb0E2AW8LMkvTdlmb1WtVtXqlq3b5hxTkk5fm7rqo6r+B7gTuHKMMJKkHzXLVR/nJTl3uP9M4NXAQyPnkiQNZrnqYwfwwSRbmBT7jVV127ixJEnHzXLVx+eBSxeQRZI0hd9MlKTmLGpJas6ilqTmRpnh5ZKd21hrMjOCJP20c0QtSc1Z1JLUnEUtSc1Z1JLUnEUtSc1Z1JLUnEUtSc1Z1JLUnEUtSc1Z1JLUnEUtSc1Z1JLUnEUtSc1Z1JLUnEUtSc1Z1JLUnEUtSc2NMsPL/sPrrOy5fYxdS08rB50JSTNwRC1JzVnUktScRS1JzVnUktScRS1JzZ2yqJNckOQzSQ4keTDJmxcRTJI0McvleU8Cf1ZV9yR5NrAvyR1V9YWRs0mSmGFEXVWPVdU9w/1vAQeAnWMHkyRNbOocdZIV4FLgrinrdidZS7J27Oj6nOJJkmYu6iTPAj4KvKWqvnni+qraW1WrVbW6Zeu2eWaUpNPaTEWd5EwmJX19Vd08biRJ0kazXPUR4H3Agap61/iRJEkbzTKivhz4A+CKJPcNt9eOnEuSNDjl5XlV9TkgC8giSZrCbyZKUnMWtSQ1Z1FLUnMWtSQ1N8pUXJfs3MaaUwxJ0lw4opak5ixqSWrOopak5ixqSWrOopak5ixqSWrOopak5ixqSWrOopak5ixqSWrOopak5ixqSWrOopak5ixqSWrOopak5ixqSWrOopak5kaZ4WX/4XVW9tw+xq4lqaWDI85q5YhakpqzqCWpOYtakpqzqCWpOYtakpo7ZVEneX+SI0keWEQgSdIPm2VE/Q/AlSPnkCSdxCmLuqo+C/z3ArJIkqaY2znqJLuTrCVZO3Z0fV67laTT3tyKuqr2VtVqVa1u2bptXruVpNOeV31IUnMWtSQ1N8vleTcA/wZcnORQkjeOH0uSdNwp//e8qvq9RQSRJE3nqQ9Jas6ilqTmLGpJam6UGV4u2bmNtRFnO5Ck04kjaklqzqKWpOYsaklqzqKWpOYsaklqzqKWpOYsaklqzqKWpOZSVfPfafIt4OG57/gntx14YtkhpuiaC/pm65oL+mYz1+YtMtsvVNV501aM8s1E4OGqWh1p3/9vSdbMtTlds3XNBX2zmWvzumTz1IckNWdRS1JzYxX13pH2+5My1+Z1zdY1F/TNZq7Na5FtlA8TJUnz46kPSWrOopak5uZa1EmuTPJwki8l2TPPfW8yxwVJPpPkQJIHk7x5WP68JHck+eLw87lLyrclyb1JbmuW69wkNyV5aHjuLuuQLcmfDK/jA0luSPKMZeVK8v4kR5I8sGHZSbMkuW44Hh5O8ptLyPY3w+v5+SS3JDl30dmm5dqw7s+TVJLtXXIledPwZz+Y5J2LzjVVVc3lBmwBvgxcCJwF3A+8aF7732SWHcBLh/vPBv4TeBHwTmDPsHwP8I4l5ftT4MPAbcPjLrk+CPzxcP8s4NxlZwN2Al8Fnjk8vhF4w7JyAa8AXgo8sGHZ1CzDv7n7gbOBFwzHx5YFZ3sNcMZw/x3LyDYt17D8AuCTwNeA7R1yAa8E/hk4e3h8/jJeyx/JOse/9GXAJzc8vg64blF/kVNk+yfgN5h8W3LHsGwHky/mLDrLLuBfgCs2FHWHXM8ZCjEnLF9qtqGoHwGex+QLWrcN5bO0XMDKCQf31CwnHgNDKV22yGwnrPsd4PplZJuWC7gJeDFwcENRLzUXk4HAq6dst/DXcuNtnqc+jh9Qxx0ali1VkhXgUuAu4PlV9RjA8PP8JUR6N/BW4PsblnXIdSHwDeADw2mZ9yY5Z9nZquow8LfA14HHgPWq+tSyc53gZFm6HRN/BHx8uL/UbEmuBg5X1f0nrFr2c/ZC4NeS3JXkX5P8codc8yzqTFm21Gv/kjwL+Cjwlqr65jKzDHmuAo5U1b5lZ5niDCZvA99TVZcC32HyNn6phvO9v83k7ebPA+ckuXa5qWbW5phI8jbgSeD644umbLaQbEm2Am8D/mra6inLFvmcnQE8F3g58BfAjUmy7FzzLOpDTM45HbcLeHSO+9+UJGcyKenrq+rmYfF/JdkxrN8BHFlwrMuBq5McBD4CXJHkQw1yweT1O1RVdw2Pb2JS3MvO9mrgq1X1jar6HnAz8CsNcm10siwtjokkrweuAn6/hvftS872i0x+8d4/HAu7gHuS/NySczH8+TfXxN1M3vluX3aueRb1fwAXJXlBkrOAa4Bb57j/mQ2/Ad8HHKiqd21YdSvw+uH+65mcu16YqrquqnZV1QqT5+fTVXXtsnMN2R4HHkly8bDoVcAXGmT7OvDyJFuH1/VVwIEGuTY6WZZbgWuSnJ3kBcBFwN2LDJbkSuAvgaur6uiGVUvLVlX7q+r8qloZjoVDTD78f3yZuQYfY/L5EUleyORD9SeWnmvOJ+Zfy+QKiy8Db1vUifYpOX6VyduSzwP3DbfXAj/L5IO8Lw4/n7fEjL/OUx8mtsgFvARYG563jzF5C7j0bMDbgYeAB4B/ZPLJ+1JyATcwOVf+PSYF88Yfl4XJW/wvM/nA8beWkO1LTM6tHj8O/n7R2ablOmH9QYYPE5edi0kxf2j4t3YPcMUyXssTb36FXJKa85uJktScRS1JzVnUktScRS1JzVnUktScRS1JzVnUktTc/wGzZAUxmfdKpQAAAABJRU5ErkJggg=="/>


```python
df[df['Legendary'] == 1]['Generation'].value_counts(sort = False).sort_index().plot.barh()
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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"/>


```python
groups = df[df['Legendary'] == 1].groupby('Generation').size()
groups.plot.bar()
```

<pre>
<AxesSubplot:xlabel='Generation'>
</pre>
<img src="data:image/png;base64,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"/>

#### Exploring the distribution of Pokemon abilities



```python
fig = plt.figure(figsize = (12, 12))
ax = fig.gca()
sns.boxplot(x = 'Generation', y = 'Total', data = df, ax = ax)
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
fig = plt.figure(figsize = (12, 12))
ax = fig.gca()
sns.boxplot(x = 'Type 1', y = 'Total', data = df, ax = ax)
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtQAAAK5CAYAAACfR7l0AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABAa0lEQVR4nO3dfZhkZ1kn/u+dTMKQQEIyGZL0QAxDggruijrgqqgQ3KAuGlcFEdYNLr9FZ1nUXRkBdVfWJcI64MuqjEZWiG4iRkUTsgLBhCgIBsJbyBuSachbh2QyCZPXyWQmz++POmM6k+7prj5d1dM9n8919dXVp85Tz91Vp05966nnnKrWWgAAgIU5ZKkLAACA5UygBgCAHgRqAADoQaAGAIAeBGoAAOhh1VIX0Mdxxx3XTj755KUuAwCAFe5Tn/rUHa21tTNdt6wD9cknn5wrrrhiqcsAAGCFq6obZrvOlA8AAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoYaSBuqr+S1VdXVVXVdWfVtXqqjq2qj5UVV/sfh8zbf03VtX1VfWFqnrRKGsDAIDFMLJAXVXrkvxMkg2ttW9IcmiSlyV5Q5JLWmunJrmk+ztV9czu+mcl+d4k76iqQ0dVHwAALIZRT/lYleTxVbUqyRFJppKckeSc7vpzkvxQd/mMJO9prT3YWvtSkuuTPHfE9QEAQC8jC9SttVuSvC3JjUluTbKjtXZxkuNba7d269ya5Mldk3VJbpp2Ezd3yx6lql5dVVdU1RXbtm0bVfkAADAvo5zycUwGo85PSzKR5Miq+nf7azLDsvaYBa2d3Vrb0FrbsHbtjN/+CAAAYzPKKR/fk+RLrbVtrbWHkrw3ybcnua2qTkyS7vft3fo3J3nqtPZPyWCKCAAAHLBGGahvTPKvquqIqqokL0xybZILk5zZrXNmkgu6yxcmeVlVPa6qnpbk1CSfGGF9AADQ26pR3XBr7fKq+oskn06yO8lnkpyd5AlJzq+qV2UQul/SrX91VZ2f5Jpu/de01vaMqj4AAFgM1dpjpikvGxs2bGhXXHHFUpcBAMAKV1Wfaq1tmOk635QIAAA9CNQAANCDQA0AAD0I1AAA0INADQAAPQjUAADQg0ANAAA9CNQAANCDQA0AAD0I1AAA0INADQAAPQjUAADQg0ANAAA9rFrqAgBgvrZs2ZLJyclZr7/llluSJOvWrZvx+vXr12fjxo0jqY3lyTbFYhCoAVgxdu7cudQlsMLYppiPaq0tdQ0LtmHDhnbFFVcsdRkAHCA2bdqUJNm8efMSV8JKYZtir6r6VGttw0zXmUMNAAA9CNQAANCDQA0AAD0I1AAA0INAzbxt3749r3vd63LnnXcudSkAAAcMgZp5O++883LVVVfl3HPPXepSAAAOGAI187J9+/ZcfPHFaa3l4osvNkoNANDxxS7My3nnnZeHH344SfLwww/n3HPPzWtf+9olrgoWZn/fjDbXt6IlvhkNgEczQs28XHrppdm9e3eSZPfu3bn00kuXuCIYjZ07d/pmNACGYoSaeTnttNPygQ98ILt3786qVaty2mmnLXVJsGD7G132rWgADMsINfPy8pe/PIccMthcDjnkkLziFa9Y4ooAAA4MAjXzsmbNmpx++umpqpx++uk59thjl7okAIADgikfzNvLX/7y3HDDDUanAQCmEaiZtzVr1uRtb3vbUpcBAHBAMeUDAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgh1VLXQAsR1u2bMnk5OSs199yyy1JknXr1s14/fr167Nx48aR1AYAjJdADSOwc+fOpS4BABgTgRoWYK7R5U2bNiVJNm/ePI5yAIAlZA41AAD0IFADAEAPAjUAAPQgUAMAQA8CNQAA9CBQAwBADwI1AAD0IFADAEAPAjUAAPQgUAMAQA8CNQAA9CBQAwBADwI1AAD0IFADAEAPAjUAAPQwskBdVV9bVZ+d9nN3Vf1cVR1bVR+qqi92v4+Z1uaNVXV9VX2hql40qtoAAGCxjCxQt9a+0Fp7dmvt2Um+Jcn9Sf4qyRuSXNJaOzXJJd3fqapnJnlZkmcl+d4k76iqQ0dVHwAALIZVY+rnhUm2ttZuqKozkjy/W35OksuSvD7JGUne01p7MMmXqur6JM9N8vEx1QgAC7Jly5ZMTk7OeN0tt9ySJFm3bt2s7devX5+NGzeOpDZYLLbz2Y1rDvXLkvxpd/n41tqtSdL9fnK3fF2Sm6a1ublb9ihV9eqquqKqrti2bdsISwaA/nbu3JmdO3cudRkwUgf7dj7yEeqqOjzJDyZ541yrzrCsPWZBa2cnOTtJNmzY8JjrAWDc9jfqtmnTpiTJ5s2bx1UOjITtfHbjGKH+viSfbq3d1v19W1WdmCTd79u75Tcneeq0dk9JMjWG+gAAYMHGEah/PI9M90iSC5Oc2V0+M8kF05a/rKoeV1VPS3Jqkk+MoT4AAFiwkU75qKojkvzrJD81bfFbk5xfVa9KcmOSlyRJa+3qqjo/yTVJdid5TWttzyjrAwCAvkYaqFtr9ydZs8+y7Rmc9WOm9c9KctYoawIAgMXkmxIBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKCHVUtdwMFsy5YtmZycnPG6W265JUmybt26WduvX78+GzduXPE1wcFgf8+9ZO7nn+ce+7I/f8Rcz6/92bp1a5Jk06ZNC2q/kPvR/mD5EagPUDt37lzqEh7jQKwJDhaefyymg217mpyczBeuvT4nHHPS0G0PefjwJMmOr+wauu1X7rpx6DbzcbA9fsuBQL2E9vfuce874c2bN4+rnCQHZk1wMJhrNMnzj2HZnz/aCceclDP/9S+Ptc9zPvTmBbWzP1h+zKEGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6GHVUhcwDlu2bMnk5OSs199yyy1JknXr1s14/fr167Nx48aR1AYAwPJ2UATquezcuXOpSwAAYJk6KAL1XKPLmzZtSpJs3rx5HOUAALCCmEMNAAA9CNQAANDDQTHlA4DhOaAbYH4EagAWxAHdAAMCNQAzckA3wPyYQw0AAD0I1AAA0INADQAAPQjUAADQg0ANAAA9CNQAANCDQA0AAD0I1AAA0INADQAAPQjUAADQg0ANAAA9CNQAANDDqqUuAFjZtmzZksnJyRmvu+WWW5Ik69atm7X9+vXrs3HjxpHUBgCLQaAGlszOnTuXugQA6E2gBkZqf6PLmzZtSpJs3rx5XOUAwKIzhxoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6GGmgrqonVdVfVNV1VXVtVX1bVR1bVR+qqi92v4+Ztv4bq+r6qvpCVb1olLUBAMBiGPUI9W8n+UBr7euSfGOSa5O8IcklrbVTk1zS/Z2qemaSlyV5VpLvTfKOqjp0xPUBAEAvIzsPdVUdleS7krwySVpru5Lsqqozkjy/W+2cJJcleX2SM5K8p7X2YJIvVdX1SZ6b5OOjqpHlYX/ftJfM/W17vmkPlpe5nvP7s3Xr1iSPnON8WPYXK9PU1FTu2XFfzvnQm8fa71fuuiH3PXzkWPtkaYzyi13WJ9mW5F1V9Y1JPpXkZ5Mc31q7NUlaa7dW1ZO79dcl+cdp7W/ulj1KVb06yauT5KSTThpd9Swbvm0PVpbJyclced11qTVrh27b2uD357dtH77t9m1DtwFIRhuoVyX55iSvba1dXlW/nW56xyxqhmXtMQtaOzvJ2UmyYcOGx1zPyjPXaJFv24OVp9aszeEv/uGx9rnroveOtT/GZ2JiIjsO2ZUz//Uvj7Xfcz705hx9wuFj7ZOlMco51Dcnubm1dnn3919kELBvq6oTk6T7ffu09Z86rf1TkkyNsD4AAOhtZIG6tfaVJDdV1dd2i16Y5JokFyY5s1t2ZpILussXJnlZVT2uqp6W5NQknxhVfQAAsBhGOeUjSV6b5NyqOjzJZJKfzCDEn19Vr0pyY5KXJElr7eqqOj+D0L07yWtaa3tGXB8AAPQy0kDdWvtskg0zXPXCWdY/K8lZo6wJAAAWk29KBACAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHlYtdQEAwMFhy5YtmZycnPX6W265JUmybt26Ga9fv359Nm7cOJLamPvx2Z+tW7cmSTZt2rSg9rM9tnPVNDU1lQceeGBBfSbJ4x//+ExMTAxV00wEagDggLBz586lLuGgNjk5meuv+aec9MSZA+b+HL770CTJrpvuHbrtjfdMzVHTF3LSUU+e8fo99+3Mw3t2D93nP7ffvTO7br7rsTXdfftQtyNQAwBjMddo397Rzc2bN4+jHGZw0hMn8ovfOt5PAX7t8i37vf6ko56cX/r2l42pmoGzPvaeodY3hxoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHpYtdQFAADATKampnLf3ffkrI+9Z6z93nD37Tly6oF5r2+EGgAAejBCDQDAAWliYiK7Hr4rv/TtLxtrv2d97D05fOKYea9vhBoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHpYtdQFAItjy5YtmZycnPX6W265JUmybt26Ga9fv359Nm7cOJLaAGAlE6hHaK6Asz9bt25NkmzatGlB7YUj9rVz586lLgEAViSBeoQmJydz/TVX56Sjjxi67eF7diVJdt3ypaHb3rjj/qHbsPzN9QZq75uzzZs3j6McADhoCNQjdtLRR+SNz3vWWPt8y0evHmt/AAAHMwclAgBADwI1AAD0IFADAEAPAjUAAPQgUAMAQA/O8gEALJql+g4G37/AUhKoAYBFMzk5mWuvuz7Hrjlp6LatHZ4kuW3brqHa3bn9xqH7gsUkUAMAi+rYNSfl+37gl8fW3/vf9+ax9QUzMYcaAAB6EKgBAKAHgRoAAHowhxroZamO6E8c1Q/AgUGgBnqZnJzM56+7MoetGb7t7jb4fd22K4du+9D24fsDgFEQqIHeDluTHHdGjbXPOy5oY+0PAGZjDjUAAPRghBoA5uBYAWB/BGoAmMPk5GSuvO6fcsiaE4Zu+3AbfBh81ba7h2+7/StDtwHGT6AGgHk4ZM0JefwP/ORY+3zgfe8aa3/AwphDDQAAPRihBlYc810BGCeBGlhxBvNdr06OO2IBrXclSa6840vDN73j/gX0B8ByJ1ADK9NxR+TQM75+rF3uueDasfYHLF8+SVtZBGoAgDGbnJzMF6+5Pk896qSh2x62+/Akyc6bdw3d9qa7bxy6DXMTqAEAlsBTjzopP/+tbxxrn2+//C1j7e9g4SwfAADQg0ANAAA9CNQAANCDQA0AAD0I1AAA0MNIz/JRVV9Ock+SPUl2t9Y2VNWxSf4syclJvpzkpa21u7r135jkVd36P9Na++Ao6zsYOe8lAMDiGsdp817QWrtj2t9vSHJJa+2tVfWG7u/XV9Uzk7wsybOSTCT526p6RmttzxhqPGhMTk7mn665MhNH19BtD93TkiT33vL5odtO7WhDtwEAWA6W4jzUZyR5fnf5nCSXJXl9t/w9rbUHk3ypqq5P8twkH1+CGle0iaMrG7/z8LH2ueUjw598HgAYn6mpqdx3z735tcu3jLXfG+6ZypFTTxhrn4tt1HOoW5KLq+pTVfXqbtnxrbVbk6T7/eRu+bokN01re3O37FGq6tVVdUVVXbFt27YRlg4AAHMb9Qj1d7TWpqrqyUk+VFXX7WfdmeYgPGaeQGvt7CRnJ8mGDRvMIwDowXEVwF4TExPZtefe/OK3jvd5+WuXb8nhE8t7hHqkgbq1NtX9vr2q/iqDKRy3VdWJrbVbq+rEJLd3q9+c5KnTmj8lydR8+/KiADC8ycnJXHndtak1xw7dtrXBmMbnt902fNvtdw7dBuBANbJAXVVHJjmktXZPd/n0JL+a5MIkZyZ5a/f7gq7JhUnOq6rfyOCgxFOTfGK+/U1OTub6a67NSUcP/6JweHew3a5bhn9RuHGHFwVgeas1x2bVi1801j53X+QkTn0ZSIIDxyhHqI9P8ldVtbef81prH6iqTyY5v6peleTGJC9Jktba1VV1fpJrkuxO8pphz/Bx0tHH5pe/8/TF/B/m9OaPXDzW/gAgGQwkXXXdF/O4NU+de+V97GqHJUm+uG3n0G0f3H7T3CvBQWZkgbq1NpnkG2dYvj3JC2dpc1aSs0ZVEwCsJI9b89ScdMYvjLXPGy/49bH2B8uBb0oEAIAeBGoAAOhBoAYAgB6W4psSgQVyVD8AHHhmDdRV9c37a9ha+/TilwPsz+TkZK659so8cfizQ2Z39zVIN9125dBt73F2SACY1f5GqN++n+taktMWuRZgHp54bPKc7xtvn598/3j7A4DlZNZA3Vp7wTgLAQCA5Whec6ir6huSPDPJ6r3LWmt/PKqiAABguZgzUFfVryR5fgaB+m+SfF+SjyYRqAFYdFNTU2l3351dF713rP227dsy9dCDY+0TWBnmc9q8H83gmw2/0lr7yQy+/fBxI60KAACWiflM+XigtfZwVe2uqqOS3J5k/YjrAlhxluq0h8vtlIcTExPZftjjcviLf3is/e666L2ZWLtmrH0CK8N8AvUVVfWkJH+Y5FNJ7k3yiVEWtVJMTU3lvh335y0fvXqs/d6w4/4cWVNj7XMlcs5nFtvk5GSuvO6aZM3Rwzdue5IkV267Zbh223cM3xcAQ5kzULfW/lN38fer6gNJjmqtDX8iW1hmJicnc+21V+aYY4Zv+/DDg99f+crwT5W77hq+P5aRNUdn1Q8+b2zd7b7wo2PrC+BgNZ+DEi9prb0wSVprX953GbObmJjIrvZg3vi8Z42137d89OocPjEx1j5XqmOOSU4f85Z+8SXj7Q8A6Gd/35S4OskRSY6rqmOSVHfVUUmkNQAAyP5HqH8qyc9lEJ6nf8343Ul+b4Q1AcvI1NRUHro7ueOCNtZ+H9qeTD3kWAEAlt7+vinxt5P8dlW9trX2O2OsCQAAlo35nOXjD6rqZ5J8V/f3ZUn+oLX20MiqApaNiYmJ3H3YHTnujJp75UV0xwUtE2vNPgNg6c0nUL8jyWHd7yT5iSRbkvx/oyqKg4/z8wIAy9X+Dkpc1VrbneQ5rbVvnHbVpVX1udGXxsFkcnIyX7j2yqx90vBtqztF3Z23DneKum1fHb4vAIB97W+E+hNJvjnJnqp6emtta5JU1foke8ZRHAeXtU9KXvqCQ8fW3/kfthkDAP3tL1DvnRD5uiQfrqq9n8efnOQnR1kUAAAsF/sL1Gur6r92l/8gyaFJ7kuyOsk3JfnwiGsDAIAD3v4C9aFJnpBHRqrT/Z0kTxxZRQAAsIzsL1Df2lr71bFVAgAAy9B85lADwEFtamoqD999Tx5437vG2u/D22/N1EP3jrXPvqamprLj7vvy/ve9eWx93rn9hux56Mix9Qf7OmQ/171wbFUAAMAytb+vHr9znIUAwIFqYmIidx52dx7/A+M9ydUD73tXJtYeNdY++5qYmMihh+3K9/3AL4+tz/e/7805fu3hY+tvMUxNTeW+u+/L2y9/y1j7venuG3LklNH8xTafb0oEAA4wU1NTefDu+3LjBb8+1n4f3H5TpkyvgEcRqAEAxmxiYiI7H96Vn//WN46137df/pasnlheo/nLgUANAMvQxMRE7jtsZ0464xfG2u+NF/x6JtauHmufHNxuvPv2nPWx98x43W333ZWdex5a8G2vPvSwHH/kMTP2eUoeu3w2AjUAAAek9evX7/f6Q6ceyCEPtAXf/qGPX53DJx4bnE/JMXP2PZ1ADQDAAWnjxo1LXcK87O+0eQAAwBwEagAA6EGgBgCAHgRqAADowUGJB5mpqancu6Nly0d2jbffHS1PqKmx9gnMbWpqKu3uHdl90QfH2m/bfmemHtoz1j4BRsUINQAA9GCE+iAzMTGRe9v2bPzO8X5L0paP7MoTJibG2icwt4mJiWw/7NCsevGLxtrv7os+mIm1x4+1T4BRMUINAAA9CNQAANCDQA0AAD0I1AAA0INADQAAPQjUAADQg0ANAAA9CNQAANCDQA0AAD0I1AAA0INADQAAPQjUAADQw6qlLgBgsU1NTSV33589F1w73o7vuD9Tu6bG2ycAS84INQAA9GCEGlhxJiYmcsfhD+bQM75+rP3uueDaTBw3MdY+AVh6RqgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKCHFXMe6qmpqdy3Y0fe/JGLx9rvDTvuzJG1Z6x9AgBw4DBCDQAAPayYEeqJiYnsaofml7/z9LH2++aPXJzDJ44fa58AABw4jFADAEAPAjUAAPSwYqZ8AADQz433TOXXLt8ydLvb7r8jSXL8EcctqM9T8oyh2x1IBGoAALJ+/foFt9219bYkyeFPfcLQbU/JM3r1fSAQqAEAyMaNGxfcdtOmTUmSzZs3L1Y5y4o51AAA0INADQAAPZjyAQCseF+568ac86E3D93uznsGc4OPfeLw3znxlbtuzNEnnDJ0O5YfgRoAWNH6HPB2x327kiRHn3D40G2PPuGUZX+wHfMjUAMAK5qD7Rg1c6gBAKAHgRoAAHow5QNmMTU1lR07kosvGW+/d92VPPzw1Hg7BQAWzAg1AAD0YIQaZjExMZFDDrkjp79wvP1efElywgkT4+0UAFgwI9QAANCDQA0AAD0I1AAA0INADQAAPQjUAADQg0ANAAA9CNQAANCD81ADjMnU1FRy947svvCj4+t0+45MPVTj6w/gIGSEGgAAehj5CHVVHZrkiiS3tNZeXFXHJvmzJCcn+XKSl7bW7urWfWOSVyXZk+RnWmsfHHV9AOMyMTGROw5rWfWDzxtbn7sv/Ggm1vrmTYBRGseUj59Ncm2So7q/35DkktbaW6vqDd3fr6+qZyZ5WZJnJZlI8rdV9YzW2p4x1AjLwtTUVO7ZkXzy/ePt9547k6k9U+PtFACWiZFO+aiqpyT5N0neOW3xGUnO6S6fk+SHpi1/T2vtwdbal5Jcn+S5o6wPAAD6GvUI9W8l+YUkT5y27PjW2q1J0lq7taqe3C1fl+Qfp613c7cM6ExMTGTPoXfkOd833n4/+f5k4njTBgBgJiMboa6qFye5vbX2qfk2mWFZm+F2X11VV1TVFdu2betVIwAA9DXKKR/fkeQHq+rLSd6T5LSq+r9JbquqE5Ok+317t/7NSZ46rf1Tkjxm0mZr7ezW2obW2oa1a9eOsHwAAJjbyAJ1a+2NrbWntNZOzuBgw0tba/8uyYVJzuxWOzPJBd3lC5O8rKoeV1VPS3Jqkk+Mqj4AAFgMS/HFLm9Ncn5VvSrJjUlekiSttaur6vwk1yTZneQ1zvABAMCBbiyBurV2WZLLusvbk7xwlvXOSnLWOGriwDI1NZW7dyTnf3h876Fu/2qyszkVHADQj29KBACAHpZiygc8xsTERFbXHXnpCw4dW5/nf3hPjj3RqeAAgH6MUAMAQA8CNQAA9CBQAwBADwI1AAD0IFADAEAPAjUAAPQgUAMAQA8CNQAA9CBQAwBADwI1AAD0IFADAEAPAjUAAPQgUAMAQA8CNQAA9CBQAwBADwI1AAD0IFADAEAPAjUAAPQgUAMAQA8CNQAA9CBQAwBADwI1AAD0IFADAEAPq5a6gJXuxh335y0fvXrodrfdtzNJcvyRqxfU5ynrhm4GAMACCNQjtH79+gW33bV1a5Lk8HVPG7rtKev69Q0AwPwJ1CO0cePGBbfdtGlTkmTz5s2LVQ4AACNgDjUAAPQgUAMAQA8CNQAA9GAONdDbQ9uTOy5oQ7fbvWPwe9XRC+sza4dvBwCLTaAGeulzRpmtdw/OZvP0tU8fvvFaZ7MB4MAgUAO9OJsNB4uHt38lD7zvXcO327E9SXLI0WsW1GfWHjV0O2C8BGoAmEO/T2K2JUmevpBgvPYon8SsYDfdfWPefvlbhm53+323JUmefOTxC+rz1JwydDv2T6AGgDn4JIbF1ueN0kNbdyVJVj/l8KHbnppTvEkbAYEaAGDMvElbWZw2DwAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgh1VLXQAA7Ktt35ZdF713+HY7vpokqaOftKA+s3bN0O0ABGoADijr169fcNutd381SfL0hQTjtWt69b0UHtx+U2684NeHbrdrx+1JksOPfvKC+szaU4duByuZQA3AAWXjxo0Lbrtp06YkyebNmxernANWvzceDyVJnr529fCN15667N54wKgJ1ACwDHnjAQcOByUCAEAPAjUAAPRgygewMt1xf/ZccO3w7XbsHPw+egFzS++4Pzlu+GYALG8C9UFoakfLlo/sGrrdHfe1JMlxR9aC+nzGuqGbwYL0Olhrx9YkydOPe9rwjY/r1zcAy5NAfZDp82J/29ZB0HjCuqcP3fYZ6wQNxsfBWgCMk0B9kBE0gH217Xdm90UfHL7djnuSJHX0ExfUZ9YeP3Q7gAORQA1wEOt3LuN7kyRPX0gwXnu8T62AFUOgBjiI+dQKoD+BGpaZe+5MPvn+4dvdP/h0PkcM/+l87rkziU/nAWBGAjUsI70+nr93cFDpU48f/qDSHO+g0kWzfUd2X/jR4dvtuG/w++gjh+4va51iB2CUBGpYRnw8v7z1m6/cnc5v2HC8dp03QwAjJlADjIk3RAArk68eBwCAHlbUCPWNO+7Mmz9y8dDtbrtvcLTW8UcOf7TWjTvuzCnrHK0FAHCwWjGBus8cwV1bB+dSPXwBwfiUdc6lCgBwMFsxgdrcRAAAloI51AAA0INADQAAPQjUAADQg0ANAAA9CNQAANCDQA0AAD0I1AAA0INADQAAPQjUAADQg0ANAAA9CNQAANDDqqUuAABYWe7cfmPe/743D93unh23JUmeePTxQ/d3/NpThu4PFotADQAsmvXr1y+47b1370qSHL/28KHaHb/2lF79Ql8jC9RVtTrJ3yd5XNfPX7TWfqWqjk3yZ0lOTvLlJC9trd3VtXljklcl2ZPkZ1prHxxVfTAfd92VXHzJ8O3uuWfw+4lPXFifJ5wwfDuAA8HGjRsX3HbTpk1Jks2bNy9WOTAWoxyhfjDJaa21e6vqsCQfrar3J/nhJJe01t5aVW9I8oYkr6+qZyZ5WZJnJZlI8rdV9YzW2p4R1giz6jPacd99W5MkJ5zw9KHbnnBCv74BgPEaWaBurbUk93Z/Htb9tCRnJHl+t/ycJJcleX23/D2ttQeTfKmqrk/y3CQfH1WNsD9GWQCA+RjpHOqqOjTJp5KckuT3WmuXV9XxrbVbk6S1dmtVPblbfV2Sf5zW/OZuGQDASGzZsiWTk5OzXr916+ATx70DJftav359rwGY5WR/99Vc91Oysu+rkQbqbrrGs6vqSUn+qqq+YT+r10w38ZiVql6d5NVJctJJJy1GmQAAM1q9evVSl7AsHOz301jO8tFa+2pVXZbke5PcVlUndqPTJya5vVvt5iRPndbsKUmmZrits5OcnSQbNmx4TOAGAJivlTpiOgruq9mN7ItdqmptNzKdqnp8ku9Jcl2SC5Oc2a12ZpILussXJnlZVT2uqp6W5NQknxhVfQAAsBhGOUJ9YpJzunnUhyQ5v7V2UVV9PMn5VfWqJDcmeUmStNaurqrzk1yTZHeS1zjDBwAAB7pRnuXjyiTfNMPy7UleOEubs5KcNaqaAABgsY1sygcAABwMBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeVi11AQCw3G3ZsiWTk5MzXrd169YkyaZNm2Ztv379+mzcuHEktQGjJ1ADwAitXr16qUsARkygBoCejC7Dwc0cagAA6EGgBgCAHkz5AIAVyIGSMD4CNQAcZBwoCYtLoAaAFcjoMoyPOdQAANCDQA0AAD0I1AAA0IM51Bwwtn01Of/De4Zu99V7B7+f9ITh+zv2xKG7AwB4FIGaA8L69esX3Pau7vRPx5749KHaHXtiv34BABKBmgNEn6PR955HdfPmzYtVDgDAvJlDDQAAPRihBg46vkEOgMUkUANM4xvkABiWQA0cdIwuA7CYzKEGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoQaAGAIAeBGoAAOhBoAYAgB4EagAA6EGgBgCAHgRqAADoYdVSF3Aw27JlSyYnJ2e8buvWrUmSTZs2zdp+/fr12bhx40hqA4DFtr/XvWTu1z6vexyoBOoD1OrVq5e6BAAYK699LFcC9RLyLhuAg4nXPcZp+/btectb3pJf/MVfzLHHHjvSvsyhBgBgxTnvvPNy1VVX5dxzzx15XwI1AAAryvbt23PxxRentZaLL744d95550j7M+UDVggH+wDAwHnnnZeHH344SfLwww/n3HPPzWtf+9qR9WeEGg4Sq1evdsAPAAeFSy+9NLt3706S7N69O5deeulI+zNCDSuE0WUAGDjttNPygQ98ILt3786qVaty2mmnjbQ/I9QAAKwoL3/5y3PIIYOYe8ghh+QVr3jFSPsTqAEAWFHWrFmT008/PVWV008/feSnzTPlAwCAFeflL395brjhhpGPTicCNQAAK9CaNWvytre9bSx9mfIBAAA9CNQAANCDQA0AAD0I1AAA0INADQAAPQjUAADQg0ANAAA9CNQAANCDQA0AAD0I1AAA0INADQAAPQjUAADQg0ANAAA9CNQAANCDQA0AAD0I1AAA0INADQAAPQjUAADQw6qlLgAAgEds2bIlk5OTs16/devWJMmmTZtmvH79+vXZuHHjSGpjZgI1AMAysnr16qUugX0I1AAHACNSwF6ey8uPQA2wDBiRAjhwCdQABwAjUgDLl7N8AABADwI1AAD0IFADAEAPAjUAAPTgoEQAZuRUfgDzI1ADsCBO5QcwIFADMCOjywDzYw41AAD0IFADAEAPpnwAI7W/A9vmOqgtcWAbAAc+gRpYMg5qA2AlEKiBkTK6DMBKN7I51FX11Kr6cFVdW1VXV9XPdsuPraoPVdUXu9/HTGvzxqq6vqq+UFUvGlVtAACwWEZ5UOLuJD/fWvv6JP8qyWuq6plJ3pDkktbaqUku6f5Od93LkjwryfcmeUdVHTrC+gAAoLeRTflord2a5Nbu8j1VdW2SdUnOSPL8brVzklyW5PXd8ve01h5M8qWquj7Jc5N8fFQ18lgOIAMAGM5YTptXVScn+aYklyc5vgvbe0P3k7vV1iW5aVqzm7tl+97Wq6vqiqq6Ytu2bSOtm0dbvXq1g8gAAPYx8oMSq+oJSf4yyc+11u6uqllXnWFZe8yC1s5OcnaSbNiw4THX04/RZeBAtr9P0ZK5P0nzKRowCiMdoa6qwzII0+e21t7bLb6tqk7srj8xye3d8puTPHVa86ckmRplfQCsLD5JA5bCyEaoazAU/X+SXNta+41pV12Y5Mwkb+1+XzBt+XlV9RtJJpKcmuQTo6oPgOXH6DJwIBrllI/vSPITST5fVZ/tlv1iBkH6/Kp6VZIbk7wkSVprV1fV+UmuyeAMIa9pre0ZYX0AANDbKM/y8dHMPC86SV44S5uzkpw1qppYnsyZBAAOZL4pkWXPfEkAYCkJ1BzwjC4DAAeysZyHGgAAVioj1LAA5nUDAHsJ1DAC5nUDwMFDoIYFMLoMAOxlDjUAAPQgUAMAQA8CNQAA9CBQAwBADwI1AAD0IFADAEAPAjUAAPRwUJyH2rfaAQAwKgdFoJ6Lb7UDAGChDopAbXQZAIBRMYcaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoAeBGgAAehCoAQCgB4EaAAB6EKgBAKAHgRoAAHoQqAEAoIdqrS11DQtWVduS3LBIN3dckjsW6bYWi5rm70CsS03zo6b5OxDrUtP8qGn+DsS61DQ/K72mr2mtrZ3pimUdqBdTVV3RWtuw1HVMp6b5OxDrUtP8qGn+DsS61DQ/apq/A7EuNc3PwVyTKR8AANCDQA0AAD0I1I84e6kLmIGa5u9ArEtN86Om+TsQ61LT/Khp/g7EutQ0PwdtTeZQAwBAD0aoAQCgB4EaAAB6WJGBuqqOr6rzqmqyqj5VVR+vqn+71HXtq6r2VNVnp/2cXFUfG1Pfv1lVPzft7w9W1Tun/f32qvqvs7R9ZVVNjKHMvf3tvZ8+V1WfrqpvH0OfrarePu3v11XVm0bd7z41XFZVjznVz7T746qq+vOqOmI/t/GDVfWGEda47zb8hv3VPo/be3ZVff9+rt9QVf97jttY0ud/9zy+aoj1H7Mf2M+6i7J/WOi+p6q+XFXHzbD8+dOfl1X101X173vUNee2PeTtvruqfnSG5RNV9Rc9bnd6ve+rqict4DaeX1UXLbSGabcz43bf7a9/dxFuf1H3+1V172Ld1hz97H2Mru5eQ/5rVR2w2Wcx6h12HzSP2/ulrp4ru9q+tap+rs/zc7bn5BxtZtpvzfn8Wej+aFirRt3BuFVVJfnrJOe01l7eLfuaJD+4z3qrWmu7x1/hozzQWnv2PsseExar6tDW2p5F7vtjSV6S5Le6J+txSY7ap46fm6XtK5NclWRqvp31vL//+X6qqhcleUuS717gbc3Xg0l+uKre0lob+oTwI96+pt8f5yb56SS/MdOKrbULk1w4ojoeVcsieXaSDUn+Zt8ruvv0iiRXzNZ4mT3/95r3fdhaW6z9w7z2PUN4fpJ7M9ivpLX2+wu8nXlv24uhtTaVZKgX9X1Mr/ecJK9JctYilDaUObb7exapm1dmyP3+AWL6Y/TkJOclOTrJr0xf6QDaJ8yr3tlU1aGLWUxVfVuSFyf55tbag90b6sOT/FmS/5vk/sXsbw6P2W/tbwBirx77o6EcsO/Sejgtya7pd2Br7YbW2u9077D/vKrel+TiqnpCVV1Sg1HPz1fVGUlSVUdW1f/r3h1eVVU/1i1/a1Vd071Le9soit/7rr171/Xhqjovyeer6tCq2lxVn+z6/6meXf1DHnkBfVYGO8p7quqYqnpckq9P8qKuv6uq6uwa+NEMAs+53TvEx1fVt1TV33WjIh+sqhO7/+Gyqvq1qvq7JD/bs969jkpyV3f7j3pnWlW/W1Wv7C5/f1VdV1Ufrar/Pdc72BnszuDI4P+y7xVV9TXddnNl9/ukbvm7q+o3qurDSf5X9/eW7nGcrKrvrqo/qqprq+rd025vS1Vd0Y0A/I8h6/xIklOq6tiq+uuupn+sqn/Z3fY/j05V1Uu6x/JzVfX33bLVVfWubvv/TFW9YFq791bVB6rqi1X160PWNf3+Or0Go2Wf7p5/T+iWP6eqPtbV84mqOjrJryb5sW7b+rGqelO37V2c5I+nP+bd83dv7VdW1Y9kuOf/bPfZm6rqddPqv6oGIyEnd4/dH3aP1cVV9fhunW/p/o+PZxCqFqxm2S911822f/ifVfWz09Y7q6p+Zsh+9972IVX1ju5/vKiq/qYePZL02mm1fV0NXtB+Osl/6R6375x+H3b7gf/VPcb/VFXf2S0/oqrO7+7/P6uqy/Po16S92/aJVfX39chI8HdW1auq6jen1f4fq+o3usv/vrvNz1XVn0y7ve/qtrfJvf9PTRvJq8E+9m3TtqfXDnP/Jfl4knXdbT2726aurKq/qqpjuuWnVNXf1iOftj19n8fgOd3zcP2Qfc+63Xd/Tsz0XK6qH+/+36uq6n9Nux/e3S37fFX9l5phvz9kfftVVb/Q9fW5qnprt+zpXc2fqqqPVNXX9e2ntXZ7klcn+c81MK9M0NXz32rwmvKhqvrTadv3bI/1jNt9z3pP7u6LT9e0T2trn/3B9NuoqvXdNvWcBd5tJya5o7X2YFfTHRm8CZ1I8uEavN7tbz8/Yz5YbDXYb32xqtZO+/v6qjquFrg/qmE/ZW2traifJD+T5Ddnue6VSW5Ocmz396okR3WXj0tyfZJK8iNJ/nBau6OTHJvkC3nkzChPWoRa9yT5bPfzV92ye7vfz09yX5KndX+/Oskvd5cfl8Eo3dN69v/lJCcl+akMXhD/Z5LvT/IdSf5+7/3UrfsnSX6gu3xZkg3d5cMyGJVa2/39Y0n+aNp671jE++m6JDuSfMu0++iiaev9bvcYr05y07T77k+nrzfPPu/NILx/uXv8X5fkTd1170tyZnf5PyT56+7yu5NclOTQaX+/p9umzkhyd5J/kUFo+FSSZ3fr7d0eD+3us3+57/28b23Ttt8LkmxM8jtJfqVbflqSz07b5n+3u/z5JOumb79Jfj7Ju7rLX5fkxu7+e2WSye5/X53khiRPnWMb/mySH5teewbPq79PcmS3/PVJ/nsGIxyTSZ7TLT+q+3/+ud5u+Zu6++rx+z7mSf5Xkt+atu4xGe75P9t99qYkr5vW7qokJ3c/u6c9bucn+Xfd5SuTfHd3eXOSqxa6H8gs+6U59g8nJ/l0d/mQJFuTrFngvudHM/iE4JAkJ2TwBvZHp+0zXttd/k9J3jnLffbPf3fbwtu7y9+f5G+7y69L8gfd5W/o7tv7Z9i2fz7JL017jjwxyZHd/3hYt/xjGTy3npXBfvq4fZ5b707y593/9Mwk10+7367qLm9M8pdJVk1vO9d+Ylpdf57ke2fYHn413Xaa5PIk/7a7vDrJEd1jeVEGAxyfSnLSMPuq7rbm2u4f81zOIBDdmGRtd39fmuSHknxLkg9Na793X3FZZtgfLfRn2n33fd3jd8Q+j9klSU7tLn9rkkv79LPPsruSHJ/5Z4INGTxXHt9tf1/MI9v3bI/1ZZlhu+9Z7xFJVnfLTk1yxX72B1cl+dokn0m3z1rg/feE7n//pyTvmPa/fjmPPM9m28/vLx+8O91+ZYhaZtpvPT+PvCb8SpKf6y6fnuQvu8tvysL2R0Nt7ytuyse+qur3kjwvya4kv5fBjuLOvVcn+bWq+q4kD2cwunB8BsHjbd079otaax+pqlVJdiZ5Z1X9vwx2gH3N9VHvJ1prX+oun57kX9YjI0VHZ/CE+tKMLedn7yj1t2fwseq67vKODJ4EL6iqX8jgSXxskqszCJPTfW0GG9+HqioZvLDcOu36P+tR317TPwL7tgxGKr9hP+t/XZLJaffdn2bwhmQorbW7q+qPM3ixemDaVd+W5Ie7y3+SZPro7Z+3R3/8/r7WWquqzye5rbX2+e7/uDqDnd5nk7y0ql6dwc78xAxe8K/cT2mPr6rPdpc/kuT/ZPBC/SNd3ZdW1ZoajPhO9w9J3l1V5yd5b7fseRkEy7TWrquqG5I8o7vuktbajq7ea5J8TQZvVKabaxv+V93/8w/d9nF4BiN5X5vk1tbaJ7u+7+76mek2LmytPTDD8u9J8rK9f7TW7tq3/RzP/+dl7vtsX19qrX22u/ypJCd3bZ7UWvu7bvmfZBAS5utR92FVHZaZ90tf2afdP+8fWmtfrqrtVfVN3bqfaa1tn2+f+3heBtvxw0m+sncEapq9286n8sjzYC7T25w8rZ/f7uq/qqquTPKNM2zb/yrJH3X3y1/vvf+r6tIkL66qazMI1p+vwajyX7Rumta0xzpd24eTXFNVx89Q4/ck+f3Wfey/T9vZ7H0untz9bx+aYXs4J8mfV9UTM3hD+1fd7e/s/o9k8Ing2UlOb4NpKL3MsN3P9Fxek+Sy1tq2bvm5Sb4rg4GV9VX1O0n+X5KL+9Yzh+/J4E39/cngfu9GN789g/tt73qPW8Q+p+8o5pMJnpfkgr37oW5EO7M91tNue6btvk+9hyX53ap6dgbh8hnT1pmeF5LBG6ULkvxIa+3qhXbcWru3qr4lyXcmeUGSP6vHHpezv/38/vLBsOZ6vfmjDP7n38pgsOtds6w33/3RUFZioL463YtkkrTWXlODOT97513eN23dV2Sw0X1La+2hqvpyBu/+/qnbgL4/yVuq6uLW2q9W1XOTvDCDF/H/nMGo1ihNr7UyGBn64CLe/scy2Gn9iwzezd6UwWjQ3RlsmO/M4B3aTTU4IG/1DLdRSa5urX3bLH3cN8vyBWmtfbx7PNdm8A5y+kfEe+ubMZUt0G8l+XRmf2ImSZt2ed//98Hu98PTLu/9e1VVPS2Dd8bP6QLhuzPz/TzdTPPIZvqf26P+aO2nq+pbk/ybJJ/tdsr7u6+m17snC9tfVAYvWD++T73/ct/69mO2bahmuI1hnv+z3WezbVfJY++Tx89SRx8z7pdmWG/f++WdGYy4nZDB83eh5nr+7L0PhtkmZmozUz8PzvCC+fddwPk3Sf6kqja31v44g//3FzP45Grv83N/j8X0x26mvhfyOD7QWnt2F6wuymC6zzmzrLu/+/XWDB7jb8rC5ijPtd3P9FyesZ5uP/SNSV6Uwf/z0gzCyajMdL8fkuSrc4SnhXU2mE6zJ8nt3aI5M0EW/pqykOfKo+xT768kuS3JN2ZwH+2ctuq++4MdGbymf0cG28eCdYNElyW5rBscOnPfMjPzfv5fZP/5YFF1WeW2qjotg081XjHLqvPdHw1lJc6hvjTJ6qraOG3ZbEeiHp3k9u6J84IM3rWnBkcy399a+79J3pbkm7t3zEe31v4mg4P1nj2i+mfzwSQbu1GaVNUzqurInrf5DxkcbHBna21P9y79SRmMwH68W+eO7n+fPofyngw+9koGH6+u7UaOU1WHVdWzetY1qxrMozs0yfYMPrp8ZlU9rntBe2G32nUZjLCc3P39Ywvtr7tPzk/yqmmLP5ZHRkZfkeSjC739DKY63JdkRzdqNszI5nR/39WSqnp+BnPe7p6+QlU9vbV2eWvtvye5I4OPfae3e0YGU4C+sMAaZvKPSb6jqk7p+jii6+e6DOZ1Pqdb/sTuU6Dp29ZcLs7gjW262zgmwz3/Z7vPvpzkm7vl35zkafsrorX21Qwev+d1i2bbic/XjPulefirJN+b5DkZ7C8W6qNJfqQGcxCPz+Aj1bkM87hN7+elSVJVz8zgjf1j1ODguttba3+YwYj1NydJa+3yDLbhl2fwKVQymCbw0qpa07U9doh6Lk7y0912OFTbbvT3ZzJ4c3x/krvqkTmzP5Hk77pt6+aq+qHu9h9Xj5wl4asZvGH4tW5bHNYw2/1elyf57hrMMT00yY8n+bsuiB/SWvvLJP8t3f2dhT3G83Fxkv+w976oqmO7++pLVfWSbll1Ib+XGsyv/f0MppXN9OZptufeR5P8QA2OOXlCBo/V3sf9MY913zr3U+/RGXyy93DX1/4OQNyVwRSef19VL+9Rw9dW1anTFj07g9fe6dvDbPv5seaDzjszOFjy/Dbcwdrz2h/tz4oboe4+Xv+hJL9Zg+kK2zIILK/PYDRpunOTvK+qrsgjc3STwR25uaoeTvJQBnPrnpjkgqra+271MQerjdg7082T7EYjt2XwZOnj8xnMfTpvn2VPaK3dUVV/2P395SSfnLbOu5P8flU9kEH4/tEk/7sLtasyGNXt9Y54H9OnOFQG85f3JLmpBtMXrsxgTttnkqS19kBV/ackH6iqO5J8omf/b8+04JbBC+cfVdWmDB6Hn1zoDbfWPldVn8ng/prM4E3OQrwpybu6j6nuz2NHEJLBNn1qBvfhJUk+l8E2//vdqMPuJK9sgyO559vv9McmST7QWvvnjwNba9tqcKDon9bgYNdkcCzAP9XgYN/fqcEBTg9k8NHvh5O8obvNt8zR95uT/F4NDirbk+R/tNbeO8Tz/02Z+T77ywxehD6bwXb/T/O4H34yg23i/vQLs8ns+6X9aq3tqsH0jK8O+UKyr7/M4M3pVRn875dnMNq1P+9L8hc1OIhrvgfzvSPJOd39/5kMnsczHXj2/CSbquqhDI5tmH76q/MzmB96V5K01q6uqrMyCIZ7utt95TzreWcGH6Ff2fX1hxkclzEvrbXPVNXnMnizfWYGz6sjMnhe791H/ESSP6iqX83gteUl09rfVlU/kOT9VfUfujcM8+17mNe9vW1urao3ZvCcqyR/01q7oAuu76pHTtX2xu73uzNtvz/LNKyhtdY+UINPy66oql0ZzN//xQzemG6pql/OYKrDezLYZw1r7z7qsAz2cX+S2c8cM+Nzr7X2yaq6sOv/hgxG/vc+J2Z7rBdqf/W+I8lfdm80Ppw5PgFurd1XVS/OYMrFfa21CxZQzxMy2E8/qavn+gymUP54Btvqra21F+xnPz/qfLCvCzP4xGp/nyrPZKb90Vz7vUfx1eOsSFX1hG7uV2Uwh/CLrbXfXOq6YFS6APTpJC9prX2x523tff6syeAN6Xe01vadw91LNyp6WGttZw3OdnFJkme01nYNcRsXZXAw3iWLWRvsa9pz4ogMPt16dWvt00tdF49WgzNz/GZrbaizqizG/mjFjVBD5z9W1ZkZHBzxmSR/sMT1wMh0H1FelMGR773CdOeibkTq8CT/c7HDdOeIDE67dVgGI6Qb5/vi1dX2iSSfE6YZk7O759nqDM73LUwfYGpwsOTGLGza3YL3R//cvxFqAABYuJV4UCIAAIyNQA0AAD0I1AAA0INADbDM1OBbHT/b/Xylqm6Z9vfhi9zXB6rqq90ZNQCYgYMSAZaxGnyL6b2ttbeN6PZfmMER8D/VWnvxKPoAWO6MUAMsf4+vqi/VI9+kelRVfbn7ZrLLquq3qupjVXVVVT23W+fIqvqjqvpkVX2m+1KWx+hOS3fPGP8XgGVHoAZY/h5Iclm6r0TO4Nv6/rK19lD395GttW9P8p+S/FG37JeSXNpae06SF2TwTZpHjq9kgJVDoAZYGd6ZR772+Cfz6K/e/dMkaa39fZKjui9GOT2PfM37ZRl8YcVJY6oVYEXxTYkAK0Br7R+q6uSq+u4kh7bWrpp+9b6rZ/BtYD/SWvvC2IoEWKGMUAOsHH+cwWj0u/ZZ/mNJUlXPS7KjtbYjyQeTvLaqqrvum8ZZKMBKIlADrBznJjkm3RSPae6qqo8l+f0kr+qW/c8khyW5sqqu6v5+jKr6SJI/T/LCqrq5ql40ksoBljGnzQNYIarqR5Oc0Vr7iWnLLkvyutbaFUtWGMAKZw41wApQVb+T5PuSfP9S1wJwsDFCDQAAPZhDDQAAPQjUAADQg0ANAAA9CNQAANCDQA0AAD38/+dLJcaoU1HMAAAAAElFTkSuQmCC"/>

-----


## 3) Supervised learning-based classification analysis


### 3-1) Data preprocessing


##### Change data type



```python
df['Legendary'] = df['Legendary'].astype(int)
df['Generation'] = df['Generation'].astype(str)
preprocessed_df = df[['Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']]
```


```python
preprocessed_df.head()
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
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


-----


##### one-hot encoding



```python
encoded_df = pd.get_dummies(preprocessed_df['Type 1'])
encoded_df.head()
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
      <th>Bug</th>
      <th>Dark</th>
      <th>Dragon</th>
      <th>Electric</th>
      <th>Fairy</th>
      <th>Fighting</th>
      <th>Fire</th>
      <th>Flying</th>
      <th>Ghost</th>
      <th>Grass</th>
      <th>Ground</th>
      <th>Ice</th>
      <th>Normal</th>
      <th>Poison</th>
      <th>Psychic</th>
      <th>Rock</th>
      <th>Steel</th>
      <th>Water</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# make type list
def make_list(x1 ,x2):
    type_list = []
    type_list.append(x1)
    if x2 is not np.nan:
        type_list.append(x2)
    return type_list

preprocessed_df['Type'] = preprocessed_df.apply(lambda x : make_list(x['Type 1'], x['Type 2']), axis = 1)
preprocessed_df.head()
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
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>0</td>
      <td>[Grass, Poison]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>[Grass, Poison]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>[Grass, Poison]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>[Grass, Poison]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>0</td>
      <td>[Fire]</td>
    </tr>
  </tbody>
</table>
</div>



```python
del preprocessed_df['Type 1']
del preprocessed_df['Type 2']
```


```python
preprocessed_df.head()
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
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>0</td>
      <td>[Grass, Poison]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>[Grass, Poison]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>[Grass, Poison]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>[Grass, Poison]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>0</td>
      <td>[Fire]</td>
    </tr>
  </tbody>
</table>
</div>


----



```python
from sklearn.preprocessing import MultiLabelBinarizer
```


```python
mlb = MultiLabelBinarizer()
preprocessed_df = preprocessed_df.join(pd.DataFrame(mlb.fit_transform(preprocessed_df.pop('Type')), columns = mlb.classes_))
```


```python
preprocessed_df.head()
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
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
      <th>Bug</th>
      <th>...</th>
      <th>Ghost</th>
      <th>Grass</th>
      <th>Ground</th>
      <th>Ice</th>
      <th>Normal</th>
      <th>Poison</th>
      <th>Psychic</th>
      <th>Rock</th>
      <th>Steel</th>
      <th>Water</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



```python
preprocessed_df = pd.get_dummies(preprocessed_df)
preprocessed_df.head()
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
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Legendary</th>
      <th>Bug</th>
      <th>Dark</th>
      <th>...</th>
      <th>Psychic</th>
      <th>Rock</th>
      <th>Steel</th>
      <th>Water</th>
      <th>Generation_1</th>
      <th>Generation_2</th>
      <th>Generation_3</th>
      <th>Generation_4</th>
      <th>Generation_5</th>
      <th>Generation_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>


##### Feature standardization



```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
scale_columns = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
preprocessed_df[scale_columns] = scaler.fit_transform(preprocessed_df[scale_columns])
preprocessed_df.head()
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
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Legendary</th>
      <th>Bug</th>
      <th>Dark</th>
      <th>...</th>
      <th>Psychic</th>
      <th>Rock</th>
      <th>Steel</th>
      <th>Water</th>
      <th>Generation_1</th>
      <th>Generation_2</th>
      <th>Generation_3</th>
      <th>Generation_4</th>
      <th>Generation_5</th>
      <th>Generation_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.976765</td>
      <td>-0.950626</td>
      <td>-0.924906</td>
      <td>-0.797154</td>
      <td>-0.239130</td>
      <td>-0.248189</td>
      <td>-0.801503</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.251088</td>
      <td>-0.362822</td>
      <td>-0.524130</td>
      <td>-0.347917</td>
      <td>0.219560</td>
      <td>0.291156</td>
      <td>-0.285015</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.749845</td>
      <td>0.420917</td>
      <td>0.092448</td>
      <td>0.293849</td>
      <td>0.831146</td>
      <td>1.010283</td>
      <td>0.403635</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.583957</td>
      <td>0.420917</td>
      <td>0.647369</td>
      <td>1.577381</td>
      <td>1.503891</td>
      <td>1.729409</td>
      <td>0.403635</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.051836</td>
      <td>-1.185748</td>
      <td>-0.832419</td>
      <td>-0.989683</td>
      <td>-0.392027</td>
      <td>-0.787533</td>
      <td>-0.112853</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



```python
from sklearn.model_selection import train_test_split
X = preprocessed_df.loc[:, preprocessed_df.columns != 'Legendary']
y = preprocessed_df['Legendary']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 33)
```


```python
X_train.shape, X_test.shape
```

<pre>
((600, 31), (200, 31))
</pre>
-----


### 3-2) Logistic Regression 



```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```


```python
lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)
```

<pre>
LogisticRegression(random_state=0)
</pre>

```python
y_pred = lr.predict(X_test)
```


```python
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
```

<pre>
0.955
0.6153846153846154
0.6666666666666666
0.64
</pre>

```python
from sklearn.metrics import confusion_matrix

confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(confmat)
```

<pre>
[[183   5]
 [  4   8]]
</pre>
-----


### 3-3) Class imbalance adjustment



```python
preprocessed_df['Legendary'].value_counts()
```

<pre>
0    735
1     65
Name: Legendary, dtype: int64
</pre>
##### 1:1 sampling



```python
positive_random_index = preprocessed_df[preprocessed_df['Legendary'] == 1].sample(65, random_state = 33).index.tolist()
negative_random_index = preprocessed_df[preprocessed_df['Legendary'] == 0].sample(65, random_state = 33).index.tolist()
```


```python
random_idx = positive_random_index + negative_random_index

X = preprocessed_df.loc[random_idx, preprocessed_df.columns != 'Legendary']
y = preprocessed_df['Legendary'][random_idx]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 33)
```


```python
X_train.shape, X_test.shape
```

<pre>
((97, 31), (33, 31))
</pre>

```python
lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)
```

<pre>
LogisticRegression(random_state=0)
</pre>

```python
y_pred = lr.predict(X_test)
```


```python
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
```

<pre>
0.9696969696969697
0.9230769230769231
1.0
0.9600000000000001
</pre>

```python
from sklearn.metrics import confusion_matrix

confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(confmat)
```

<pre>
[[20  1]
 [ 0 12]]
</pre>
## 4) Cluster classification analysis based on unsupervised learning


### 4-1) Kmeans


##### 2-D Cluster Analysis (Kmeans)



```python
from sklearn.cluster import KMeans
```


```python
X = preprocessed_df[['Attack', 'Defense']]

k_list = []
cost_list = []
for k in range(1,6):
    kmeans = KMeans(n_clusters = k).fit(X)
    inertia = kmeans.inertia_
    print('k:', k, '| cost:', inertia)
    k_list.append(k)
    cost_list.append(inertia)
    
plt.plot(k_list, cost_list)
    
```

<pre>
k: 1 | cost: 1599.9999999999998
k: 2 | cost: 853.3477298974243
k: 3 | cost: 642.310401639615
k: 4 | cost: 480.49450250321513
k: 5 | cost: 403.8390696550314
</pre>
<pre>
[<matplotlib.lines.Line2D at 0x202cda974c0>]
</pre>
<img src="data:image/png;base64,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"/>


```python
kmeans = KMeans(n_clusters = 4).fit(X)
cluster_num = kmeans.predict(X)
cluster = pd.Series(cluster_num)
preprocessed_df['cluster_num'] = cluster.values
preprocessed_df.head()
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
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Legendary</th>
      <th>Bug</th>
      <th>Dark</th>
      <th>...</th>
      <th>Rock</th>
      <th>Steel</th>
      <th>Water</th>
      <th>Generation_1</th>
      <th>Generation_2</th>
      <th>Generation_3</th>
      <th>Generation_4</th>
      <th>Generation_5</th>
      <th>Generation_6</th>
      <th>cluster_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.976765</td>
      <td>-0.950626</td>
      <td>-0.924906</td>
      <td>-0.797154</td>
      <td>-0.239130</td>
      <td>-0.248189</td>
      <td>-0.801503</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.251088</td>
      <td>-0.362822</td>
      <td>-0.524130</td>
      <td>-0.347917</td>
      <td>0.219560</td>
      <td>0.291156</td>
      <td>-0.285015</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.749845</td>
      <td>0.420917</td>
      <td>0.092448</td>
      <td>0.293849</td>
      <td>0.831146</td>
      <td>1.010283</td>
      <td>0.403635</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.583957</td>
      <td>0.420917</td>
      <td>0.647369</td>
      <td>1.577381</td>
      <td>1.503891</td>
      <td>1.729409</td>
      <td>0.403635</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.051836</td>
      <td>-1.185748</td>
      <td>-0.832419</td>
      <td>-0.989683</td>
      <td>-0.392027</td>
      <td>-0.787533</td>
      <td>-0.112853</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>



```python
print(preprocessed_df['cluster_num'].value_counts())
```

<pre>
2    309
1    253
3    128
0    110
Name: cluster_num, dtype: int64
</pre>
##### Cluster visualization



```python
# Visualization
plt.scatter(preprocessed_df[preprocessed_df['cluster_num'] == 0]['Attack'], 
            preprocessed_df[preprocessed_df['cluster_num'] == 0]['Defense'], 
            s = 50, c = 'red', label = 'Pokemon Group 1')
plt.scatter(preprocessed_df[preprocessed_df['cluster_num'] == 1]['Attack'], 
            preprocessed_df[preprocessed_df['cluster_num'] == 1]['Defense'], 
            s = 50, c = 'green', label = 'Pokemon Group 2')
plt.scatter(preprocessed_df[preprocessed_df['cluster_num'] == 2]['Attack'], 
            preprocessed_df[preprocessed_df['cluster_num'] == 2]['Defense'], 
            s = 50, c = 'blue', label = 'Pokemon Group 3')
plt.scatter(preprocessed_df[preprocessed_df['cluster_num'] == 3]['Attack'], 
            preprocessed_df[preprocessed_df['cluster_num'] == 3]['Defense'], 
            s = 50, c = 'yellow', label = 'Pokemon Group 4')
plt.title('Pokemon Cluster by Attack, Defense')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.legend()
plt.show()
```

<img src="data:image/png;base64,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"/>

##### Multidimensional Cluster Analysis (Kmeans)



```python
from sklearn.cluster import KMeans

# K-means train & Elbow method
X = preprocessed_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

k_list = []
cost_list = []
for k in range (1, 15):
    kmeans = KMeans(n_clusters=k).fit(X)
    interia = kmeans.inertia_
    print ("k:", k, "| cost:", interia)
    k_list.append(k)
    cost_list.append(interia)
    
plt.plot(k_list, cost_list)
```

<pre>
k: 1 | cost: 4799.999999999997
k: 2 | cost: 3275.3812330305987
k: 3 | cost: 2862.177290379439
k: 4 | cost: 2566.7760221182552
k: 5 | cost: 2328.181090934398
k: 6 | cost: 2181.2276997543813
k: 7 | cost: 2061.156324293113
k: 8 | cost: 1959.59645118871
k: 9 | cost: 1858.875884725915
k: 10 | cost: 1778.363327659496
k: 11 | cost: 1716.8504013806114
k: 12 | cost: 1657.8019627464134
k: 13 | cost: 1583.8896610770373
k: 14 | cost: 1526.6089686239015
</pre>
<pre>
[<matplotlib.lines.Line2D at 0x202ce2947c0>]
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAX0AAAD4CAYAAAAAczaOAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAk3UlEQVR4nO3de3RV9Z338fc3JyFXQshJiCHhloCWiwIlAoHqtF5aaq3oM+MM01Zta0uXY73MdJ5Obdf0cZ5nOdM1M05b26mttVapHR3tZbDWS6mttiqCQbmjkgBCIJKEWyDccvk+f5xNGjGQEwjsc/m81tor5/zO3jnfwyKfvc/v99t7m7sjIiLpISPsAkRE5OxR6IuIpBGFvohIGlHoi4ikEYW+iEgayQy7gP6UlJT42LFjwy5DRCSprFixotXdS49vT/jQHzt2LHV1dWGXISKSVMzs7b7a1b0jIpJGFPoiImlEoS8ikkYU+iIiaUShLyKSRhT6IiJpRKEvIpJGUjb0f/LK2/xq1Y6wyxARSSgJf3LWqXq8bhs5mRE+PnVk2KWIiCSMlD3Sr62K8vq2PRw62hV2KSIiCSNlQ392dZSOLmfF23vCLkVEJGGkbOhfOLaYSIaxdFNr2KWIiCSMlA39guxMLqgcxtKGXWGXIiKSMFI29CHWr7+6cR8HjnSGXYqISEJI6dCfU11CZ7fz6pbdYZciIpIQUjr0Z4wZTlbEeEVdPCIiQIqHfu6QCNNHDWfpJoW+iAikeOhDbOrm2u37aDvcEXYpIiKhS/nQr62K0u2wfJP69UVEUj70p48uYkhmhrp4RERIg9DPyYowY/RwzdcXESENQh+gtjrKhnfa2HvwaNiliIiEKm1C3x1eUb++iKS5uEPfzCJm9rqZPRk8v9PMtpvZymC5ote6d5hZvZm9aWYf6dU+w8zWBK/dY2Y2uB+nb1Mri8jNirC0QdfhEZH0NpAj/duADce1fdPdpwXLUwBmNglYAEwG5gHfM7NIsP69wEJgQrDMO53i4zUkM4OasZqvLyISV+ibWSXwMeD+OFafDzzq7kfcfTNQD8w0s3Kg0N2XursDi4CrT63sgautjvLWzgO0Hjhytt5SRCThxHuk/y3gy0D3ce1fNLPVZvaAmQ0P2iqAbb3WaQzaKoLHx7e/h5ktNLM6M6traWmJs8STq62KAvCKjvZFJI31G/pmdiXQ7O4rjnvpXqAamAY0AXcf26SPX+MnaX9vo/t97l7j7jWlpaX9lRiX8yuGUZCdqambIpLW4rlH7lzgqmCgNgcoNLOH3f1Tx1Ywsx8CTwZPG4FRvbavBHYE7ZV9tJ8VmZEMLlS/voikuX6P9N39DnevdPexxAZof+funwr66I+5BlgbPH4CWGBm2WY2jtiA7XJ3bwL2m9nsYNbO9cDiwfww/amtjrKppZ2dbYfP5tuKiCSM05mn/6/B9MvVwIeAvwVw93XAY8B64BngZnc/dnfym4gNBtcDDcDTp/H+A1ZbVQKoX19E0lc83Ts93P154Png8XUnWe8u4K4+2uuAKQOqcBBNGllIYU6sX3/+tD7HkEVEUlpanJF7TCTDmDkuqn59EUlbaRX6EOvXf3vXQbbvPRR2KSIiZ13ahf6c6th8fU3dFJF0lHahf17ZUIbnZSn0RSQtpV3oZ2QYs6uivLJpF7GrQYiIpI+0C32I9etv33uIbbvVry8i6SU9Qz+4Ds/STbrUsoikl7QM/fEjCigpyFa/voiknbQMfTNjdlUxS9WvLyJpJi1DH2L9+jvbjrC5tT3sUkREzpr0Df2efn118YhI+kjb0B9Xkk9ZYTYvq19fRNJI2oa+mTGnuoRl6tcXkTSStqEPsS6e1gNH2dh8IOxSRETOivQOfV2HR0TSTFqH/qjiPCqKchX6IpI20jr0IXa0/8rmXXR3q19fRFKfQr8qyt6DHbzxzv6wSxEROeMU+tWary8i6SPtQ39kUS5jonnq1xeRtJD2oQ+xLp5lm3fRpX59EUlxcYe+mUXM7HUzezJ4XmxmS8xsY/BzeK917zCzejN708w+0qt9hpmtCV67x8xscD/OqamtjrL/cCfrduwLuxQRkTNqIEf6twEbej3/CvCcu08AngueY2aTgAXAZGAe8D0ziwTb3AssBCYEy7zTqn6Q9FyHR108IpLi4gp9M6sEPgbc36t5PvBQ8Pgh4Ope7Y+6+xF33wzUAzPNrBwodPelHrvuwaJe24RqRGEO1aX5GswVkZQX75H+t4AvA9292srcvQkg+DkiaK8AtvVarzFoqwgeH9+eEGqro7y6eTcdXd39rywikqT6DX0zuxJodvcVcf7Ovvrp/STtfb3nQjOrM7O6lpaWON/29NRWldB+tIs129WvLyKpK54j/bnAVWa2BXgUuMTMHgZ2Bl02BD+bg/UbgVG9tq8EdgTtlX20v4e73+fuNe5eU1paOoCPc+pmVxUD6tcXkdTWb+i7+x3uXunuY4kN0P7O3T8FPAHcEKx2A7A4ePwEsMDMss1sHLEB2+VBF9B+M5sdzNq5vtc2oYsWZHNe2VBeUb++iKSw05mn/w3gcjPbCFwePMfd1wGPAeuBZ4Cb3b0r2OYmYoPB9UAD8PRpvP+gq62OUrdlD0c71a8vIqkpcyAru/vzwPPB413ApSdY7y7grj7a64ApAy3ybJldFeXBl7ewqnEvF44tDrscEZFBpzNye5ldVYyZ+vVFJHUp9HspyhvCpPJCXm5oDbsUEZEzQqF/nNqqKK9t3cvhjq7+VxYRSTIK/ePUVkc52tnNa1v3hF2KiMigU+gf58JxxWQYvKJ+fRFJQQr94xTmZHF+xTBdh0dEUpJCvw+zq6Os3LaXQ0fVry8iqUWh34faqigdXU7d27vDLkVEZFAp9Ptw4dhiMjNM8/VFJOUo9PuQn53JBZXq1xeR1KPQP4Ha6iirG/dx4Ehn2KWIiAwahf4JzKkuoavbeXWz+vVFJHUo9E9gxpjhDIlkqItHRFKKQv8EcrIiTBtdpMFcEUkpCv2TqK2Ksm7HPvYd6gi7FBGRQaHQP4na6ijdDsvVry8iKUKhfxLTRxeRnZmhLh4RSRkK/ZPIzowwY8xwDeaKSMpQ6PejtirKhqY29rQfDbsUEZHTptDvR211FIBlm3W0LyLJT6Hfjwsqi8jNiqhfX0RSgkK/H0MyM7hwXDEvK/RFJAX0G/pmlmNmy81slZmtM7N/CtrvNLPtZrYyWK7otc0dZlZvZm+a2Ud6tc8wszXBa/eYmZ2ZjzW4aquibGw+QMv+I2GXIiJyWuI50j8CXOLuU4FpwDwzmx289k13nxYsTwGY2SRgATAZmAd8z8wiwfr3AguBCcEyb9A+yRl0rF//Fc3iEZEk12/oe8yB4GlWsPhJNpkPPOruR9x9M1APzDSzcqDQ3Ze6uwOLgKtPq/qzZMrIQgqyMzV1U0SSXlx9+mYWMbOVQDOwxN2XBS990cxWm9kDZjY8aKsAtvXavDFoqwgeH9/e1/stNLM6M6traWmJ/9OcIZmRDGaOK9bN0kUk6cUV+u7e5e7TgEpiR+1TiHXVVBPr8mkC7g5W76uf3k/S3tf73efuNe5eU1paGk+JZ1xtVZRNre3sbDscdikiIqdsQLN33H0v8Dwwz913BjuDbuCHwMxgtUZgVK/NKoEdQXtlH+1J4Vi/vqZuikgyi2f2TqmZFQWPc4HLgDeCPvpjrgHWBo+fABaYWbaZjSM2YLvc3ZuA/WY2O5i1cz2wePA+ypk1sbyQwpxMhb6IJLXMONYpBx4KZuBkAI+5+5Nm9hMzm0asi2YL8AUAd19nZo8B64FO4GZ37wp+103Ag0Au8HSwJIVIhjGrKqrBXBFJav2GvruvBqb30X7dSba5C7irj/Y6YMoAa0wYtVVRlqzfSeOeg1QOzwu7HBGRAdMZuQMwZ7z69UUkuSn0B+DcEUMpzh+iLh4RSVoK/QHIyDBmV8Xm68fOLxMRSS4K/QGqrYqyY99htu4+GHYpIiIDptAfIM3XF5FkptAfoOrSAkqHZqtfX0SSkkJ/gMyM2VVRlqpfX0SSkEL/FFw2cQTN+4/wjaffUPCLSFKJ54xcOc5VU0dSt2UPP/jDJgpzs7j5Q+PDLklEJC4K/VNgZvzTVZM5cKSTf3v2TQpzMrmudmzYZYmI9Euhf4oyMox//YsL2H+4k39cvI6CnEyumV7Z/4YiIiFSn/5pyIpk8N1PTKe2KsrfP76aJet3hl2SiMhJKfRPU05WhB/eUMP5FcO4+b9e46X61rBLEhE5IYX+ICjIzuTBz1zIuGg+n19Ux+tb94RdkohInxT6g6Qobwg/uXEmpUOz+fSPX+WNd9rCLklE5D0U+oNoRGEOD984i9ysCNf9aDlbWtvDLklE5F0U+oNsVHEeD39uJp1d3Xzy/mU07TsUdkkiIj0U+mfA+BFDWfTZWew71MGn7l/GrgNHwi5JRARQ6J8x51cO40c31NC45xA3/Hg5bYc7wi5JREShfybNqory/etm8EbTfj73YB2Hjnb1v5GIyBmk0D/DPnTeCL61YBqvvr2bm366gqOd3WGXJCJprN/QN7McM1tuZqvMbJ2Z/VPQXmxmS8xsY/BzeK9t7jCzejN708w+0qt9hpmtCV67x8zszHysxHLlBSP5l2vO5/k3W/jbx1bS1a0rc4pIOOI50j8CXOLuU4FpwDwzmw18BXjO3ScAzwXPMbNJwAJgMjAP+J6ZRYLfdS+wEJgQLPMG76MktgUzR/O1Kyby69VNfO2Xa3RJZhEJRb+h7zEHgqdZweLAfOChoP0h4Org8XzgUXc/4u6bgXpgppmVA4XuvtRjibeo1zZp4fMXV3HLJeN59NVt/PNTGxT8InLWxXWVzeBIfQUwHvhPd19mZmXu3gTg7k1mNiJYvQJ4pdfmjUFbR/D4+Pa+3m8hsW8EjB49Ov5PkwT+7vJz2X+4kx/+cTOFOVnccumEsEsSkTQSV+i7excwzcyKgF+a2ZSTrN5XP72fpL2v97sPuA+gpqYmpQ6HzYyvXzmJtsMd3L3kLYbmZPLpuePCLktE0sSArqfv7nvN7HliffE7zaw8OMovB5qD1RqBUb02qwR2BO2VfbSnnYwM41///AIOHO7kzl+tZ2hOFn8+Q9fiF5EzL57ZO6XBET5mlgtcBrwBPAHcEKx2A7A4ePwEsMDMss1sHLEB2+VBV9B+M5sdzNq5vtc2aSczksF3PjGdD4wv4X//bBXPrH0n7JJEJA3EM3unHPi9ma0GXgWWuPuTwDeAy81sI3B58Bx3Xwc8BqwHngFuDrqHAG4C7ic2uNsAPD2InyXpZGdG+MF1M5g6qohbH3mdFzfqWvwicmZZos8gqamp8bq6urDLOKP2Hezgr+5bytu7DvLw52YxY8zw/jcSETkJM1vh7jXHt+uM3AQwLC+LRTfOpKwwm8/8eDnrd+ha/CJyZij0E8SIoTk8/LlZ5Gdncv0Dy9i4c3/YJYlIClLoJ5DK4Xk8/LlZgHH1f77Er1al5eQmETmDFPoJprq0gCdv+QATywu55ZHX+fritRzp1NU5RWRwKPQT0DnDcnhk4WwWXlzFoqVv85ffX8q23QfDLktEUoBCP0FlRTL46hUT+cF1M9jU2s6V33mR5zbsDLssEUlyCv0E95HJ5/DrWy6icnguNz5UxzeefoPOLl2TX0ROjUI/CYyO5vHzm+bwiVmj+f4LDXzi/mU0tx0OuywRSUIK/SSRkxXhn685n2/91TTWNO7jinv+yMv1OoNXRAZGoZ9krp5ewRNfnEtR3hA+9aNlfOe5jXTrTlwiEieFfhKaUDaUxTfP5aqpI7l7yVt85sFX2d1+NOyyRCQJKPSTVH52Jt/8q2ncdc0Uljbs4sp7/shrW/eEXZaIJDiFfhIzMz45awy/+Js5RCLGX35/KQ+8uFm3YRSRE1Lop4ApFcN48paLuOR9I/i/T67nb376Gm2HO8IuS0QSkEI/RQzLzeIH183ga1dM5Dfrd3LVd15k3Y59YZclIglGoZ9CzIzPX1zFfy+czeGObq753ss8unyruntEpIdCPwXVjC3m17d+gFnjivnKL9bwpcdXcfBoZ9hliUgCUOinqGhBNg9+Zia3XzaBX76+nav/8yXqmw+EXZaIhEyhn8IiGcbtl53Los/OpPXAUeZ/90We0DX6RdKaQj8NXDShlKduvYiJ5YXc+sjrfPWXa9iv2T0iaUmhnyaOXaP/CxdX8cjyrVx69ws8uXqHBnlF0ky/oW9mo8zs92a2wczWmdltQfudZrbdzFYGyxW9trnDzOrN7E0z+0iv9hlmtiZ47R4zszPzsaQvWZEM7rhiIv/zN3MZUZjNF//rda5/YDlbWtvDLk1EzpJ4jvQ7gS+5+0RgNnCzmU0KXvumu08LlqcAgtcWAJOBecD3zCwSrH8vsBCYECzzBu+jSLymjipi8c0f4M6PT+L1rXv58Lf+wLd/u1G3ZRRJA/2Gvrs3uftrweP9wAag4iSbzAcedfcj7r4ZqAdmmlk5UOjuSz3Wp7AIuPp0P4CcmkiG8em543juS3/GhyeV8c3fvsVHv/VHXtLlmkVS2oD69M1sLDAdWBY0fdHMVpvZA2Y2PGirALb12qwxaKsIHh/f3tf7LDSzOjOra2lpGUiJMkBlhTl89xPvZ9FnZ9LlzifvX8Ztj75O837dpEUkFcUd+mZWAPwcuN3d24h11VQD04Am4O5jq/axuZ+k/b2N7ve5e42715SWlsZbopyGi88t5dnbL+bWSyfw9Jp3uPTuF/jJ0i106Vr9IiklrtA3syxigf9Td/8FgLvvdPcud+8GfgjMDFZvBEb12rwS2BG0V/bRLgkiJyvC311+Ls/cfhEXVA7jHxev43997yXWbtc1fERSRTyzdwz4EbDB3f+jV3t5r9WuAdYGj58AFphZtpmNIzZgu9zdm4D9ZjY7+J3XA4sH6XPIIKoqLeDhG2fx7QXT2L73MFd990XufGKd5vaLpIDMONaZC1wHrDGzlUHbV4G/NrNpxLpotgBfAHD3dWb2GLCe2Myfm9392LSQm4AHgVzg6WCRBGRmzJ9WwQfPG8G/P/smDy3dwlNrmvj6xyfxsfPL0WxbkeRkiX5yTk1NjdfV1YVdRtpbtW0vX/ufNazd3sZFE0r4f/OnMLYkP+yyROQEzGyFu9cc364zciUumtsvkhoU+hI3ze0XSX4KfRkwze0XSV4KfTllx+b239Zrbv+ipVs42tkddmkicgIayJVBsanlAF9fvI4X61spzh/C/GkjuXbGKCaNLAy7NJG0dKKBXIW+DBp354W3Wni8rpEl63dytKubySMLuXZGJfOnVTA8f0jYJYqkDYW+nFV72o+yeOV2Hl/RyLodbQyJZHDZpBFcO2MUF00oITOinkWRM0mhL6FZv6ONx1dsY/HKHexuP0pZYTbXTK/k2ppKqksLwi5PJCUp9CV0Rzu7+d0bO3m8rpHn32qhq9t5/+girq0ZxZUXlDM0JyvsEkVShkJfEkrz/sP88rVY90998wFysjL46JRyrq2pZPa4KBkZusyDyOlQ6EtCcndWbtvL4ysa+dWqHew/3Enl8Fz+YkYlf/7+SkYV54VdokhSUuhLwjvc0cWz697h8bpGXmpoxR3mVEe5tqaSeZPLyR0S6f+XiAig0Jcks33vIX6+opGfrWhk6+6DDM3O5Mqp5Vw9rYIZY4Zr9o9IPxT6kpS6u53lW3bzeF0jT61p4lBHF8PzsrjkfWV8eHIZF08o1TcAkT4o9CXptR/p5IW3WvjNunf43RvNtB3uJDszg4smlPDhSedwycQRlBRkh12mSEI4UejHcxMVkYSQn53JFeeXc8X55XR0dfPq5t38Zv1OlqzfyW83NGMGM0YP5/JJZXx48jmM0/X+Rd5DR/qS9Nyd9U1t/GZdbAewvqkNgPEjCmI7gEllTK0s0jRQSSvq3pG00bjnIEuCbwDLNu+mq9sZMTSbSyfGdgC11VFysjQOIKlNoS9pae/Bo/z+zWaWrN/JC2+20H60i/whEf7svFIun1TGJeeVMSxPZwJL6lHoS9o73NHF0oZd/Gb9Tn67YSct+48QyTBmjSvm8kllXDaxTCeDScpQ6Iv00t3trGzc29MNVN98AICx0TzmjC/hA+NLqK2K6nLQkrROOfTNbBSwCDgH6Abuc/dvm1kx8N/AWGAL8JfuvifY5g7gRqALuNXdnw3aZwAPArnAU8Bt3k8BCn05Gza1HOB3bzTzcsMulm3aRfvRLsxgUnkhc8eXMKc6ysxxxeQN0YQ3SQ6nE/rlQLm7v2ZmQ4EVwNXAp4Hd7v4NM/sKMNzd/8HMJgGPADOBkcBvgXPdvcvMlgO3Aa8QC/173P3pk72/Ql/Oto6ublY37uWl+l28VN/Ka1v30NHlZEWM6aOGM2d8lLnjS5g2qogsnRksCWrQunfMbDHw3WD5oLs3BTuG5939vOAoH3f/l2D9Z4E7iX0b+L27vy9o/+tg+y+c7P0U+hK2Q0e7eHXLbl6qb+WlhlbW7WjDHfKGRJg5rpi51SXMGR9l4jmFmhYqCWNQTs4ys7HAdGAZUObuTQBB8I8IVqsgdiR/TGPQ1hE8Pr69r/dZCCwEGD169EBKFBl0uUMiXHxuKRefWwrEZgQtbdjFSw2tvFy/i+ff3ABAcf4QaqujzK0uYe74KKOL8zDTTkASS9yhb2YFwM+B29297ST/mft6wU/S/t5G9/uA+yB2pB9vjSJnQ1HeED56fjkfPb8cgKZ9h3ipfhcvB98Efr26CYCKolzmBl1BtdVRRgzNCbNsESDO0DezLGKB/1N3/0XQvNPMynt17zQH7Y3AqF6bVwI7gvbKPtpFklr5sNj1//9iRiXuTkNLOy83tPJSfSvPrH2Hx+piX3DPLStgTnVsUHhWVZRhuTo/QM6+eAZyDXiI2KDt7b3a/w3Y1Wsgt9jdv2xmk4H/4k8Duc8BE4KB3FeBW4h1Dz0FfMfdnzrZ+6tPX5JZV7ezbse+2DeBhlZe3bKbwx3dZBicXzGMOcHMoJoxxbpaqAyq05m98wHgj8AaYlM2Ab5KLLgfA0YDW4Fr3X13sM3XgM8CncS6g54O2mv405TNp4FbNGVT0smRzi5e37qXlxt2sbShlde37qWz2xkSyWD66CLmBOMBUzUzSE6TTs4SSUDtRzpZvmV3bGC4vpX1Te+dGVRbHWVSuWYGycDo0soiCSg/O5MPnTeCD50Xm/y2p/0or2zaxcsNse6gu4KZQUV5WdRWRZlTHWXO+BKqSvI1M0hOiUJfJIEMz3/3zKB39h3m5YbW2E6gvpWn174DwDmFOT07gDnVUUYW5YZZtiQRde+IJAl35+1dB2PnBzTsYmnDLna3HwVgdHEeU0cVMbVyGBdUFjGlolCXjEhz6t4RSXJmxtiSfMaW5PPJWWPo7nbe3Lmflxt28erm3azYsptfrYrNgs4wOLdsKBcEO4GplUWcd85QhmRqcDjd6UhfJIU07z/M6m37WN24l1WNsZ97DnYAMCQzg0nlhT3fBqaOGkZVSYEGiFOUZu+IpCF3p3HPIVZu29uzI1i7fR8Hj3YBUJCdyfkVw7hg1DCmVhZxQeUwKopyNUicAtS9I5KGzIxRxXmMKs7j41NHArETxhpaDrBq215WNe5ldeM+HnhxMx1dsQPAaP4QLqgcFowRxHYE0YLsMD+GDCKFvkiaiWQY55YN5dyyoVxbE7tiypHOLt5o2t/zbWDVtr08/1YLxzoCSgqGUF1aQPWIAsaXFjB+ROzxyGE5+laQZBT6IkJ2ZiR2ZD+qiOuCtgNHOlm7fR9rGvexsXk/DS3t/Hp1E/sOdfRslzckQlVp/p92BMHPMdF8DRonKIW+iPSpIDuT2VVRZldFe9rcnV3tR6lvPkBDywHqm2PLq1v28D8r/3T9xEiGMaY4j+peO4Lq0nyqRxRQmKMLzYVJoS8icTMzSgqyKSnIftfOAGKXlNjc2t6zIzi2U3j+zeae8QKAssLsXjuC2FJVmk+5uorOCoW+iAyK/OxMplQMY0rFsHe1d3Z1s3X3wWBH0N6zQ/jla9vZf6SzZ73crAjjSvKpKs2nqjT2zaCqpIBxpfkUZCuqBov+JUXkjMqMZFBVWkBVacG72t2d5v1HaGg5wKaW9tjSeoDVjft4ak0T3b1mk5cVZlNVUtCzQ6gqzae6pICK4blEdJ7BgCj0RSQUZkZZYQ5lhTnMqS5512uHO7rYuvsgm1pi3w6O7RCePG4geUhmBmOjeX3uEIblaeygLwp9EUk4OVmRnmmlvbk7u9uPBjuCA2xqjf18a+d+frthJ529vh5E82PTTCeWD2XyyGFMGlnIhLICsjPT+2Y1OiNXRFJCRzB2sOnYDqGlnfqWA2xoaus5AzkrYowfMZTJIwuDZRgTy4cyNAVnFOmMXBFJaVmRjJ7ZQFDW097d7WzZ1c66HW2s29HG+qY2nn+zmZ+taOxZZ0w0r2cnMCnYIaTqjewV+iKS0jIyrGcg+dilKI4NIq/bsY/1wc5g7fY2nlrzTs92JQXZ7/pGMHlkIaOL85L+AnUKfRFJO70HkS9535++FbQd7mBDsBOILft4qb61Z6ygIDvzXWME55YNZVxJPsNyk6d7SKEvIhIozMliVlWUWb1OPDvS2cXGnQdYt2NfrHtoRxuP1W3rGSeA2KDxuJL82FKaT1Vw34Ox0XxyshJr4FihLyJyEtmZkfecdHZsnKChpZ3NrQfY3BqbVvrCWy083muswAxGDsvt2SGMLYntEMaV5FM5PJfMyNm/PlG/oW9mDwBXAs3uPiVouxP4PNASrPZVd38qeO0O4EagC7jV3Z8N2mcADwK5wFPAbZ7oU4dERPrQe5yg96AxxC5Ut6W1nc29lk2t7SxeuZ22w386AzkzwxgdzYt9K4jGviGMK4mdhVxWmH3GLkkRz5H+g8B3gUXHtX/T3f+9d4OZTQIWAJOBkcBvzexcd+8C7gUWAq8QC/15wNOnVb2ISIIpOMHlKI6dY7BlV+xbQe+dwh83tnKks7tn3bwhEcZE83l04exBHy/oN/Td/Q9mNjbO3zcfeNTdjwCbzawemGlmW4BCd18KYGaLgKtR6ItImjAzogXZRAuymTGm+F2vdXc7TW2H2RJ8K9jc0k7jnoMU5gx+D/zp/MYvmtn1QB3wJXffA1QQO5I/pjFo6wgeH98uIpL2MjKMiqJcKopymTu+pP8NTue9TnG7e4FqYBrQBNwdtPfVCeUnae+TmS00szozq2tpaTnRaiIiMkCnFPruvtPdu9y9G/ghMDN4qREY1WvVSmBH0F7ZR/uJfv997l7j7jWlpaWnUqKIiPThlELfzMp7Pb0GWBs8fgJYYGbZZjYOmAAsd/cmYL+ZzbbYkPT1wOLTqFtERE5BPFM2HwE+CJSYWSPwf4APmtk0Yl00W4AvALj7OjN7DFgPdAI3BzN3AG7iT1M2n0aDuCIiZ52usikikoJOdJVN3a5eRCSNKPRFRNKIQl9EJI0kfJ++mbUAb4ddRx9KgNawizhFqj0cqv3sS9a64fRrH+Pu75nznvChn6jMrK6vQZJkoNrDodrPvmStG85c7ereERFJIwp9EZE0otA/dfeFXcBpUO3hUO1nX7LWDWeodvXpi4ikER3pi4ikEYW+iEgaUegPkJmNMrPfm9kGM1tnZreFXdNAmFnEzF43syfDrmWgzKzIzH5mZm8E//61YdcUDzP72+D/yloze8TMcsKu6UTM7AEzazaztb3ais1siZltDH4OD7PGEzlB7f8W/H9ZbWa/NLOiEEs8ob5q7/Xa35uZm9mg3F1FoT9wncTuFDYRmA3cHNwbOFncBmwIu4hT9G3gGXd/HzCVJPgcZlYB3ArUuPsUIELsPtKJ6kFi96/u7SvAc+4+AXgueJ6IHuS9tS8Bprj7BcBbwB1nu6g4Pch7a8fMRgGXA1sH640U+gPk7k3u/lrweD+x4EmKWz+aWSXwMeD+sGsZKDMrBC4GfgTg7kfdfW+oRcUvE8g1s0wgj5PcQChs7v4HYPdxzfOBh4LHDxG7v3XC6at2d/+Nu3cGT1/h3TdzShgn+HcH+CbwZU5yp8GBUuifhuCG8dOBZSGXEq9vEfsP1B1yHaeiCmgBfhx0T91vZvlhF9Ufd98O/DuxI7UmYJ+7/ybcqgasLLgREsHPESHXc6o+SxLdx8PMrgK2u/uqwfy9Cv1TZGYFwM+B2929Lex6+mNmVwLN7r4i7FpOUSbwfuBed58OtJO43Qw9gv7v+cA4YCSQb2afCreq9GNmXyPWNfvTsGuJh5nlAV8Dvj7Yv1uhfwrMLItY4P/U3X8Rdj1xmgtcZWZbgEeBS8zs4XBLGpBGoNHdj32r+hmxnUCiuwzY7O4t7t4B/AKYE3JNA7Xz2C1Sg5/NIdczIGZ2A3Al8ElPnhOTqokdKKwK/mYrgdfM7JzT/cUK/QEK7vH7I2CDu/9H2PXEy93vcPdKdx9LbCDxd+6eNEec7v4OsM3MzguaLiV2W85EtxWYbWZ5wf+dS0mCAejjPAHcEDy+gSS6v7WZzQP+AbjK3Q+GXU+83H2Nu49w97HB32wj8P7g7+C0KPQHbi5wHbEj5ZXBckXYRaWJW4CfmtlqYBrwz+GW07/gm8nPgNeANcT+5hL20gDBPbGXAueZWaOZ3Qh8A7jczDYSm0nyjTBrPJET1P5dYCiwJPhb/X6oRZ7ACWo/M++VPN92RETkdOlIX0QkjSj0RUTSiEJfRCSNKPRFRNKIQl9EJI0o9EVE0ohCX0Qkjfx/iGmy0+IJ5i4AAAAASUVORK5CYII="/>


```python
# selected by elbow method (5)
kmeans = KMeans(n_clusters=5).fit(X)
cluster_num = kmeans.predict(X)
cluster = pd.Series(cluster_num)
preprocessed_df['cluster_num'] = cluster.values
preprocessed_df.head()
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
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Legendary</th>
      <th>Bug</th>
      <th>Dark</th>
      <th>...</th>
      <th>Rock</th>
      <th>Steel</th>
      <th>Water</th>
      <th>Generation_1</th>
      <th>Generation_2</th>
      <th>Generation_3</th>
      <th>Generation_4</th>
      <th>Generation_5</th>
      <th>Generation_6</th>
      <th>cluster_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.976765</td>
      <td>-0.950626</td>
      <td>-0.924906</td>
      <td>-0.797154</td>
      <td>-0.239130</td>
      <td>-0.248189</td>
      <td>-0.801503</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.251088</td>
      <td>-0.362822</td>
      <td>-0.524130</td>
      <td>-0.347917</td>
      <td>0.219560</td>
      <td>0.291156</td>
      <td>-0.285015</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.749845</td>
      <td>0.420917</td>
      <td>0.092448</td>
      <td>0.293849</td>
      <td>0.831146</td>
      <td>1.010283</td>
      <td>0.403635</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.583957</td>
      <td>0.420917</td>
      <td>0.647369</td>
      <td>1.577381</td>
      <td>1.503891</td>
      <td>1.729409</td>
      <td>0.403635</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.051836</td>
      <td>-1.185748</td>
      <td>-0.832419</td>
      <td>-0.989683</td>
      <td>-0.392027</td>
      <td>-0.787533</td>
      <td>-0.112853</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>


##### Visualize features by cluster



```python
fig = plt.figure(figsize = (12, 12))
ax = fig.gca()
sns.boxplot(x = "cluster_num", y = "HP", data=preprocessed_df, ax=ax)
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
fig = plt.figure(figsize = (12, 12))
ax = fig.gca()
sns.boxplot(x = "cluster_num", y = "Attack", data=preprocessed_df, ax=ax)
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
fig = plt.figure(figsize = (12, 12))
ax = fig.gca()
sns.boxplot(x = "cluster_num", y = "Defense", data=preprocessed_df, ax=ax)
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
fig = plt.figure(figsize = (12, 12))
ax = fig.gca()
sns.boxplot(x = "cluster_num", y = "Sp. Atk", data=preprocessed_df, ax=ax)
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAs8AAAK6CAYAAADGnbHFAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAiP0lEQVR4nO3de7Dnd13f8dd7s4sJLDaSjaI54AonUfEGw5pqmbGIJMNaq+OlFVt1pzKlN3O0WustOhHj1BlbW09qtalYFi8wtEi1yA6JSqA6QNiQGAjB7IFZ9CiYbGiEJQts2E//2LOwpHt5bzjn9z2Xx2PmzO7vdy7f18me2fPMd7/n96sxRgAAgPPbNvUAAADYKMQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAEDT9qkHXIhdu3aN3bt3Tz0DAIBN7o477jgyxrj80fdvqHjevXt3Dh48OPUMAAA2uap635nud9kGAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGWATO3LkSK677ro8+OCDU08B2BTEM8Amtn///tx9993Zv3//1FMANgXxDLBJHTlyJAcOHMgYIwcOHHD2GWAViGeATWr//v0ZYyRJTpw44ewzwCoQzwCb1K233prjx48nSY4fP55bbrll4kUAG594BtikrrnmmuzYsSNJsmPHjlx77bUTLwLY+MQzwCa1b9++VFWSZNu2bdm3b9/EiwA2PvEMsEnt2rUre/fuTVVl7969ueyyy6aeBLDhbZ96AABrZ9++fTl8+LCzzgCrRDwDbGK7du3KTTfdNPUMgE3DZRsAANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACatk89AGCrWVxczNLS0kyOtby8nCSZm5ubyfHm5+ezsLAwk2MBTEE8A2xix44dm3oCwKYingFmbJZnZk8da3FxcWbHBNjMXPMMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCApsniuaourqrbq+pPq+qeqvqZqbYAAEDH9gmP/bEkzxtjHK2qHUn+uKoOjDHeMuEmAAA4q8nieYwxkhxdublj5WVMtQcAAM5n0mueq+qiqroryf1Jbh1jvHXKPQAAcC6TxvMY4xNjjGcmmUtydVV9+aPfpqpeXFUHq+rgAw88MPONAABwyrp4tI0xxkNJbkvygjO87uYxxp4xxp7LL7981tMAAOCTpny0jcur6tKV31+S5PlJ3j3VHgAAOJ8pH23j85Psr6qLcjLiXzXGeO2EewAA4JymfLSNu5M8a6rjAwDAhVoX1zwDAMBGIJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN26ceAADAmS0uLmZpaWkmx1peXk6SzM3Nrfmx5ufns7CwsObHWQviGQCAHDt2bOoJG4J4BgBYp2Z5dvbUsRYXF2d2zI3INc8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATdunHgCb2eLiYpaWlmZyrOXl5STJ3NzcTI43Pz+fhYWFmRwLANYL8QybxLFjx6aeAACbnniGNTTLM7OnjrW4uDizYwLAVuOaZwAAaJosnqvqKVX1hqq6t6ruqaofmGoLAAB0THnZxiNJfniM8faqemKSO6rq1jHGuybcBAAAZzXZmecxxvvHGG9f+f2Hk9yb5Iqp9gAAwPmsi2ueq2p3kmcleesZXvfiqjpYVQcfeOCBmW8DAIBTJo/nqtqZ5NVJfnCM8aFHv36McfMYY88YY8/ll18++4EAALBi0niuqh05Gc6/Ncb4nSm3AADA+Uz5aBuV5KVJ7h1j/OJUOwAAoGvKM8/PSfI9SZ5XVXetvHzjhHsAAOCcJnuoujHGHyepqY4PcLrFxcUsLS1NPWPVHTp0KMlsn+1yFubn5zfd5wRsDJ6eGyDJ0tJS7rznzuTSqZesshMnf7nzL++cdsdqemjqAcBWJp4BTrk0OfHcE1Ov4Dy23Tb5A0UBW5i/gQAAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwDYYm6//fY897nPzR133DH1FNhwxDMAbDE33HBDTpw4kZ/6qZ+aegpsOOIZALaQ22+/PUePHk2SHD161NlnuEDiGQC2kBtuuOHTbjv7DBdGPAPAFnLqrPPZbgPnJp4BYAvZuXPnOW8D5yaeAWALefRlGz/7sz87zRDYoMQzAGwhV1999SfPNu/cuTPPfvazJ14EG4t4BoAt5oYbbsi2bducdYbHYPvUAwCA2br66qtz2223TT0DNiRnngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzxvQkSNHct111+XBBx+cegoAwJYinjeg/fv35+67787+/funngIAsKWI5w3myJEjOXDgQMYYOXDggLPPAAAzJJ43mP3792eMkSQ5ceKEs88AADMknjeYW2+9NcePH0+SHD9+PLfccsvEiwAAtg7xvMFcc8012bFjR5Jkx44dufbaaydeBACwdYjnDWbfvn2pqiTJtm3bsm/fvokXAQBsHeJ5g9m1a1f27t2bqsrevXtz2WWXTT0JAGDL2D71AC7cvn37cvjwYWedAQBmTDxvQLt27cpNN9009QwAgC3HZRsAANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0OQZBgGSLC8vJ3+TbLvNOYV176FkeSxPvQLYonyXAACAJmeeAZLMzc3lgXogJ557YuopnMe227Zl7oq5qWcAW5QzzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN26ceAAAki4uLWVpamsmxlpeXkyRzc3Nrfqz5+fksLCys+XFgVsQzAGwxx44dm3oCbFjiGQDWgVmenT11rMXFxZkdEzYL1zwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0PSY4rmqPmu1hwAAwHp33niuql9/1O2dSV63ZosAAGCd6px5/suq+pUkqarPSXJLkt9cjYNX1a9X1f1V9c7V+HgAALCWzhvPY4yfSvKhqvrVnAzn/zDG+O+rdPyXJXnBKn2sLeO+++7L3r17s7S0NPUUAIAt5azxXFXfduolye1JvibJnUnGyn2fsTHGm5J8cDU+1lZy44035iMf+Uhe8pKXTD0FAGBL2X6O1/39R92+M8mOlftHkt9Zq1Gnq6oXJ3lxkjz1qU+dxSHXtfvuuy+HDx9Okhw+fDhLS0uZn5+fdhQAwBZx1ngeY/yTJKmq54wx/uT011XVc9Z62Gk7bk5yc5Ls2bNnzOq469WNN974abdf8pKX5OUvf/lEawAAtpbODwze1LyPGTh11vlstwEAWDtnPfNcVV+b5O8kubyqfui0V312kovWehhntnv37k8L5t27d0+2BQBgqznXmefHJdmZk4H9xNNePpTk21fj4FX1iiRvTvLFVbVcVS9ajY+7mV1//fWfdvunf/qnJ1oCALD1nOua5zcmeWNVvWyM8b5T91fVU5K8MMkvfKYHH2N812f6Mbaaq6666pNnn3fv3u2HBQEAZqjzOM/vq6pdVfUvqupNSW5L8nlrvoyzuv766/OEJzzBWWcAgBk71zXPT0zyrUn+UZKrkrwmydPGGHMz2sZZXHXVVTlw4MDUMwAAtpxzPc7z/Tn55CjXJ/njMcaoqm+dzSwAAFh/znXZxk8kuTjJryT58ap6+mwmAQDA+nTWeB5j/Mcxxt9O8s1JKsn/SvIFVfWjVXXVjPYBAMC60fmBwfeOMX5ujPEVSb46yd9K4oJbAAC2nM4zDH7SGOMdY4yfGGO4hAMAgC3nguIZAAC2MvG8AR05ciTXXXddHnzwwamnAABsKeJ5A9q/f3/uvvvu7N+/f+opAABbymOK56q6YZV30HTkyJEcOHAgY4wcOHDA2WcAgBk615OknMsdq7qCtv3792eMkSQ5ceJE9u/fnx/6oR+aeBVsEg8l227bZP8gd3Tl152TrlhdDyW5YuoRwFb1mOJ5jPG/V3sIPbfeemuOHz+eJDl+/HhuueUW8QyrYH5+fuoJa+LQoUNJkiuvuHLiJavois375wWsf+eN56p6WpJfSvK1SU4keXOSfz3GeO8ab+MMrrnmmrzuda/L8ePHs2PHjlx77bVTT4JNYWFhYeoJa+LU57W4uDjxEoDNofPvk7+d5FVJnpzkC5L8jySvWMtRnN2+fftSVUmSbdu2Zd++fRMvAgDYOjrxXGOM3xhjPLLy8ptJxloP48x27dqVvXv3pqqyd+/eXHbZZVNPAgDYMjrXPL+hqn4syStzMpq/M8nvV9WTkmSM8cE13McZ7Nu3L4cPH3bWGQBgxjrx/J0rv/6zR93/fTkZ009b1UWc165du3LTTTdNPQMAYMs5bzyPMb5oFkNgVhYXF7O0tDT1jFV36lEVNtsPvs3Pz2+6zwmAjeus8VxVX53kL8YYH1i5/b1Jvj3J+5Lc4HINNqqlpaXc986356k7PzH1lFX1uOMnf4Tho4ffNvGS1fPnRy+aegIAfJpznXn+r0menyRV9XVJfj7JdUmemeTmJN+x1uNgrTx15ydy/Z6j539DJnXjwc30zB4AbAbniueLTju7/J1Jbh5jvDrJq6vqrjVfBgAA68y5Hqruoqo6FdffkOSPTnvdY31abwAA2LDOFcGvSPLGqjqS5FiS/5MkVTWf5G9msA0AANaVs8bzGOPnquoPk3x+klvGGKeeGGVbTl77DAAAW8o5L78YY7zlDPfdt3ZzAABg/eo8PTcAABA/+LdqZvnEG8vLy0mSubm5mRzPk1QAwKd4sq2NZbU7RjxvQMeOHZt6AgBsWUtLS7nnHffm0sd/7tRTVtWJj1eS5C/f8+DES1bPQw/fv+ofUzyvkln+X9qpYy0uLs7smADAp1z6+M/N13/JC6eewXm84d2vXPWP6ZpnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAEDT9qkHAMB6tbi4mKWlpalnrLpDhw4lSRYWFiZesrrm5+c33efE+iOeAeAslpaW8u677sqTpx6yyk79s/NDd9015YxV9YGpB7BliGcAOIcnJ3lRauoZnMdLM6aewBbhmmcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaPM4zAMAFWF5ezt88/OG84d2vnHoK5/HQw/dnLB9b1Y/pzDMAADQ58wwAcAHm5uZSH3swX/8lL5x6Cufxhne/MlfMXbaqH9OZZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAICm7VMPgFlbXl7ORz58UW48uHPqKZzH+z58UZ6wvDz1DAD4JGeeAQCgyZlntpy5ubl89JH35/o9R6eewnnceHBnLp6bm3oGAHySM88AANA0aTxX1Quq6s+qaqmqfmzKLQAAcD6TxXNVXZTkl5PsTfKMJN9VVc+Yag8AAJzPlGeer06yNMZ47xjj40lemeRbJtwDAADnNGU8X5HkL067vbxyHwAArEtTxnOd4b7x/71R1Yur6mBVHXzggQdmMAsAAM5synheTvKU027PJfmrR7/RGOPmMcaeMcaeyy+/fGbjAADg0aaM57clubKqvqiqHpfkhUl+b8I9AABwTpM9ScoY45Gq+v4kr09yUZJfH2PcM9UeAAA4n0mfYXCM8bokr5tyAwAAdHmGQQAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoGn71APW0uLiYpaWlqaeseoOHTqUJFlYWJh4yeqan5+f2ef050cvyo0Hd87kWLPy1w+f/H/hz3v8iYmXrJ4/P3pRrpp6BFva8vJyPpzkpRlTT+E83p/k6PLyzI730MP35w3vfuXMjjcLRz/6f5MkOy/+nImXrJ6HHr4/V+SyVf2Ymzqel5aWcuc73pUTj3/S1FNWVX385F/id7znAxMvWT3bHv7gzI41Pz8/s2PN0sdX/qfq4t1XTrxk9VyVzfvnBWxcm/XvpUOHTn4vvuLpqxubU7oil636n9emjuckOfH4J+Wjz/imqWdwHhe/67UzO9ZmO2N/yqnPa3FxceIlsHnMzc3loSNH8qLU1FM4j5dm5NK5uZkcy/eRrc01zwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABo2j71AICtZnFxMUtLSzM51qFDh5IkCwsLMzne/Pz8zI4FMAXxDLCJXXLJJVNPANhUxDPAjDkzC7BxueYZAACanHkGgHP4QJKXZkw9Y1U9uPLrZZOuWF0fSHLp1CPYEsQzAJzF/Pz81BPWxAMrP0h66ZVXTrxk9VyazfvnxfoingHgLDbr9emnPq/FxcWJl8DG45pnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN26cesJaWl5ez7eG/ycXveu3UUziPbQ8/mOXlR6aeAQBwTs48AwBA06Y+8zw3N5e//tj2fPQZ3zT1FM7j4ne9NnNzT556BgDAOTnzDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANE0Sz1X1D6rqnqo6UVV7ptgAAAAXaqozz+9M8m1J3jTR8QEA4IJtn+KgY4x7k6Sqpjg8AAA8Juv+mueqenFVHayqgw888MDUcwAA2MLW7MxzVf1Bkief4VU/Ocb43e7HGWPcnOTmJNmzZ89YpXkAAHDB1iyexxjPX6uPDQAAU1j3l20AAMB6MdVD1X1rVS0n+dokv19Vr59iBwAAXIipHm3jNUleM8WxAQDgsXLZBgAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGia5Om5Z2nbwx/Mxe967dQzVlV99ENJknHxZ0+8ZPVse/iDSZ489QwAgHPa1PE8Pz8/9YQ1cejQh5MkVz59M8XmkzftnxcAsHls6nheWFiYesKaOPV5LS4uTrwEAGBrcc0zAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQNP2qQfAZra4uJilpaWZHOvQoUNJkoWFhZkcb35+fmbHAoD1QjzDJnHJJZdMPQEANj3xDGvImVkA2Fxc8wwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQJN4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANC0feoBAACc2eLiYpaWlmZyrEOHDiVJFhYW1vxY8/PzMznOWhDPAADkkksumXrChiCeAQDWqY16dnYzc80zAAA0iWcAAGgSzwAA0CSeAQCgSTwDAECTeAYAgCbxDAAATeIZAACaxDMAADSJZwAAaBLPAADQJJ4BAKBJPAMAQFONMabe0LZnz55x8ODBqWec0eLiYpaWlmZyrEOHDiVJrrzyypkcb35+PgsLCzM5FsBWtVm/j/gewkZVVXeMMfY8+v7tU4zhM3PJJZdMPQGADcz3EXjsnHkGAIBHOduZZ9c8AwBAk3gGAIAm8QwAAE2TxHNV/UJVvbuq7q6q11TVpVPsAACACzHVmedbk3z5GOMrk9yX5Mcn2gEAAG2TxPMY45YxxiMrN9+SZG6KHQAAcCHWwzXP35fkwNQjAADgfNbsSVKq6g+SPPkMr/rJMcbvrrzNTyZ5JMlvnePjvDjJi5PkqU996hosBQCAnjWL5zHG88/1+qral+SbknzDOMcztYwxbk5yc3LySVJWdSQAAFyASZ6eu6pekORHk/zdMcbDU2wAAIALNdU1z/85yROT3FpVd1XVr060AwAA2iY58zzGmJ/iuAAA8JlYD4+2AQAAG4J4BgCAJvEMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEAoEk8AwBAk3gGAIAm8QwAAE3iGQAAmsQzAAA0iWcAAGgSzwAA0CSeAQCgqcYYU29oq6oHkrxv6h3rxK4kR6Yewbrj64Iz8XXBmfi64Ex8XXzKF44xLn/0nRsqnvmUqjo4xtgz9Q7WF18XnImvC87E1wVn4uvi/Fy2AQAATeIZAACaxPPGdfPUA1iXfF1wJr4uOBNfF5yJr4vzcM0zAAA0OfMMAABN4hkAAJrE8wZTVS+oqj+rqqWq+rGp97A+VNWvV9X9VfXOqbewPlTVU6rqDVV1b1XdU1U/MPUmpldVF1fV7VX1pytfFz8z9SbWj6q6qKrurKrXTr1lPRPPG0hVXZTkl5PsTfKMJN9VVc+YdhXrxMuSvGDqEawrjyT54THGlyb5miT/yt8XJPlYkueNMb4qyTOTvKCqvmbaSawjP5Dk3qlHrHfieWO5OsnSGOO9Y4yPJ3llkm+ZeBPrwBjjTUk+OPUO1o8xxvvHGG9f+f2Hc/Ib4hXTrmJq46SjKzd3rLx45ABSVXNJ/l6SX5t6y3onnjeWK5L8xWm3l+ObIXAeVbU7ybOSvHXiKawDK/80f1eS+5PcOsbwdUGS/Kck/zbJiYl3rHvieWOpM9znjAFwVlW1M8mrk/zgGONDU+9hemOMT4wxnplkLsnVVfXlE09iYlX1TUnuH2PcMfWWjUA8byzLSZ5y2u25JH810RZgnauqHTkZzr81xvidqfewvowxHkpyW/y8BMlzknxzVR3OyUtCn1dVvzntpPVLPG8sb0tyZVV9UVU9LskLk/zexJuAdaiqKslLk9w7xvjFqfewPlTV5VV16crvL0ny/CTvnnQUkxtj/PgYY26MsTsn2+KPxhjfPfGsdUs8byBjjEeSfH+S1+fkD/+8aoxxz7SrWA+q6hVJ3pzki6tquapeNPUmJvecJN+Tk2eQ7lp5+capRzG5z0/yhqq6OydPyNw6xvCwZHABPD03AAA0OfMMAABN4hkAAJrEMwAANIlnAABoEs8AANAkngEmVFU3VNW/eQzvd2lV/cu12ATA2YlngI3p0iQXFM91kr/3AT4D/hIFmKGq+t6quruq/rSqfuNRr7utqvas/H7XylPlpqq+rKpuX3mik7ur6sokP5/k6Sv3/cLK2/1IVb1t5W1+ZuW+3VV1b1X9lyRvT/KUs+w6WlU/t7LrLVX1eSv3v6yqvuP0t1v59blV9caqelVV3VdVP19V/3hl5zuq6umr/J8OYF0QzwAzUlVfluQnkzxvjPFVSX6g+a7/PMkvjTGemWRPkuUkP5bkPWOMZ44xfqSqrk1yZZKrkzwzybOr6utW3v+Lk7x8jPGsMcb7znKMJyR5y8quNyX5p41dpz6Hr8jJZzO8aoxxdZJfS3Jd83MD2FDEM8DsPC/J/xxjHEmSMcYHm+/35iQ/UVU/muQLxxjHzvA216683JmTZ5i/JCdjOkneN8Z4y3mO8fEkp56m+Y4kuxu73jbGeP8Y42NJ3pPklpX739F8f4ANRzwDzE4lGed4/SP51N/LF5+6c4zx20m+OcmxJK+vqued5WP/u5Uz0c8cY8yPMV668rqPNLYdH2Oc2vaJJNsfvamqKsnjTnufj532+xOn3T5x2vsDbCriGWB2/jDJP6yqy5Kkqp70qNcfTvLsld+ffp3x05K8d4yxmOT3knxlkg8neeJp7/v6JN9XVTtX3ueKqvrcVdh8+qZvSbJjFT4mwIblzADAjIwx7qmqn0vyxqr6RE5eYnH4tDf590leVVXfk+SPTrv/O5N8d1UdT/KBJC8ZY3ywqv6kqt6Z5MDKdc9fmuTNJ08Q52iS787Js8ifif+W5Her6vacjP/OWWyATas+9a90AADAubhsAwAAmly2AbCFVNVbk3zWo+7+njHGO6bYA7DRuGwDAACaXLYBAABN4hkAAJrEMwAANIlnAABoEs8AAND0/wDl5y9zDjG6FgAAAABJRU5ErkJggg=="/>


```python
fig = plt.figure(figsize = (12, 12))
ax = fig.gca()
sns.boxplot(x = "cluster_num", y = "Speed", data=preprocessed_df, ax=ax)
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
```
