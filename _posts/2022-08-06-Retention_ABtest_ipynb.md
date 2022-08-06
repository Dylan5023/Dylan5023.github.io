---
layout: single
title:  "Retention_ABtest"
categories: Kaggle
tag: [python, blog, jekyll, matplotlib, seaborn, ML, Kaggle]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

**[Notice]** [Increasing_Customer_Retention]
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


# Increase customer retention with A/B testing



-----


## Data

Data is (https://www.kaggle.com/yufengsui/mobile-games-ab-testing)



 * **userid** - An identification number that identifies individual users.

  * **version** - You can see which user belongs to the experimental group or control group. (gate_30, gate_40)

  * **sum_gamerounds** - Number of rounds played by users in 14 days after first install.

  * **retention_1** - Whether the user returned within 1 day of installation.

  * **retention_7** - Whether the user returned within 7 days of installation.

 


-----


## problem definition

  * In the Cookie Cats game, when a specific stage is reached, the stage is locked.

  * In case of Area Locked, you can get 3 keys by playing a special edition game to get Keys, ask a Facebook friend, or purchase a paid item and open it immediately.

  * When locking in the stage at which stage, it is necessary to decide which is best for user retention.


-----


### Data exploration



```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```


```python
df = pd.read_csv('Data/cookie_cats.csv')
print(df.shape)
df.tail() 
```

<pre>
(90189, 5)
</pre>
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
      <th>userid</th>
      <th>version</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90184</th>
      <td>9999441</td>
      <td>gate_40</td>
      <td>97</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90185</th>
      <td>9999479</td>
      <td>gate_40</td>
      <td>30</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90186</th>
      <td>9999710</td>
      <td>gate_30</td>
      <td>28</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90187</th>
      <td>9999768</td>
      <td>gate_40</td>
      <td>51</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90188</th>
      <td>9999861</td>
      <td>gate_40</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 90189 entries, 0 to 90188
Data columns (total 5 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   userid          90189 non-null  int64 
 1   version         90189 non-null  object
 2   sum_gamerounds  90189 non-null  int64 
 3   retention_1     90189 non-null  bool  
 4   retention_7     90189 non-null  bool  
dtypes: bool(2), int64(2), object(1)
memory usage: 2.2+ MB
</pre>

```python
df.groupby("version").count()
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
      <th>userid</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
    <tr>
      <th>version</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gate_30</th>
      <td>44700</td>
      <td>44700</td>
      <td>44700</td>
      <td>44700</td>
    </tr>
    <tr>
      <th>gate_40</th>
      <td>45489</td>
      <td>45489</td>
      <td>45489</td>
      <td>45489</td>
    </tr>
  </tbody>
</table>
</div>



```python
sns.boxenplot(data=df, y="sum_gamerounds")
```

<pre>
<AxesSubplot:ylabel='sum_gamerounds'>
</pre>
<img src="data:image/png;base64,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"/>


```python
df.loc[df["sum_gamerounds"] > 45000]
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
      <th>userid</th>
      <th>version</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57702</th>
      <td>6390605</td>
      <td>gate_30</td>
      <td>49854</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Remove users who have played more than 45000. 
df = df[df["sum_gamerounds"] < 45000 ]
print(df.shape)
df.tail()
```

<pre>
(90188, 5)
</pre>
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
      <th>userid</th>
      <th>version</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90184</th>
      <td>9999441</td>
      <td>gate_40</td>
      <td>97</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90185</th>
      <td>9999479</td>
      <td>gate_40</td>
      <td>30</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90186</th>
      <td>9999710</td>
      <td>gate_30</td>
      <td>28</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90187</th>
      <td>9999768</td>
      <td>gate_40</td>
      <td>51</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90188</th>
      <td>9999861</td>
      <td>gate_40</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Look at the percentile.
df["sum_gamerounds"].describe()
```

<pre>
count    90188.000000
mean        51.320253
std        102.682719
min          0.000000
25%          5.000000
50%         16.000000
75%         51.000000
max       2961.000000
Name: sum_gamerounds, dtype: float64
</pre>

```python
sns.boxenplot(data=df, y="sum_gamerounds")
```

<pre>
<AxesSubplot:ylabel='sum_gamerounds'>
</pre>
<img src="data:image/png;base64,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"/>

-----


### Data analysis



```python
# Count the number of users for each game run.
plot_df = df.groupby("sum_gamerounds")["userid"].count()
plot_df
```

<pre>
sum_gamerounds
0       3994
1       5538
2       4606
3       3958
4       3629
        ... 
2251       1
2294       1
2438       1
2640       1
2961       1
Name: userid, Length: 941, dtype: int64
</pre>

```python
ax = plot_df[:100].plot(figsize=(10,6))
ax.set_title("The number of players that played 0-100 game rounds during the first week")
ax.set_ylabel("Number of Players")
ax.set_xlabel('# Game rounds')
```

<pre>
Text(0.5, 0, '# Game rounds')
</pre>
<img src="data:image/png;base64,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"/>


```python
sns.distplot(df["sum_gamerounds"])
```

<pre>
<AxesSubplot:xlabel='sum_gamerounds', ylabel='Density'>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZUAAAEHCAYAAABm9dtzAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAml0lEQVR4nO3df5xcdX3v8dd7ZndDEsAkZNGQBBMwSIOVCCngVSy9V2qgXlPbawWtKLWlaaH39treh7H2oWhrr1p/PMpDSqQ1D6GtRLz4I3qjiFTB3hohaESCBEJEWEgh/Ar5ubuz87l/nDO7J5OZ2dmdmf2R834+HvPYme8533O+3x3YT74/jyICMzOzdihMdgHMzOzo4aBiZmZt46BiZmZt46BiZmZt46BiZmZt0zXZBZhM8+fPjyVLlkx2MczMppV77rnn6YjorXUs10FlyZIlbNmyZbKLYWY2rUj6Rb1j7v4yM7O2cVAxM7O2cVAxM7O2cVAxM7O26WhQkbRK0nZJOyStrXFckq5Jj98r6azMsfWSnpJ0X1WeL0ramr4ekbQ1TV8i6WDm2LpO1s3MzI7UsdlfkorAtcCFQB9wt6SNEXF/5rSLgGXp61zguvQnwOeBzwA3Zq8bEW/N3OOTwJ7M4YcjYkVbK2JmZk3rZEvlHGBHROyMiAFgA7C66pzVwI2R2AzMkbQAICLuBJ6td3FJAn4HuKkjpTczszHrZFBZCDyW+dyXpo31nHrOB56MiIcyaUsl/VjSHZLOr5VJ0hWStkjasnv37iZvZWZmzehkUFGNtOqHtzRzTj2XcngrZRdwckS8CngP8AVJxx9x8YjrI2JlRKzs7a25IHRc/mnzL/izm3/StuuZmU1HnVxR3wcsznxeBDwxjnOOIKkL+C3g7EpaRPQD/en7eyQ9DJwGTMiS+c0PP8M9v3huIm5lZjZldbKlcjewTNJSST3AJcDGqnM2Apels8DOA/ZExK4mrv164IGI6KskSOpNJwcg6RSSwf+d7ahIM144NEip7Kdomlm+daylEhElSVcBtwJFYH1EbJO0Jj2+DtgEXAzsAA4Al1fyS7oJuACYL6kP+GBEfC49fAlHDtC/DviwpBIwBKyJiLoD/e32wqESQ+XyRN3OzGxK6uiGkhGxiSRwZNPWZd4HcGWdvJc2uO67aqTdAtwy3rK2aq9bKmZmXlHfLnsPlRhyUDGznMv11vfttPfQII4pZpZ3bqm0wUCpzKHBslsqZpZ7DiptsPfQIABD5SAZJjIzyycHlTbYe6g0/N6NFTPLMweVNnghbakAlDyt2MxyzEGlDbItFY+rmFmeOai0wd7DWioOKmaWXw4qbfBCtqUy5KBiZvnloNIGLxx0S8XMDBxU2sJjKmZmCQeVNsgGFc/+MrM8c1Bpg+xAvVsqZpZnDipt8IJnf5mZAQ4qbXHYinoHFTPLMQeVNjh8TMVBxczyy0GlDfYeGmRWTxHwmIqZ5ZuDShu8cKjE3Fk9gFsqZpZvDiotigj2HhpkzqxuAD+n3sxyzUGlRf2lMoNDwbzZaUvF27SYWY51NKhIWiVpu6QdktbWOC5J16TH75V0VubYeklPSbqvKs/Vkh6XtDV9XZw59r70WtslvaGTdauoTCeek3Z/eUzFzPKsY0FFUhG4FrgIWA5cKml51WkXAcvS1xXAdZljnwdW1bn8pyNiRfralN5vOXAJcEaa7+/TMnTUvnTm14tmdgEeUzGzfOtkS+UcYEdE7IyIAWADsLrqnNXAjZHYDMyRtAAgIu4Enh3D/VYDGyKiPyJ+DuxIy9BRlZbJjC7P/jIz62RQWQg8lvncl6aN9Zxarkq7y9ZLmtvitVpSiSHdxeRX6ZaKmeVZJ4OKaqRV/8Vt5pxq1wGnAiuAXcAnx3ItSVdI2iJpy+7du0e51egqLZOersJhn83M8qiTQaUPWJz5vAh4YhznHCYinoyIoYgoA//ASBdXU9eKiOsjYmVErOzt7W2qIo2Uo9L95aBiZtbJoHI3sEzSUkk9JIPoG6vO2Qhcls4COw/YExG7Gl20MuaSejNQmR22EbhE0gxJS0kG/+9qR0UaqQSV7mLSUPLW92aWZ12dunBElCRdBdwKFIH1EbFN0pr0+DpgE3AxyaD6AeDySn5JNwEXAPMl9QEfjIjPAR+XtIKka+sR4A/T622TdDNwP1ACroyIoU7Vr2K4+6voloqZWceCCkA63XdTVdq6zPsArqyT99I66e9ocL+PAB8ZV2HHqRJDetLZXx6oN7M884r6FlV3f7mlYmZ55qDSourZX26pmFmeOai0qNJSGR5TGfJAvZnll4NKiyqTvbz40czMQaVlwy0Vr1MxM3NQadXQ8EB94bDPZmZ55KDSonL1Ni1+noqZ5ZiDSouG16l4TMXMzEGlVZUxlEIBigV5TMXMcs1BpUWRjqEUC6JYkFsqZpZrDiotqgzMFyS6CmLIG0qaWY45qLRouPtLbqmYmTmotKgyg7jS/eUxFTPLMweVFo20VKDLLRUzyzkHlRZlx1SKBXmdipnlmoNKiyqzvwoF0VUoeEW9meWag0qLKpsSF+UxFTOzjj75MQ8qG0p+bevjHBgo8fDufXzhh48OH3/buSdPVtHMzCacWyotqgQVSRSk4b3AzMzyyEGlRcOzv0gG6x1TzCzPHFRaVAkikigURlouZmZ51NGgImmVpO2SdkhaW+O4JF2THr9X0lmZY+slPSXpvqo8fyvpgfT8r0iak6YvkXRQ0tb0ta6TdauodHdJlZaKg4qZ5VfHgoqkInAtcBGwHLhU0vKq0y4ClqWvK4DrMsc+D6yqcenbgFdExCuBB4H3ZY49HBEr0teatlRkFNl1KsmYykTc1cxsaupkS+UcYEdE7IyIAWADsLrqnNXAjZHYDMyRtAAgIu4Enq2+aER8OyJK6cfNwKKO1aAJIwP1yap6r1MxszzrZFBZCDyW+dyXpo31nEZ+D/hm5vNSST+WdIek88dS2PFy95eZ2YhOrlNRjbTqv7jNnFP74tL7gRLwL2nSLuDkiHhG0tnAVyWdEREvVOW7gqSrjZNPbn0NSWWgviBRKIhSyUHFzPKrky2VPmBx5vMi4IlxnHMESe8E3gi8PdJ9UiKiPyKeSd/fAzwMnFadNyKuj4iVEbGyt7d3DNWprTKlWCTdX26pmFmedTKo3A0sk7RUUg9wCbCx6pyNwGXpLLDzgD0RsavRRSWtAt4LvCkiDmTSe9PJAUg6hWTwf2f7qlNbOQLJix/NzKCD3V8RUZJ0FXArUATWR8Q2SWvS4+uATcDFwA7gAHB5Jb+km4ALgPmS+oAPRsTngM8AM4DbJAFsTmd6vQ74sKQSMASsiYgjBvrbrRxBISmHFz+aWe51dO+viNhEEjiyaesy7wO4sk7eS+ukv6xO+i3ALeMu7DgNlZPNJCHZqdizv8wsz7yivkURQSH9LRaEu7/MLNccVFo0VB7p/ip6SrGZ5ZyDSouGIoa7v+QxFTPLOQeVFkUkYyngKcVmZg4qLUq6v5L3hYKnFJtZvjmotGgogmLBU4rNzMBBpWURgYYH6t39ZWb55qDSoqHyyEC9N5Q0s7xzUGlRORjp/iq4+8vM8s1BpUXlcrL3F3jxo5mZg0qLqgfqA4+rmFl+Oai0qByH7/0FydoVM7M8clBp0WHdX5U0RxUzyykHlRYNleOwgXrwuIqZ5ZeDSouqn6eSpE1miczMJo+DSosODyojaWZmeeSg0qLqdSpJmoOKmeWTg0qLDttQ0t1fZpZzDiotKkdktr73QL2Z5VtTQUXSLZJ+Q5KDUBWPqZiZjWg2SFwHvA14SNJHJZ3ewTJNK4dtKJlGlSEHFTPLqaaCSkR8JyLeDpwFPALcJunfJV0uqbtePkmrJG2XtEPS2hrHJema9Pi9ks7KHFsv6SlJ91XlmSfpNkkPpT/nZo69L73WdklvaKZurSoHFNLfosdUzCzvmu7OknQC8C7g94EfA39HEmRuq3N+EbgWuAhYDlwqaXnVaRcBy9LXFSQtoorPA6tqXHotcHtELANuTz+TXvsS4Iw039+nZeiocrlG95ejipnlVLNjKl8Gvg/MAv5rRLwpIr4YEX8CHFsn2znAjojYGREDwAZgddU5q4EbI7EZmCNpAUBE3Ak8W+O6q4Eb0vc3AL+ZSd8QEf0R8XNgR1qGjqreUBI8pmJm+dXV5Hn/GBGbsgmSZqR/wFfWybMQeCzzuQ84t4lzFgK7GpTlxRGxCyAidkk6MXOtzTWudRhJV5C0ijj55JMb3KY55cAr6s3MUs12f/11jbQfjJJHNdKq/9w2c06zmrpWRFwfESsjYmVvb+84bzWinF2nUhi+R8vXNTObjhq2VCS9hORf+zMlvYqRP9zHk3SFNdIHLM58XgQ8MY5zqj0paUHaSlkAPNXCtVp22IaS8uwvM8u30bq/3kAyOL8I+FQmfS/wF6PkvRtYJmkp8DjJIPrbqs7ZCFwlaQNJ19ieStdWAxuBdwIfTX9+LZP+BUmfAk4iGfy/a5Rrtawcgaq7v8qdvquZ2dTUMKhExA3ADZJ+OyJuGcuFI6Ik6SrgVqAIrI+IbZLWpMfXAZuAi0kG1Q8Al1fyS7oJuACYL6kP+GBEfI4kmNws6d3Ao8Bb0uttk3QzcD9QAq6MiKGxlHk8yjGyTqXoxY9mlnOjdX/9bkT8M7BE0nuqj0fEp2pkyx7fRBI4smnrMu8DuLJO3kvrpD8D/Jc6xz4CfKRRmdrNG0qamY0Yrftrdvqz3rTh3DvsyY/u/jKznBut++uz6c8PTUxxph+vUzEzG9Hs4sePSzpeUrek2yU9Lel3O1246SA7piKPqZhZzjW7TuXXI+IF4I0kU3dPA/5Xx0o1jZTLDM/+KrqlYmY512xQqWwaeTFwU0TU2j4ll5J1Ksn74YF6j6mYWU41u03L1yU9ABwE/lhSL3Coc8WaPvw8FTOzEc1ufb8WeDWwMiIGgf0cuTlkLtV88qNjipnlVLMtFYBfIlmvks1zY5vLM+2UY2QspTILrOT+LzPLqaaCiqR/Ak4FtgKVVeqBgwpDmQ0lu9Il9aUhN1XMLJ+abamsBJaHt989Qrk80v3VlW5T7JaKmeVVs7O/7gNe0smCTFfZgfpiQRTkloqZ5VezLZX5wP2S7gL6K4kR8aaOlGoaya6oB+gqFhgcckvFzPKp2aBydScLMZ1ln/wI0F0QJU//MrOcaiqoRMQdkl4KLIuI70iaRbKdfe5ln/wIlZaKg4qZ5VOze3/9AfB/gM+mSQuBr3aoTNPKEd1fBXmg3sxyq9mB+iuB1wAvAETEQ8CJnSrUdBERRHX3V7HggXozy61mg0p/RAxUPqQLIHP/l7MydJINKl1FeaDezHKr2aByh6S/AGZKuhD4EvD1zhVrehhKo0ox81vsLhY8UG9mudVsUFkL7AZ+CvwhySOC/7JThZouKhtHSoePqbilYmZ51ezsr7KkrwJfjYjdnS3S9FEJKtXrVEqHSpNVJDOzSdWwpaLE1ZKeBh4AtkvaLekDE1O8qa3Sy1U8bKDeLRUzy6/Rur/+lGTW169ExAkRMQ84F3iNpP852sUlrZK0XdIOSWtrHJeka9Lj90o6a7S8kr4oaWv6ekTS1jR9iaSDmWPrmvoNtKAypqLsOpWCx1TMLL9G6/66DLgwIp6uJETEzvT59N8GPl0vo6QicC1wIckjiO+WtDEi7s+cdhGwLH2dC1wHnNsob0S8NXOPTwJ7Mtd7OCJWjFKntimXa3V/iZJbKmaWU6O1VLqzAaUiHVfprnF+1jnAjojYmU5H3sCRD/ZaDdwYic3AHEkLmsmrZHT8d4CbRilHx9QaU+kuiEG3VMwsp0YLKgPjPAbJqvvHMp/70rRmzmkm7/nAk+lCzIqlkn4s6Q5J59cqlKQrJG2RtGX37tbmHAzVmv1VLLilYma5NVr315mSXqiRLuCYUfKqRlr1P+HrndNM3ks5vJWyCzg5Ip6RdDbwVUlnRMRh5Y+I64HrAVauXNlSk6KyG0v1QH05kvGWbAvGzCwPGgaViGhl08g+YHHm8yLgiSbP6WmUN13R/1vA2Zmy9pNuyx8R90h6GDgN2NJCHRqqdH8VNDITLPugrmLBe26aWb40u/hxPO4GlklaKqkHuATYWHXORuCydBbYecCeiNjVRN7XAw9ERF8lQVJvOsCPpFNIBv93dqpyMDL7q1A1UA9+UJeZ5VOzz1MZs4goSboKuJVkm/z1EbFN0pr0+DqSlfkXAzuAA8DljfJmLn8JRw7Qvw74sKQSMASsiYhnO1W/pJzJz6JEKe2d6073bPFaFTPLo44FFYCI2EQSOLJp6zLvg2QH5KbyZo69q0baLcAtLRR3zCoD9YVMe68rbbV4rYqZ5VEnu7+OesPdX1Vb34NbKmaWTw4qLYg4Mqh4TMXM8sxBpQVDtTaUHJ795aBiZvnjoNKCyjqVQtU6FXD3l5nlk4NKC7LrVCq60jEVd3+ZWR45qLRgqMaGkt3p+8GyWypmlj8OKi0YbqlUPaQL3FIxs3xyUGlBudHsL7dUzCyHHFRaMFRrQ8lCZZ2KWypmlj8OKi2oPVBfWafiloqZ5Y+DSgvKtTaUrAzUu6ViZjnkoNKCyvrG7OwvSXQV5DEVM8slB5UWDNXo/oLKc+rdUjGz/HFQaUG5xoaSkGwq6RX1ZpZHDiotKNfY+wtIu7/cUjGz/HFQaUGtre/BLRUzyy8HlRbUWvwIHlMxs/xyUGlBpYerUPVb7CoUPPvLzHLJQaUFwxtKHtH9Ja9TMbNcclBpQa0NJcEtFTPLLweVFtQbU3FLxczyqqNBRdIqSdsl7ZC0tsZxSbomPX6vpLNGyyvpakmPS9qavi7OHHtfev52SW/oZN2g9oaSkGx/772/zCyPujp1YUlF4FrgQqAPuFvSxoi4P3PaRcCy9HUucB1wbhN5Px0Rn6i633LgEuAM4CTgO5JOi4ihTtWx0lJR9Yp6r1Mxs5zqZEvlHGBHROyMiAFgA7C66pzVwI2R2AzMkbSgybzVVgMbIqI/In4O7Eiv0zHlGk9+BK9TMbP86mRQWQg8lvncl6Y1c85oea9Ku8vWS5o7hvsh6QpJWyRt2b1791jqc4RaG0oCHNNdoH+wPNySMTPLi04GFdVIq/4rW++cRnmvA04FVgC7gE+O4X5ExPURsTIiVvb29tbI0ryhOt1fM3u6CKB/0K0VM8uXjo2pkLQUFmc+LwKeaPKcnnp5I+LJSqKkfwC+MYb7tVW5zjqVWT1FAA4MlDp5ezOzKaeTLZW7gWWSlkrqIRlE31h1zkbgsnQW2HnAnojY1ShvOuZS8Wbgvsy1LpE0Q9JSksH/uzpVOai/oeSs7kpQ6dgcATOzKaljLZWIKEm6CrgVKALrI2KbpDXp8XXAJuBikkH1A8DljfKml/64pBUkXVuPAH+Y5tkm6WbgfqAEXNnJmV8wsqJeVS2VmWlL5eCgg4qZ5Usnu7+IiE0kgSObti7zPoArm82bpr+jwf0+AnxkvOUdq7otlZ7k1+qWipnljVfUt2B4Q8kjBuo9pmJm+eSg0oJ6z1OZmY6pHHRLxcxyxkGlBVGn+6tYEMd0FzjgMRUzyxkHlRZUFs1Xt1Qgaa24pWJmeeOg0oKh4V2Kjzw2q6fLYypmljsOKi2ICAo6ckoxJAsg3VIxs7xxUGnBUDlqdn1BMgPMU4rNLG8cVFowFHHEUx8rZnY7qJhZ/jiotCCi9ngKJGMqhwaHhvcHMzPLAweVFgyV44jNJCtm9RQJYO8hD9abWX44qLSg3Kj7K11V//zBgYkskpnZpHJQaUG5wUB9Zfv75w4MTmSRzMwmlYNKC4YijlhNX1HZ/v75A26pmFl+OKi0oNxgoH5mulPxnoNuqZhZfjiotKCp7q/9bqmYWX44qLSg3KD7a2ZPkYLgqb39E1wqM7PJ46DSgqFy7c0kIUmfM6uHvucOTnCpzMwmj4NKC5IpxfWPz53VzWPPHZi4ApmZTTIHlRaUo/7iR4C5s3p47Fm3VMwsPxxUWtBoQ0mAubN7eHpfv3crNrPccFBpQaMV9ZC0VAD63AVmZjnR0aAiaZWk7ZJ2SFpb47gkXZMev1fSWaPllfS3kh5Iz/+KpDlp+hJJByVtTV/rOlk3qLRU6h+fO6sbwIP1ZpYbHQsqkorAtcBFwHLgUknLq067CFiWvq4Armsi723AKyLilcCDwPsy13s4IlakrzWdqdmIQ4NlZqYr52uZOztpqXiw3szyopMtlXOAHRGxMyIGgA3A6qpzVgM3RmIzMEfSgkZ5I+LbEVHZ+nczsKiDdWhof3+J2TO66h4/bkYXM7oKPPasg4qZ5UMng8pC4LHM5740rZlzmskL8HvANzOfl0r6saQ7JJ1fq1CSrpC0RdKW3bt3N1eTOvYPDDGrp35QkcSiuTM9A8zMcqOTQaXWaEP1E6vqnTNqXknvB0rAv6RJu4CTI+JVwHuAL0g6/oiLRFwfESsjYmVvb+8oVWhsf3+JY2fU7/4CWDR3Fn3Pu6ViZvnQyaDSByzOfF4EPNHkOQ3zSnon8Ebg7RERABHRHxHPpO/vAR4GTmtLTeo4MFBiVoPuL4DF89xSMbP86GRQuRtYJmmppB7gEmBj1TkbgcvSWWDnAXsiYlejvJJWAe8F3hQRw00ASb3pAD+STiEZ/N/Zwfqxr7/EsaMElZf1Hsueg4M88bwDi5kd/Rr/RWxBRJQkXQXcChSB9RGxTdKa9Pg6YBNwMbADOABc3ihveunPADOA25QsPNyczvR6HfBhSSVgCFgTEc92qn5D5eDQYHl4N+J6zn7pPAB+9OhznDRnZqeKY2Y2JXQsqABExCaSwJFNW5d5H8CVzeZN019W5/xbgFtaKe9Y7B9IJqCN1lI5fcFxHNNd4J5fPMcbX3nSRBTNzGzSeEX9OB3oT7ZeaTSlGKC7WODMRXP40aPPT0CpzMwml4PKOO3rT1oqo3V/AZz10rlse3wPhwa9B5iZHd0cVMZpf39z3V8AZ588l1I5uLdvT6eLZWY2qTo6pnI0q4ypNFr8CPCFHz463Kr5x+/vZMdT+wB427knd7aAZmaTwC2Vcdqfjqk001I5dkYXC+fM5KePu6ViZkc3B5VxOlBpqYyyor7irJPnsGvPIXbt8XoVMzt6OaiM074xjKkAnLloDkWJH/3iuU4Wy8xsUjmojFNlSnEzs78AZs3o4vQFx7G1bw+lcrmTRTMzmzQOKuM0MqW4+bkO5yyZx/7+Elu9ZsXMjlIOKuN0YKDEzO4ixUaPfqzyshOPZeGcmXzvwd2UhtxaMbOjj4PKOO3rHxp1NX01Sfzay0/k2f0DfG1r9YbNZmbTn4PKOB0YGP1ZKrWcvuA4TppzDB/71gPsOTjYgZKZmU0eB5Vx2t9fGtN4SkVB4s0rFvH0vn4++s0HOlAyM7PJ46AyTvv7h5qeTlxt4dyZ/P75p3DTXY9y/Z0Pt7lkZmaTx9u0jNP+gRLzZveMO/+f/fppPP7cQf5m0wM8s3+AtatOJ30+jJnZtOWgMk77+0ssnjdr3PlvuedxXn3qCeze189n79jJlkee4zdXLOQdr35pG0tpZjaxHFTGaX//ELObXPhYT0Fi9ZknceyMLv71gafYvbef1y8/kQUv8hMizWx68pjKOO0fKI15SnEtknj9L72Yt/7KYv5jzyH+8yfu4FO3PTi8uNLMbDpxS2UcIoL9/SVmj2P2Vz1nLprD4rmz+Nl/vMA1tz/EF374C/7ogpdxya8sbkvwMjObCG6pjEN/qUw5Rn+U8FjNm93Da06dzx/96qkcd0w3f/WN+zn7r2/jnevv4vHnvbuxmU19/ifwOFS6pmaPY/FjMxbPm8UfnH8Kjz57gH97aDd3Prib13z0X1lywixOf8nxLO2dzdL5szll/mxO7T2WuS3MQjMza6eOBhVJq4C/A4rAP0bER6uOKz1+MXAAeFdE/KhRXknzgC8CS4BHgN+JiOfSY+8D3g0MAf89Im7tRL0qOxS3s/urlpPnzeJt576UZ/cPcP8Te3jkmQPc84vnuO3+JxmKGD7vxcfPYPHcWZx4/AxOPO6YkZ/HzeDFxyc/58zq9pRlM+u4jv1VlFQErgUuBPqAuyVtjIj7M6ddBCxLX+cC1wHnjpJ3LXB7RHxU0tr083slLQcuAc4ATgK+I+m0iBhqd91GWioT09CbN7uH1y7r5bXLks9D5eD5AwM8vW+Ap/Ye4j/2HOKZ/QM88swB9h4apL905GaVPcUCvcfNYP5xM5jdU2Rmd5Fj0p8zu4vM6ilyTHeRmWlaT1eB7mKB7qLoKabvuwr0FAv0dCk9lrx6igW607Se9JyuggggAoKRAChEQckEhcpPMzt6dPKv4jnAjojYCSBpA7AayAaV1cCNERHAZklzJC0gaYXUy7sauCDNfwPwPeC9afqGiOgHfi5pR1qGH7S7YsWCeOWiF9F73OR0OxUL4oRjZ3DCsTN4+UuOO+L4QKnM3kODvHCoxN5Dg+zN/NzXX+L5AwMMDpUZLAUDQ+Xk/VCZwaGocbfOk0AkU6yVBprs5+F0oFBQw3PbFaKmerCbisWbamVS2/5raJ+p9Dv6tZefyNVvOqPt1+1kUFkIPJb53EfSGhntnIWj5H1xROwCiIhdkk7MXGtzjWsdRtIVwBXpx32StjdboWpf/5PDPs4Hnh7vtaYQ12PqOVrq4npMIXcCHxp/Xequ0u5kUKkVk6v/KVzvnGbyjud+RMT1wPWjXGvMJG2JiJXtvu5Ecz2mnqOlLq7H1NOJunRySnEfsDjzeRFQ/RCReuc0yvtk2kVG+vOpMdzPzMw6qJNB5W5gmaSlknpIBtE3Vp2zEbhMifOAPWnXVqO8G4F3pu/fCXwtk36JpBmSlpIM/t/VqcqZmdmROtb9FRElSVcBt5JMC14fEdskrUmPrwM2kUwn3kEypfjyRnnTS38UuFnSu4FHgbekebZJuplkML8EXNmJmV8NtL1LbZK4HlPP0VIX12Pqaf9QQMTkzPgxM7Ojj7dpMTOztnFQMTOztnFQaZGkVZK2S9qRrvCf0iQ9IumnkrZK2pKmzZN0m6SH0p9zM+e/L63bdklvmLySg6T1kp6SdF8mbcxll3R2+jvYIekaTfBKxzr1uFrS4+n3slXSxdOgHoslfVfSzyRtk/Q/0vRp9Z00qMd0/E6OkXSXpJ+kdflQmj5x30lE+DXOF8kkgoeBU4Ae4CfA8sku1yhlfgSYX5X2cWBt+n4t8LH0/fK0TjOApWldi5NY9tcBZwH3tVJ2klmBryZZ2/RN4KIpUI+rgT+vce5UrscC4Kz0/XHAg2l5p9V30qAe0/E7EXBs+r4b+CFw3kR+J26ptGZ4K5qIGAAq28lMN6tJtrwh/fmbmfQNEdEfET8nmaV3zsQXLxERdwLPViWPqexK1jYdHxE/iOT/nBszeSZEnXrUM5XrsSvSDWAjYi/wM5JdLKbVd9KgHvVMyXoARGJf+rE7fQUT+J04qLSm3jYzU1kA35Z0j5Ita6Bq6xsgu/XNVK/fWMu+MH1fnT4VXCXp3rR7rNI9MS3qIWkJ8CqSfxlP2++kqh4wDb8TSUVJW0kWht8WERP6nTiotGY828lMttdExFkkO0RfKel1Dc6djvWraOcWQBPhOuBUYAWwC/hkmj7l6yHpWOAW4E8j4oVGp9ZImzJ1qVGPafmdRMRQRKwg2VXkHEmvaHB62+vioNKaabc1TEQ8kf58CvgKSXfWdN76Zqxl70vfV6dPqoh4Mv1jUAb+gZFuxildD0ndJH+I/yUivpwmT7vvpFY9put3UhERz5Ps4r6KCfxOHFRa08xWNFOGpNmSjqu8B34duI/pvfXNmMqeNv33Sjovnc1yWSbPpKn8D596M8n3AlO4Hul9Pwf8LCI+lTk0rb6TevWYpt9Jr6Q56fuZwOuBB5jI72QiZyYcjS+SbWYeJJk18f7JLs8oZT2FZKbHT4BtlfICJwC3Aw+lP+dl8rw/rdt2JngmS43y30TSDTFI8i+pd4+n7MBKkj8QDwOfId1ZYpLr8U/AT4F70//RF0yDeryWpEvkXmBr+rp4un0nDeoxHb+TVwI/Tst8H/CBNH3CvhNv02JmZm3j7i8zM2sbBxUzM2sbBxUzM2sbBxUzM2sbBxUzM2sbBxUzM2sbBxUza4qkd0n6zGSXw6Y2BxWzo4ikrskug+Wbg4rlRrpNzf9NH2B0n6S3Knlo2fz0+EpJ30vfXy3pBknfTs/5LUkfTx9a9K10r6h697lY0gOS/i19uNE30vRzJP27pB+nP1+epr9L0lclfV3SzyVdJek96XmbJc1Lzzs1vfc9kr4v6fQ0/fOSPiXpu8DHJK1I890r6SuV3XUlfU/SyvT9fEmPZO7/5fTaD0n6eKYul0t6UNIdwGsy6W9Jf4c/kXRn+74lm+4cVCxPVgFPRMSZEfEK4FujnH8q8Bskz5z4Z+C7EfHLwME0/QiSjgE+S7LdxWuB3szhB4DXRcSrgA8Af5M59grgbSSbFn4EOJCe9wOSfZcArgf+JCLOBv4c+PtM/tOA10fEn5E8++K9EfFKkm1GPjhKPSHZifetwC8Db1XyNMQFwIdIgsmFJA90qvgA8IaIOBN4UxPXt5xwU9ny5KfAJyR9DPhGRHxfjZ+Q+s2IGJT0U5KnfFaC0E+BJXXynA7sjOSBR5Ds81V5bs2LgBskLSPZayrb2vluJA+I2itpD/D1zL1eqWRb9v8EfClT5hmZ/F+KiCFJLwLmRMQdafoNwJcaVTJ1e0TsAZB0P/BSYD7wvYjYnaZ/kSR4Afw/4POSbga+XON6llMOKpYbEfGgpLNJNgv835K+DZQYabEfU5WlP81XljQYIxvllan//06jKPVXJMHjzUoeBvW96ntlrt+fed+VlvH5SJ6TUcv+BvetGLWuqSFG6ldzc8CIWCPpXJIW21ZJKyLimSbKYEc5d39Zbkg6iaRb6Z+BT5A8J/4R4Oz0lN9uw20eAE5JgwYkXUoVLwIeT9+/aywXjeShUT+X9BZItmuXdGaN8/YAz0k6P016B1BptTzCSF3/WxO3/SFwgaQT0jGkt1QOSDo1In4YER8AnubwZ3JYjrmlYnnyy8DfSiqTbDv/R8BM4HOS/oKRR8iOW0QclPTHwLckPc3hz5/5OEn313uAfx3H5d8OXCfpL0m6zjaQPMag2juBdZJmATuBy9P0TwA3S3pHM/ePiF2SriYZ19kF/IikGxCS3+MykpbZ7XXKYTnkre/N2kzSsRGxL3240bXAQxHx6ckul9lEcPeXWfv9gaStJA9CexHJbDCzXHBLxWycJH0FWFqV/N6IuHUyymM2FTiomJlZ27j7y8zM2sZBxczM2sZBxczM2sZBxczM2ub/AyGbrHhRC4a2AAAAAElFTkSuQmCC"/>

* You can see that the number of users who have installed the game and never run it is significant.

* Some users can see how addicted (?) to the game after running it enough in the first week of installation.

* In the video game industry, **1-day retention** is a key metric for how fun and addicting a game is.

* With a high **1-day retention**, you can easily grow your subscriber base.



```python
# Look at the average of 1-day retention. 
df["retention_1"].mean()
```

<pre>
0.4452144409455803
</pre>
* You can see that less than half of the users played the game again the day after installation.



```python
# Look at the average of 1-day retention by group.
df.groupby('version')['retention_1'].mean()
```

<pre>
version
gate_30    0.448198
gate_40    0.442283
Name: retention_1, dtype: float64
</pre>
* Simply comparing the averages between groups, the number of plays is higher when the gate is 30 (44.8%) than when it is 40 (44.2%).

* It's a small difference, but it will affect retention and, ultimately, long-term returns.

* By the way, can this alone convince you that putting the gate at 30 is better than putting it at 40?



```python
# Look at the average of 7-day retention.
df["retention_7"].mean()
```

<pre>
0.1860557945624695
</pre>

```python
# Look at the average of 7-day retention by group.
df.groupby("version")["retention_7"].mean()
```

<pre>
version
gate_30    0.190183
gate_40    0.182000
Name: retention_7, dtype: float64
</pre>
* Simply comparing the means between groups, the survival rate is higher with the gate 30 (19.0%) than with the gate 40 (18.2%).

* It's a small difference, but it will affect retention and, ultimately, long-term returns.

* The difference is larger when the 7th is compared to the 1st. But does this alone convince me that putting the gate at 30 is better than putting it at 40?


-----


#### Bootstrapping 




```python
# Create a list of bootstrapped means values for each AB group.
boot_1d = []
for i in range(1000):
    boot_mean = df.sample(frac = 1,replace = True).groupby('version')['retention_1'].mean()
    boot_1d.append(boot_mean)
    
# Convert list to DataFrame.
boot_1d = pd.DataFrame(boot_1d)
    
# A Kernel Density Estimate plot of the bootstrap distributions
boot_1d.plot(kind='density')
```

<pre>
<AxesSubplot:ylabel='Density'>
</pre>
<img src="data:image/png;base64,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"/>

* The above two distributions express the bootstrap uncertainty that 1 day retention can have for both groups AB.

* Although small, there seems to be evidence of a difference.

* Let's plot the % difference to take a closer look.



```python
boot_1d['diff'] = (boot_1d.gate_30 - boot_1d.gate_40)/boot_1d.gate_40*100

ax = boot_1d['diff'].plot(kind='density')
ax.set_title('% difference in 1-day retention between the two AB-groups')

print('High probability of 1-day retention when the gate is at level 30:',(boot_1d['diff'] > 0).mean())
```

<pre>
High probability of 1-day retention when the gate is at level 30: 0.958
</pre>
<img src="data:image/png;base64,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"/>

* In the diagram above, the most likely % difference is around 1%-2%, with 95% of the distribution above 0%, favoring gates at level 30.

* Bootstrap analysis shows that the daily retention rate is more likely to be higher when the gate is at level 30.

* However, most players haven't reached level 30 yet, since players only played for one day.

* That is, most users would not have an effect on retention depending on whether the gate was at 30 or not.

* After playing for a week, you should also check the 7-day retention as more players reach levels 30 and 40.



```python
df.groupby('version')['retention_7'].sum() / df.groupby('version')['retention_7'].count()
```

<pre>
version
gate_30    0.190183
gate_40    0.182000
Name: retention_7, dtype: float64
</pre>
* As with 1-day retention, 7-day retention is lower at gate level 40 (18.2%) than at gate level 30 (19.0%).

* This difference is larger than the 1-day retention, probably because more players had time to open the first gate.

* Full 7-day retention is lower than Full 1-day retention. This is because fewer people play the game a week after installation than a day after installation.

* As before, let's use bootstrap analysis to see if there are any differences between the AB groups.



```python
boot_7d = []
for i in range(500):
    boot_mean = df.sample(frac=1,replace=True).groupby('version')['retention_7'].mean()
    boot_7d.append(boot_mean)
    
boot_7d = pd.DataFrame(boot_7d)

boot_7d['diff'] = (boot_7d.gate_30 - boot_7d.gate_40)/boot_7d.gate_40*100

ax = boot_7d['diff'].plot(kind='density')
ax.set_title('% difference in 7-day retention between the two AB-groups')

print('High probability of 7-day retention when the gate is at level 30:',(boot_7d['diff'] > 0).mean())
```

<pre>
High probability of 7-day retention when the gate is at level 30: 1.0
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZQAAAEICAYAAAB4YQKYAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAA13klEQVR4nO3dd3wcd5n48c+jblu9uUi25d7iElux48RxnEoqhgAhgTRa8B054A4OchzH5Sj3A47jyN0FQgIcEEoIkOKQhPQ4dmLHvcRxr5Jlq1jdsvrz+2NGznqzklby7s7u6nm/XnppZ+c7s89+d2aeme93iqgqxhhjzLlK8DoAY4wx8cESijHGmJCwhGKMMSYkLKEYY4wJCUsoxhhjQsISijHGmJAYUglFRO4SkTU+w80iMtF9PUxEnhaRBhH5o/vet0WkRkROeBXzuRCRS0Rkj0efXSIiKiJJXnx+JInIgyLyL2GY730i8ptQzzeaichhEbnS6zjM4ER9QhGRH4lInYisFZEin/c/LiL3n8u8VTVdVQ+6gx8GRgJ5qvoRERkLfAmYqaqjzuVzvKKqq1V12mCmFZGvuQm35++0iHSLSH6o4wynUCc2/50SAFVdoarfCsX8QyUWkpGI/FJEvh3G+auITA7X/P0+a4K7fvy4lzhOuetRjYj8XkSyIxFXpEV1QhGRhcACYBSwBvgn9/0s4MvAN0L4ceOBvara6TN8UlWrBjojcUR13fZHVf/dTbjpqpoOfA94TVVrvI7N11A4AjIx4Q6gDrhFRFIDjJ/rrkcTgRzgvlB8aNRta1Q1av+AjwL/z319DfCs+/p/gY8FMX0esBJoBNYD3wLW+IxXYDLwb0A70AE0A58FTgPd7vAv3fIXAm8C9cA2YJnPvF4DvgO84U47GZgOvAjUAnuAm33K/xJ4AHgGaALeAib5jJ/lM20l8DX3/QTgXuAAcBJ4DMjt5fsvA8p9hg/jJOLtQAPwByAtiHoU9/Pu7KNMIvADoAY4CHzOrd8kd/wngF3udz0IfNZn2reBG32Gk935zOvtOwFfBU4Aj/RVJ8BRN45m92+x+/4n3XjqgOeB8X7LxQpgnzv+AbcOZgCtQJc7r3qf3/LbPtN/Btjv/nYrgTH9zbuXOr0P+JP7OzUBm3E2TD3jxwB/BqqBQ8DnfdYV3+V5G3AZsMNn2peA9T7Da4AP9DXf/pY/oMT9fne69V4D/HMv3+1uN752N8ang1lGgRuArTjr4JvAnF7m/7obyyl3/h8FVgEfcscvccdf5w5fCWz1+Y5fB44AVcCvgax+1pEDwN/grKsf9hunwGSf4b8FXuhnXfpPt/4OAfdw9rr0Gu/d1lwEbHDrbANwkd96f6XfcvUbv9/sbqACOA58yafsQmAjzja0Evhhn/XQ38bEyz/gPHdBHwb8h/tXCrwY5PSPugv8CHdexwiQUPwrWX02XD7DRTgr0HXuAneVO1zg8yMfxUkESUAWUIazIU0C5rsLyCyfjVCt+4MlAb8FHnXHZfT8sECaO7zIHfdFYB1QDKQCPwV+38v39/8Oh3ES6xggF2eDuiKIelyKs1Km91FmBbAbGOvO+1W/leB6YBLOhvlSoAWY7477CvAHn3ktx2fjF+A7deIcMaW6y0avdcK7K0ySzzw+gLPBn+HW/deBN/2Wi78A2cA4nA3rNe64u/BZhnx+y2+7ry93f+f5biz/A7wezLwDfNf7cDa6H8ZJsl/G2cAk4yyDm3CO0lNw9nwPAu/rZXlOw9n45Lvf+QTOBiTDrcPTODtg/c03mLp+2J3nXKANmNHL9ztTb8Eso26dVgGLcDa6d7rlU3uZv/+G/JvA/7ivv4aTBL7nM+5+9/Un3eVjIpAOPA480seyf4n7PXPc33tlb3G4ZV4AvtnPuvSOW8c5OMnfP6H4bmtG4uyc3O4O3+oO5/nUaX8J5fc428nZOMvkle74tcDt7ut04MI+txXBbJi9/AP+HmcP6w84K8MbOBuCz+PshfwWyA4wXSLOyjjd571/Z/AJ5av+CxXOnu2dPj/yN33GfRRY7Vf+p8C/+qxMP/MZdx2w2319K7Cll/rYBVzhMzza/Z5JAcr6f4fDwG0+w98HHgziN/g57lFaH2VewSc5AVfjtyH3K/8k8AX39RicPfBMd/hPwFd6mW4Zzl6t715rr3VC4ITyHPApn+EEnAQ33me5WOIz/jHgXvf1XfSdUH4OfN9nXLobS0l/8w7wXe8D1vnFeRxnA7YIOOpX/p+A/wu0PLvvrQZuwjnSfsH97Gtwjl62u2X6m28wdV3sM349cEsv3+9MvQWzjAI/Ab7lV34PcGkv8/dPKFf4fM+/Ap/uqV+co5eb3NcvA3/rM900elnH3PE/A550Xy92yxb6xdGIc1TVhbPjVdTPuuR7BH8l700ovtua2/E52nTfWwvc5VOn/SUU3+3k94Gfu69fx2nBye9r/e/5i562t16o6n+p6lxV/SjuRhpnxbobZwHZhXMI7q8AZyEv83nvyDmEMh74iIjU9/zhHDaP9ilT5ld+kV/5j+P0B/XwPXusBWfjA85e/oE+4njCZ567cBbSkUF+j94+MyARGQZ8BPiVz3uX+HTW73TfHkMfdS0i14rIOhGpdeO+DmcHAVWtwNlR+JDbWXktzo5Cb6pVtdVneKB1Mh6436d8Lc6RU5FPmQHVk48x+Hx3VW3GOZId7LzP1KmqduM0941xv8MYv+Xra/S9HKzCSchL3dev4RwtXuoOE8R8g6nrwdZdf9OPB77kF9tYnPoIxlpgqoiMBObhNGWNdU80WYiz8QS/39B93XMkcBaf9eO3AKq6Fufo4WN+ReerajbOkeJPgNUikhbkulTGe/m+5x9vT8xFBM9/3e2p008BU4HdIrJBRG7oayYx06HpLgSfxdm7uhFnT6NDRDYAXwgwSTVO08hYnD0CcJoYBqsM5wjlM32UUb/yq1T1qkF+1q19jPukqr4xiPkOxk04G9zXet5Q1dW8dyNxHKeue5ypa7eT8s84HZdPub/bkzgb8R6/wtljTALWquqxPmJSv+Fe60RExgeYvgz4jqr2lbSC/Wx/FTgbvp7PH4HTlNTX9+nLmTp1O1+L3c/oBA6p6pQBxLkKp23+KPBdnGaRh3Gaax5wy5T1M9++6rqkvy8TRIx96fndvjPA6ZwPU20RkU0424u3VbVdRN4E/gE4oO+ecHLWb4izLHfi9CH4+yCQCfxYRP7HfS8bZ1n/UYAYOkTkZ+648/pYl4p9hsfyXr515x9vT8x/dV+fAob7jAt01qr/drLCjXcfcKu77N0E/ElE8lT1VIB5RP8Rio8f4jQXteC0I18gIuk4e1wH/QurahdO2+d9IjJcRGbitLkO1m+AG0XkfSKS6O5dLBOR4l7K/wVnb+h2EUl2/y4QkRlBfNZfgFEi8kURSRWRDBFZ5I57EPhOz4ZSRApEZPk5fK/+3An8Wt3j3z48BnxeRIpFJIezjxpTcNrbq4FOEbkWp0nM15M4beRfwNlzHIi+6qQa5+SKiX7l/0lEZrnls0TkI0F+ViVQLCIpvYz/HfAJEZnnJtJ/B95S1cMD+kbvWiAiN7lns30RZ+O/DqcpqVFEvirONVSJInKeiFzgE2eJ3xlAb+I03yzEaSLZiXskzbt75/3NN5TLXyVn/y79eRhYISKL3LObRojI9SKSMYD5r8Lp5O45InvNbxic/oS/F+dU4HSc3/AP+u4ZoL7uBH6B0/cwz/27GJgnIrP9C4tIIk6/6mkCbLdcjwFfEJEi94j9q72U6/EszrbmYyKSJCIfBWbibEfAOYnhFncbVIrTJ+fvX9zt5Cw3vj+48d4mIgXu0XG9W7art0BiIqGIyGU4/SRPAKjqepyzo8pw2n+/28uk9+Bk/xM47bX/N9gYVLUMp7P4azgbqTLgH+mlDlW1CWejeQtOtj/Bux3J/X1WE06n/43udPtwvifA/ThnDr0gIk04G5dFgeZzrsS57udygtvAP4zTp7QN52ykx3tGuN/n8zgrSh1Oc8BK34lV9TTOUcwE32mD1GuduDsg3wHecJtJLnSXo+8Bj4pII85ZZtcG+VmvADuBEyLynlOoVfVl4F/c73Ic50SEWwb4fXw9hdPUW4fTVn6Tqna4O0w34mzADuGcCPAznJNBAP7o/j8pIpvd2E7h/DY7VbXdHb8WOKLu6fFBzDeUy9/PgZnu7/Jkf4VVdSPOGXT/i1Mf+3H6tHpzH/Ard/43u++twjkR4fVehsFJEI+47x3CObPv7/xn7q4fVwA/UtUTPn+bcI4OfHdgt4lIsxv3ncAHVbW2l7gfxunj2g5swUkYnfSyIVfVkzhnv30Jp3n1K8ANPkdc/4KzHNbh9If8LsBsVuHU58vAD1T1Bff9a4Cdbuz34/SHtQaY3qmT/nc8jYkMEfkGMFVVb/M6FmOihXtE/6CqBmq+Pdd5l+CeOdjLEdiAxMQRiol/IpKL0wH4kNexGOMlt6nxOrf5qgj4V+AJr+MKhiUU4zkR+QxOE+Jzqvp6f+WNiXOC0zRVh9PktYvQ3hUkbKzJyxhjTEjYEYoxxpiQiJnrUIKRn5+vJSUlXodhjDExY9OmTTWqWhCKecVVQikpKWHjxo1eh2GMMTFDRM7lDiJnsSYvY4wxIWEJxRhjTEhYQjHGGBMSllCMMcaEhCUUY4wxIWEJxRhjTEhYQjHGGBMScXUdijFeUFVW76thW1k94/NHcM2sUaQk2b6aGXrCmlBE5Bqce+gn4jw//bt+4z/Ouw+PaQb+RlW3ueMO4zxnvAvoVNXScMZqzGA0tXbwud9t4fW91WfemzE6k4fvWEBxzvA+pjQm/oRtN8p9MtkDOA8umonzGMmZfsUOAZeq6hzgW7z31uWXqeo8SyYmGnV0dfOpX27kzf013HfjTHZ98xoevG0Bx+pauOPn66lvae9/JsbEkXAely8E9qvqQffpcI/iPPHwDFV9U1Xr3MF1nP0cZWOi2g9f3Mv6w7X84CNzueviCQxLSeSa80bxi7su4GhtC/+6cqfXIRoTUeFMKEU4z7joUe6+15tPAc/5DCvOY0Y3icjdvU0kIneLyEYR2VhdXd1bMWNC6u1jDTy46gC3LhzLB84/e7EuLcnlnssn89TWCt7c/56nBBsTt8KZUCTAewEfvuI+M/5TvNufAnCxqs7HaTL7nIgsDTStqj6kqqWqWlpQEJIbZhrTJ1Xlm0+/Q87wFO69dkbAMisuncTorDT+44U92DOHzFARzoRSDoz1GS4GKvwLicgc4GfAclU92fO+qla4/6twHn+5MIyxGhO0V/dUsf5wLV+6eipZw5IDlklLTuQLV0xhy9F6XttjR85maAhnQtkATBGRCSKSAtwCrPQtICLjgMeB21V1r8/7I0Qko+c1cDXwdhhjNSYoqsr/vrKfouxh3Fw6ts+yH1pQzKjMNH7xxqEIRWeMt8KWUFS1E7gHeB7nmciPqepOEVkhIivcYt8A8oAfi8hWEel5mMlIYI2IbAPWA8+o6l/DFasxwXrrUC2bj9az4tKJJCf2vfokJyZw24XjWL2vhv1VTRGK0BjvxNUz5UtLS9UesGXC6fafv8Wu442s+erlpCUn9lv+ZHMbi7/7Ch9bOI773j8rAhEaMzAisilUl2bY5bzGBGnPiSZW76vhExdPCCqZAOSlp3LVjJE8va2Cjq7uMEdojLcsoRgTpN+sO0JKUgK3Lhw3oOk+eH4RJ0+1n3U1vTHxyBKKMUFobuvk8c3l3DBnNLkjUgY07aXTCsgdkcLjW46FKTpjooMlFGOC8MSWY5xq7+KOxSUDnjY5MYEb54zmxXcqaWrtCH1wxkQJSyjG9ENV+c3aI8wuymJucdag5nHj3DG0d3bzql2TYuKYJRRj+rH+UC17Kpu4/cLxiAS6AUT/5o/LIT89led3nghxdMZED0soxvTjkXVHyExL4sa5YwY9j4QE4aqZI3ltdxWtHV0hjM6Y6GEJxZg+VDW18te3T/CR0rEMSwnuVOHeXHPeKE61d/GG3TDSxClLKMb04Q/ry+jsVj6+aGCnCgeyeGIeGWlJ1uxl4pYlFGN60dnVze/WH+WSKflMLEg/5/mlJCVw+fRCXtpVRVd3/NyhwpgellCM6cWqvdUcb2gNydFJj8unF1J7qp3t5fUhm6cx0cISijG9+P36MvLTU7lixsiQzXPplAISBDt92MQlSyjGBFDZ2Mqre6r48ILifu8qPBA5I1I4f1wOq/ZUhWyexkQLSyjGBPDHjWV0dSu3XND3M08G47JpBWwrb6C6qS3k8zbGS5ZQjPHT3a08uqGMxRPzKMkfEfL5L5tWCGA3izRxxxKKMX7W7K+hvO40tywM/dEJwKwxmRRmpPKqNXuZOGMJxRg/j244SvbwZN43a1RY5i8iLJtWwOt7q+m0Z6SYOGIJxRgftafaefGdSm46vzjoh2gNxrJphTS2drKlrD5sn2FMpFlCMcbHMzuO09GlfGhBUVg/Z8mUfBIThFd3W7OXiR+WUIzx8eSWY0wdmc7M0Zlh/ZzMtGQWjM9hlXXMmzhiCcUY19GTLWw6UscHzi8a9G3qB+LSqQXsrGi004dN3LCEYozrqa3OI3qXzwtvc1ePpVMKAFiz345STHywhGKMa+W2ChZOyKUoe1hEPm/WmEzyRqTw+l67nb2JD5ZQjAEOVDezr6qZ62ePjthnJiQIS6bks3pfNd1292ETByyhGAO8+E4lAFfNDN2NIIOxdEoBNc3tvHO8MaKfa0w4WEIxBnhh5wnOK8pkTISau3pcMjUfgNf3WT+KiX2WUMyQV9XUypayeq6eGZ4r4/tSmJHGjNGZdl8vExcsoZgh7+VdVahGvrmrx9Kp+Ww6Useptk5PPt+YULGEYoa8l96ppDhnGNNHZXjy+ZdOKaCjS1l74KQnn29MqFhCMUNaR1c36w6e5NKpBRG5mDGQBSU5DEtOtH4UE/PCmlBE5BoR2SMi+0Xk3gDjPy4i292/N0VkbrDTGhMK28rqOdXexZLJ+Z7FkJqUyOJJedaPYmJe2BKKiCQCDwDXAjOBW0Vkpl+xQ8ClqjoH+Bbw0ACmNeacrdlfgwgsnpTnaRyXTMnn8MkWjp5s8TQOY85FOI9QFgL7VfWgqrYDjwLLfQuo6puqWucOrgOKg53WmFB4Y38Ns4uyyB6e4mkcS6c6t2FZZc1eJoaFM6EUAWU+w+Xue735FPDcQKcVkbtFZKOIbKyutpXRBO9UWydbjtZzsYfNXT0m5o+gKHuYNXuZmBbOhBKohzPg/SVE5DKchPLVgU6rqg+paqmqlhYUFAwqUDM0rT9US2e3etp/0kNEWDq1gLUHTtJhT3E0MSqcCaUc8H0odzFQ4V9IROYAPwOWq+rJgUxrzLl480ANKUkJLBif43UoAFw6NZ/mtk42H6nrv7AxUSicCWUDMEVEJohICnALsNK3gIiMAx4HblfVvQOZ1phztfFIHXOLs8L6qN+BuGiy8xRHO33YxKqwJRRV7QTuAZ4HdgGPqepOEVkhIivcYt8A8oAfi8hWEdnY17ThitUMPa0dXbx9rIEF43O9DuWMzLRkzh+bbbezNzErKZwzV9VngWf93nvQ5/WngU8HO60xobK9vIGOLqU0Spq7eiydWsB/vbSXk81t5KWneh2OMQNiV8qbIWnjkVoA5kdhQlF1ro8xJtZYQjFD0qbDdUwsGEHuCG+vP/HnXBOTbM1eJiZZQjFDTne3suloXdQ1dwEkJghLJjtPcVS1pzia2GIJxQw5B2uaqW/poDSKOuR9LZ1aQFVTG7tPNHkdijEDYgnFDDmbj9YDMH98tqdx9GbpFOcCXbtq3sQaSyhmyNlR3kB6ahIT89O9DiWgUVlpTBuZYdejmJhjCcUMOduPNTBrTCYJCd48/yQYS6fms+FQHS3t9hRHEzssoZghpaOrm13HG5lTnOV1KH1aOrWA9q5u3jpY63UoxgTNEooZUvZWNtHe2c15RdGdUC4oySUtOYFV1o9iYoglFDOk7ChvAGBOcba3gfQjLTmRRRPyrB/FxBRLKGZI2XGsgYy0JMbnDvc6lH5dMiWfg9WnKK+zpzia2GAJxQwpO441cN6YrKjukO9x6dSe04ftqnkTGyyhmCGjvbOb3cebor5DvsfkwnRGZ6XZ9SgmZlhCMUPG3som2ru6mR0jCUVEWDqlgDcO1NBpT3E0McASihkydhxzOuRnR/kZXr6WTi2gqbWTrWX1XodiTL8soZghY3t5A5lpSYyLgQ75Hksm55MgdhsWExssoZghY8exemYXZyES/R3yPbKGJzN3bDar9lnHvIl+llDMkNDW2cWeE03MLsr2OpQBWzqlgO3l9dS3tHsdijF9soRihoQ9J5ro6NKYOcPL15Ip+ajCOrsNi4lyllDMkBCLHfI95hZnMyw5kTcPWLOXiW6WUMyQsKO8gezhyRTnDPM6lAFLSUpg4YRc3jxw0utQjOmTJRQzJOw41sDsotjqkPd10aQ89lc1U9nY6nUoxvTKEoqJe60dPR3ysdfc1ePiyfkArLWjFBPFLKGYuLfnRBOd3bHZId9jxuhMsoYl88Z+60cx0csSiol7290O+Wh/BkpfEhOExRPzePPASVTV63CMCcgSiol7O8rryRmeTFF27HXI+7poch7H6k9ztNZuZ2+ikyUUE/e2lzcwpzg7Zjvke1w0yelHsbO9TLSyhGLi2un2LvZVNcd0/0mPSQUjKMxItX4UE7UsoZi49s7xRrq6NabP8OohIlw8OZ+11o9iopQlFBPXdpTXA9H/DPlgXTQpj5On2tlT2eR1KMa8R1gTiohcIyJ7RGS/iNwbYPx0EVkrIm0i8mW/cYdFZIeIbBWRjeGM08Sv7ccaKMhIZWRmqtehhMRF7vUob+63fhQTfcKWUEQkEXgAuBaYCdwqIjP9itUCnwd+0MtsLlPVeapaGq44TXzbUd7AnBi+Qt5fUfYwSvKGW8e8iUrhPEJZCOxX1YOq2g48Ciz3LaCqVaq6AegIYxxmiDrV1sn+6uaYeeRvsBZNyGPD4Vq6u60fxUSXcCaUIqDMZ7jcfS9YCrwgIptE5O7eConI3SKyUUQ2VlfbU+3Mu3ZWNKJKXJzh5WvRxFwaTnew+4T1o5joEs6EEqiNYSC7VBer6nycJrPPicjSQIVU9SFVLVXV0oKCgsHEaeLUdrdDPpavkA9k0cQ8AN46ZM1eJroElVBE5M8icr2IDCQBlQNjfYaLgYpgJ1bVCvd/FfAEThOaMUHbXt7A6Kw0CjPSvA4lpIqyh1GcM4y37IFbJsoEmyB+AnwM2Cci3xWR6UFMswGYIiITRCQFuAVYGcyHicgIEcnoeQ1cDbwdZKzGAO/esj4eLZqQx/rDtXY9iokqQSUUVX1JVT8OzAcOAy+KyJsi8gkRSe5lmk7gHuB5YBfwmKruFJEVIrICQERGiUg58A/A10WkXEQygZHAGhHZBqwHnlHVv57bVzVDScPpDg7VnIq7/pMeiybmUnuqnX1VzV6HYswZScEWFJE84DbgdmAL8FtgCXAnsCzQNKr6LPCs33sP+rw+gdMU5q8RmBtsbMb423K0DoDzx+V4HEl4XDjB7Uc5eJKpIzM8jsYYR7B9KI8Dq4HhwI2q+n5V/YOq/h2QHs4AjRmMzUfrSRCYOzbb61DCYmzuMEZnpbHukPWjmOgR7BHKz9yjjTNEJFVV2+yiQxONNh+pY9qoTNJTgz4IjykiwqIJuazZ79zXK14u3DSxLdhO+W8HeG9tKAMxJlS6upWtZfXMH5ftdShhtWhiHjXNbRysOeV1KMYA/RyhiMgonIsRh4nI+bx7bUkmTvOXMVFnX1UTzW2dLBgfn/0nPRZNyAXgrYO1TCqwlmfjvf7aA94H3IXTcf5Dn/ebgK+FKSZjzsmmI06H/Pw47ZDvMSF/BPnpqbx16CQfWzTO63CM6TuhqOqvgF+JyIdU9c8RismYc7L5SD25I1IYnxffB9EiwqKJubx1sNb6UUxU6K/J6zZV/Q1QIiL/4D9eVX8YYDJjPLX5aB3zx8X+I3+DceGEXJ7ZfpyjtS2MzxvhdThmiOuvU75nCU0HMgL8GRNVKhtbOVRzioVu/0K8O3NfL7sNi4kC/TV5/dT9/2+RCceYc7PuoHPDxMUT8z2OJDKmFKaTOyKFdYdOcvMFY/ufwJgwCvbCxu+LSKaIJIvIyyJSIyK3hTs4YwZq7YGTZKQlMXNMptehRISIsLAk145QTFQI9jqUq1W1EbgB5y7CU4F/DFtUxgzSuoMnWTQhl8SE+O8/6bFoYi7H6k9zrP6016GYIS7YhNJzA8jrgN+rqu0OmahzvOE0h0+2cKHbrzBULPK5r5cxXgo2oTwtIruBUuBlESkAWsMXljEDt9Z9zvriSUMroUwflUHWsGRr9jKeC/b29fcCi4FSVe0ATuH3fHhjvLb2wEmyhiUzY9TQ6D/pkZAgXFCSa09wNJ4byJ3zZuBcj+I7za9DHI8xg6KqvL6vmosm5ZEwhPpPelw4MZeXdlVS2djKyMz4ekKliR3BnuX1CPADnOefXOD+2V2GTdTYdbyJysY2Lpte6HUonujpR1ln/SjGQ8EeoZQCM9WeN2qi1Kt7qgBYNq3A40i8MXNMJhmpSbx1qJbl84q8DscMUcF2yr8NjApnIMaci1d3VzG7KIvCjKHZ3JOYIJSW5NiZXsZTwSaUfOAdEXleRFb2/IUzMGOCVd/SzuajdVw2RI9OeiyamMeB6lNUN7V5HYoZooJt8rovnEEYcy5W7a2mW2HZEO0/6dHzfJT1h2q5fs5oj6MxQ1Gwpw2vAg4Dye7rDcDmMMZlTNBe2FlJ3ogU5hZnex2Kp84rymJ4SqKdPmw8E+xZXp8B/gT81H2rCHgyTDEZE7Sm1g5e2lXJDXNGD6nbrQSSnJjAgvE5doGj8UywfSifAy4GGgFUdR8wtNsXTFR4YWclbZ3dvN/ObALgwol57KlsovZUu9ehmCEo2ITSpqpnllD34kY7hdh47qltFRTnDGP+uGyvQ4kKC336UYyJtGATyioR+RowTESuAv4IPB2+sIzpX3VTG2/sr2H5vDFD4umMwZhTnEVqUoL1oxhPBJtQ7gWqgR3AZ4Fnga+HKyhjgvHsjuN0datdyOcjNSmR+eOsH8V4I6jThlW1W0SeBJ5U1erwhmRMcJ7aeozpozKYOtKeRu1r0cRc7n95Hw0tHWQNT+5/AmNCpM8jFHHcJyI1wG5gj4hUi8g3IhOeMYEdPdnC5qP1dnQSwKIJeajChsN2lGIiq78mry/inN11garmqWousAi4WET+PtzBGdObp7dXAHDjXLuAz9/547JJSbR+FBN5/SWUO4BbVfVQzxuqehC4zR1nTMSpKk9uOcYFJTkU5wz3Opyok5acyLyx2bxlZ3qZCOsvoSSrao3/m24/Sr+NsyJyjYjsEZH9InJvgPHTRWStiLSJyJcHMq0ZunafaGJfVbNde9KHRRNzeftYA02tHV6HYoaQ/hJKX1dH9XnllIgkAg8A1wIzgVtFZKZfsVrg8zjPWhnotGaIemprBUkJwvWzrbmrN4sn5tGt2NleJqL6SyhzRaQxwF8TMLufaRcC+1X1oHtR5KP4PTZYVatUdQPgvxvV77RmaOruVp7eVsElU/LJHZHidThRa0FJDmnJCazZ/54GBmPCps+EoqqJqpoZ4C9DVftr8ioCynyGy933ghH0tCJyt4hsFJGN1dV2RnO823S0jmP1p+3srn6kJiWyaEIeq/fZOmEiJ9gLGwcj0KXLwd6uJehpVfUhVS1V1dKCgqH9PIyh4Kmtx0hLTuCqmSO9DiXqXTIlnwPVp6ioP+11KGaICGdCKQfG+gwXAxURmNbEqY6ubp7ZfpyrZo5iRGqwj/IZupZMyQdgzT5r9jKREc6EsgGYIiITRCQFuAUI9imP5zKtiVNr9tVQ19LB8rljvA4lJkwbmUFBRiqrrR/FREjYdvNUtVNE7gGeBxKBX6jqThFZ4Y5/UERGARuBTKBbRL4IzFTVxkDThitWExue2nqMrGHJLJ1qTZvBEBGWTM53nmjZrSQM8efFmPALa7uBqj6LcyNJ3/ce9Hl9Aqc5K6hpzdB1ur2LF96pZPm8MaQkhfPAOr4smZzPE1uO8c7xRs4ryvI6HBPnbM00MeHVPVW0tHdx4xxr7hqIM/0o1uxlIsASiokJf9leQX56Cosm5nkdSkwZmZnG1JHp1jFvIsISiol6p9o6eWV3FdeeZ8+NH4ylUwpYf6iW5rZOr0Mxcc4Siol6L++uorWjmxvm2K1WBuPyGYW0d3XbUYoJO0soJur9ZVsFhRmpXFCS63UoMemCklwy0pJ4eVel16GYOGcJxUS1ptYOXttbzXWzR9tpr4OUnJjAsmmFvLqniu7uYG9WYczAWUIxUe2V3VW0d1pz17m6ckYhNc3tbCuv9zoUE8csoZio9sI7leSnpzJ/XI7XocS0S6cWkJggvLyryutQTByzhGKiVntnN6/vqeaK6YXW3HWOsoensGB8Di9ZP4oJI0soJmptOFxLU1snV8wo9DqUuHDljEJ2n2iivK7F61BMnLKEYqLWi+9UkpqUcOZqb3Nurpzh3PL/hZ12lGLCwxKKiUqqysu7K7l4cj7DU+xW9aEwsSCd6aMyeHbHca9DMXHKEoqJSvuqmimrPX1mr9qExvWzR7PxSB0nGlq9DsXEIUsoJir1dB5b/0loXeeefv3c23aUYkLPEoqJSi/vqmJ2URYjM9O8DiWuTLJmLxNGllBM1Gls7WBrWT3LptmDtMLhOrfZq7LRmr1MaFlCMVFn7YGTdHUrSybb2V3hcN3s0ajCc3aUYkLMEoqJOmv21TA8JZHz7er4sJhc6DR7Pb3dEooJLUsoJuqs3lfNhRPz7FG/YbR8XhGbjtRx5OQpr0MxccTWWBNVympbOHyyxZq7wuwD549BBJ7YcszrUEwcsYRiokrPs8+XTrWEEk6js4axeGIeT2w5hqrd0t6EhiUUE1XW7KthVGYakwrSvQ4l7t00v5gjJ1vYfLTO61BMnLCEYqJGV7fyxoEalkzJR8TuLhxu15w3irTkBB7fbM1eJjQsoZiosbOigfqWDi6xm0FGRHpqEu+bNYqnt1XQ1tnldTgmDlhCMVFj9T6n/+Ri65CPmA+eX0Rjayev2IO3TAhYQjFRY/W+amaMziQ/PdXrUIaMJZPzGZmZymMby7wOxcQBSygmKrS0d7LpSJ01d0VYUmICH1kwllV7q6moP+11OCbGWUIxUeGtQ7V0dNntVrxwc+lYuhX+uLHc61BMjLOEYqLCmn01pCQlsHBCrtehDDnj8oazZHI+j20so6vbrkkxg2cJxUSFNftqWFiSS1pyotehDEkfvWAsx+pPn7mw1JjBCGtCEZFrRGSPiOwXkXsDjBcR+W93/HYRme8z7rCI7BCRrSKyMZxxGm9VNbayp7LJnh3voatnjSRneDJ/2HDU61BMDAtbQhGRROAB4FpgJnCriMz0K3YtMMX9uxv4id/4y1R1nqqWhitO472e04Wt/8Q7qUmJfGh+MS++U0lNc5vX4ZgYFc4jlIXAflU9qKrtwKPAcr8yy4Ffq2MdkC0io8MYk4lCa/bXkDcihZmjM70OZUi7ZeFYOrqUxzdb57wZnHAmlCLA9+T2cve9YMso8IKIbBKRu3v7EBG5W0Q2isjG6urqEIRtIklVWb2vhosn55OQYLdb8dLkwgxKx+fw6IYyu2GkGZRwJpRAWwf/pbSvMher6nycZrHPicjSQB+iqg+paqmqlhYU2CNjY83uE03UNLdZ/0mUuGXhOA5Wn2LdwVqvQzExKJwJpRwY6zNcDFQEW0ZVe/5XAU/gNKGZOLPG7T+xCxqjww1zRpM9PJlH1h32OhQTg8KZUDYAU0RkgoikALcAK/3KrATucM/2uhBoUNXjIjJCRDIARGQEcDXwdhhjNR5Zvb+GyYXpjM4a5nUoBkhLTuTm0rE8v7OSEw2tXodjYkzYEoqqdgL3AM8Du4DHVHWniKwQkRVusWeBg8B+4GHgb933RwJrRGQbsB54RlX/Gq5YjTdaO7p46+BJO7sryty2aDzdqvxuvZ1CbAYmKZwzV9VncZKG73sP+rxW4HMBpjsIzA1nbMZ7m47U0dbZbU9njDLj8oazbGoBv19/lHsum0xKkl3/bIJjS4rxzOp9NSQnCosm5HkdivFzx+ISqpvaeH7nCa9DMTHEEorxzOp91cwfl8OI1LAeKJtBuHRqAeNyh/PI2iNeh2JiiCUU44ma5jZ2VjTa2V1RKiFBuO3Ccaw/XMvuE41eh2NihCUU44nX9jgXoS6bVuhxJKY3N5eOJTUpgV/bUYoJkiUU44lXdlcyMjOVWWPsdivRKnt4Cu+fO4Yntxyj4XSH1+GYGGAJxURce2c3r++t4fLphYjY7Vai2V0Xl9DS3sXv7RRiEwRLKCbi1h+qpbmtkyumj/Q6FNOPWWOyuGhSHr984zDtnd1eh2OinCUUE3Ev764kNSmBi+2CxpjwmaUTOdHYyjM7/O+cZMzZLKGYiFJVXt5VxUWT8hiWYk9njAXLphYwpTCdh14/ZHchNn2yhGIi6p3jjRytbeGqmaO8DsUESUT49CUT2HW8kTcPnPQ6HBPFLKGYiHpm+3ESE4T3zbL+k1iyfF4R+emp/PT1g16HYqKYJRQTMarKszuOc9GkPPLSU70OxwxAWnIin1xSwut7q9lytM7rcEyUsoRiImZnRSOHT7Zw/Wx7ynMsumNxCTnDk7n/5X1eh2KilCUUEzErt1WQlCBcPcv6T2JRemoSn1k6kdf22FGKCcwSiomIjq5uHt9czuXTC8kdkeJ1OGaQeo5SfvSSHaWY97KEYiLild1V1DS389ELxvZf2ESt9NQk7l46iVV7q9l42J47b85mCcVExGMbyijMSOXSqQVeh2LO0R2Lx1OYkcq3ntlFd7ddl2LeZQnFhF1ZbQuv7qniwwuKSUq0RS7WjUhN4ivXTGdbWT1PbTvmdTgmitjabcLu52sOkSDCHYtLvA7FhMhN5xcxpziL7z23h1NtnV6HY6KEJRQTVvUt7Ty2sYz3zxvDqKw0r8MxIZKQIPzrjbOobGrle3/d7XU4JkpYQjFh9ZPXDnC6o4vPLp3kdSgmxBaMz+ETF03g12uPsNZuyWKwhGLCqKL+NP/35mFuOr+YaaMyvA7HhME/vm8a4/OG85U/b7OHcBlLKCY8VJV/e3onAH9/1RSPozHhMiwlkR/ePI/j9a186bGtdtbXEGcJxYTFMzuO8/zOSv7hqqkU5wz3OhwTRgvG5/D162fw0q4q/vPFPV6HYzyU5HUAJv7sr2rm3j/vYG5xFp9eMsHrcEwE3HlRCbtPNPHAqwfIGZ7Cpy+Z6HVIxgOWUExIHW84zad+tYHUpAR+fNsCu+5kiBARvvPB2TSc7uDbz+ziVFsXn79iMiLidWgmgmxtNyGz+0QjN/90LbXN7Tx8ZylF2cO8DslEUGKC8N+3ns9N84v4r5f28vlHt9LYah31Q4kdoZhz1tbZxf+9cZgfvbSXjLRkfvPpRcwdm+11WMYDyYkJ/OdH5jKpIJ0fvriXzUfquPfa6dwwZ7QdrQwBEk/PiC4tLdWNGzd6HcaQUdPcxp82lfPI2iMcqz/NFdML+X83zaYw0y5gNLDpSB3//MQOdp9oYtaYTO5cXML1c0YzItX2Y6OJiGxS1dKQzMsSiglWR1c3bx9r4K1Dtbz0TiWbjtahCosm5PI3yyaxbFqh1yGaKNPVrfx5czkPv36QfVXNpCQmsHBCLhdOzGXaqEymjcxgVFYaKUnW+u6VmEkoInINcD+QCPxMVb/rN17c8dcBLcBdqro5mGkDsYQSOqpKZWMbu443sqWsng2HatlSVkdrRzcAM0dnctXMkVw3e7RdtGj6papsPFLHi+9U8tqeKvZWNp81PndECoUZqRRmpjEyI5WRmWkUZqZSmJHGyEx3OCPVTvIIg5hIKCKSCOwFrgLKgQ3Arar6jk+Z64C/w0koi4D7VXVRMNMGYgkleO2d3TS2dtDU2kl9SzsV9a2U17VQXneagzXN7DreRO2pdgASBGaNyaK0JIeFJbksKMmhMMOatczgNbV2sLeymQNVzZxobKWysZWqpjbnf2Mb1c1tdPldJJmYIIzKTGN0VhpjsocxJnsYRdlpjM5yXmcPT2ZEahIjUhIt8QxAKBNKOBszFwL7VfUggIg8CiwHfJPCcuDX6mS1dSKSLSKjgZIgpg2ZG/5nNa0d3fQk1zOLsZ71773jAT1TRs8e9svTA5rWbzx+4/uK6d3Pf+/n9Qy0d3XT1tlNIFnDkinJG87VM0cyc0wmM0Y7f+nW7m1CKCMtmQXjc1gwPifg+K5u5eSpNqoa26hqauV4QyvH61upqD9NRcNptpbV89zbx+noCrxDnJqUwPCURBIThASRs/4nJggiEIlTBCJ1IkLu8BQeW7E4Ip/Vl3BuJYqAMp/hcpyjkP7KFAU5LQAicjdwN8C4ceMGFejkgvR3F0w569+ZBeLd4bPHByrz7jwk4DS+y9h7yvjNJPDn9Tf/sxdi/89LThQy0pLIHJZMRloSWcOS3b29YWSkJWOM1xIThMKMNPdIOCtgme5upaa5jYoGJ9E0nO7gVFsnp9q6aGnvpKW9iy5VuruVrm4987pboSsSfccR6p5WlMwoWW/DmVACpWb/Ku6tTDDTOm+qPgQ8BE6T10AC7PGjW84fzGTGGA8lJAiFmWkUZqYxz05TjwrhTCjlgO8DxIuBiiDLpAQxrTHGmCgSzp6rDcAUEZkgIinALcBKvzIrgTvEcSHQoKrHg5zWGGNMFAnbEYqqdorIPcDzOKf+/kJVd4rICnf8g8CzOGd47cc5bfgTfU0brliNMcacO7uw0RhjhrBQnjZsJ2sbY4wJCUsoxhhjQsISijHGmJCwhGKMMSYk4qpTXkSqgVNAjdex9COf6I8RLM5QioUYweIMtViIc5qqhuQOr3F1gyZVLRCRjaE6YyFcYiFGsDhDKRZiBIsz1GIhThEJ2amx1uRljDEmJCyhGGOMCYl4TCgPeR1AEGIhRrA4QykWYgSLM9RiIc6QxRhXnfLGGGO8E49HKMYYYzxgCcUYY0xIxHRCEZH7ROSYiGx1/67rpdw1IrJHRPaLyL0exPkfIrJbRLaLyBMikt1LucMissP9LhG7y2V/9eM+XuC/3fHbRWR+pGJzP3+siLwqIrtEZKeIfCFAmWUi0uCzLHwjkjH6xNHnb+h1XboxTPOpp60i0igiX/Qr40l9isgvRKRKRN72eS9XRF4UkX3u/4DPDY7Uet5LjFG3jvcSZ3i3maoas3/AfcCX+ymTCBwAJuI8uGsbMDPCcV4NJLmvvwd8r5dyh4H8CMfWb/3gPGLgOZwnaV4IvBXhGEcD893XGcDeADEuA/4SybgG8xt6XZe9/P4ngPHRUJ/AUmA+8LbPe98H7nVf3xto/Ynket5LjFG3jvcSZ1i3mTF9hBKkhcB+VT2oqu3Ao8DySAagqi+oaqc7uA7nCZTRIpj6WQ78Wh3rgGwRGR2pAFX1uKpudl83AbuAokh9foh5WpcBXAEcUNUjHsZwhqq+DtT6vb0c+JX7+lfABwJMGrH1PFCM0biO91KXwRh0XcZDQrnHPcz8RS+HwkVAmc9wOd5ujD6Js4caiAIviMgmEbk7QvEEUz9RU4ciUgKcD7wVYPRiEdkmIs+JyKzIRnZGf79h1NSl6xbg972Mi4b6BBipzpNccf8XBigTTfUabeu4v7BtM6P+1isi8hIwKsCofwZ+AnwL50f6FvCfOD/mWbMIMG3Iz5XuK05Vfcot889AJ/DbXmZzsapWiEgh8KKI7Hb3MsIpmPqJSB32R0TSgT8DX1TVRr/Rm3GabZrdduEngSkRDhH6/w2joi4BxHm89vuBfwowOlrqM1hRUa9Ruo77Cus2M+oTiqpeGUw5EXkY+EuAUeXAWJ/hYqAiBKGdpb84ReRO4AbgCnUbKgPMo8L9XyUiT+AceoZ7YQumfiJSh30RkWScZPJbVX3cf7xvglHVZ0XkxyKSr6oRvTFfEL+h53Xp41pgs6pW+o+Ilvp0VYrIaFU97jYPVgUo43m9RvE67vv5Z37rcGwzY7rJy6/t+YPA2wGKbQCmiMgEd4/sFmBlJOLrISLXAF8F3q+qLb2UGSEiGT2vcTr5An2fUAumflYCd7hnKF0INPQ0QUSCiAjwc2CXqv6wlzKj3HKIyEKcZftkpGJ0PzeY39DTuvRzK700d0VDffpYCdzpvr4TeCpAGU/X8yhfx31jCO82MxJnG4TrD3gE2AFsd7/waPf9McCzPuWuwzkz6ABOE1Sk49yP0ya51f170D9OnDMqtrl/OyMZZ6D6AVYAK9zXAjzgjt8BlEa4/pbgHHJv96nD6/xivMett204naIXefA7B/wNo6kufWIdjpMgsnze87w+cRLccaADZ0/5U0Ae8DKwz/2f65b1ZD3vJcaoW8d7iTOs20y79YoxxpiQiOkmL2OMMdHDEooxxpiQsIRijDEmJCyhGGOMCQlLKMYYY0LCEooxxpiQsIRijDEmJP4/7N8ZJoTnwL4AAAAASUVORK5CYII="/>

* Bootstrap results indicate that there is strong evidence for a higher 7-day retention when the gate is at level 30 than when it is at level 40.

* Bottom line, gates should not be moved from level 30 to level 40 to increase retention


-----


#### T-test



```python
df_30 = df[df["version"] == "gate_30"] 
print(df_30.shape)
df_30.tail()
```

<pre>
(44699, 5)
</pre>
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
      <th>userid</th>
      <th>version</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90179</th>
      <td>9998576</td>
      <td>gate_30</td>
      <td>14</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90180</th>
      <td>9998623</td>
      <td>gate_30</td>
      <td>7</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90182</th>
      <td>9999178</td>
      <td>gate_30</td>
      <td>21</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90183</th>
      <td>9999349</td>
      <td>gate_30</td>
      <td>10</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90186</th>
      <td>9999710</td>
      <td>gate_30</td>
      <td>28</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_40 = df[df["version"] == "gate_40"] 
print(df_40.shape)
df_40.tail()
```

<pre>
(45489, 5)
</pre>
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
      <th>userid</th>
      <th>version</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90181</th>
      <td>9998733</td>
      <td>gate_40</td>
      <td>10</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90184</th>
      <td>9999441</td>
      <td>gate_40</td>
      <td>97</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90185</th>
      <td>9999479</td>
      <td>gate_40</td>
      <td>30</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90187</th>
      <td>9999768</td>
      <td>gate_40</td>
      <td>51</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90188</th>
      <td>9999861</td>
      <td>gate_40</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python
from scipy import stats
# Independent Sample T-Test (2 Sample T-Test)

tTestResult = stats.ttest_ind(df_30['retention_1'], df_40['retention_1'])

tTestResultDiffVar = stats.ttest_ind(df_30['retention_1'], df_40['retention_1'], equal_var=False)

tTestResult
```

<pre>
Ttest_indResult(statistic=1.7871153372992439, pvalue=0.07392220630182522)
</pre>

```python
tTestResult = stats.ttest_ind(df_30['retention_7'], df_40['retention_1'])
tTestResultDiffVar = stats.ttest_ind(df_30['retention_7'], df_40['retention_1'], equal_var=False)

tTestResult
```

<pre>
Ttest_indResult(statistic=-84.48321935747556, pvalue=0.0)
</pre>
-----


##### T Score

- A large t-score means that the two groups are different.

- A small t-score means that the two groups are similar.



##### P-values

- The p-value is 0.05 at the 5% level.

- Small p-values are recommended. This means that the data did not happen by chance.

- For example, a p-value of 0.01 means that there is only a 1% chance that the result is by chance.

- In most cases, a p-value of the 0.05 (5%) level is taken as a reference. In this case, it is said to be statistically significant.



[T-test Reference](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/t-test/)


* Looking at the above analysis results, it can be seen that there is no significant difference in retention_1 between the two groups and there is a significant difference in retention_7.

* Again, it is not accidental that gate30 has a higher retention_7 than gate40.

* In other words, the gate at 30 is a better choice for retention 7 dimensions than at 40.


-----


#### chi-square

* In fact, the t-test was analyzed with retention set to 0,1.



* However, retention is actually a categorical variable.

A chi-square test is a better method than this method.



* The chi-square test is also used to test whether a categorical random variable 𝑋 is independent or correlated with another categorical random variable 𝑌.

* When the chi-square test is used to check for independence, it is called the chi-square test of independence.


* If two random variables are independent, then the 𝑌 distribution for 𝑋=0 and the 𝑌 distribution for 𝑋=1 must be the same.

* In other words, the distribution of Y is the same for both versions 30 and 40.

* Therefore, if the chi-square test is adopted with the null hypothesis that the sample sets come from the same probability distribution, the two random variables are independent.

* If rejected, then the two random variables are correlated.

* In other words, if the chi-square test result is rejected, the value of retention will change depending on whether the gate is 30 or 40.



* If each 𝑌 distribution according to the value of 𝑋 is given in the form of a two-dimensional table (contingency table), the difference between the distribution in the case of independence and the actual y sample size is calculated as a test statistic.

* If this value is large enough, 𝑋 and 𝑌 are correlated.



```python
df.groupby('version').sum()
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
      <th>userid</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
    <tr>
      <th>version</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gate_30</th>
      <td>222937707836</td>
      <td>2294941</td>
      <td>20034</td>
      <td>8501</td>
    </tr>
    <tr>
      <th>gate_40</th>
      <td>227857702576</td>
      <td>2333530</td>
      <td>20119</td>
      <td>8279</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.groupby('version').count()
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
      <th>userid</th>
      <th>sum_gamerounds</th>
      <th>retention_1</th>
      <th>retention_7</th>
    </tr>
    <tr>
      <th>version</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gate_30</th>
      <td>44699</td>
      <td>44699</td>
      <td>44699</td>
      <td>44699</td>
    </tr>
    <tr>
      <th>gate_40</th>
      <td>45489</td>
      <td>45489</td>
      <td>45489</td>
      <td>45489</td>
    </tr>
  </tbody>
</table>
</div>


### Create a contingency table for each version.

||retention_1=False|retention_1=True|

|------|---|---|

|version=gate30|(44699-20034)|20034|

| version=gate40|(45489-20119)|20119|







||retention_7=False|retention_7=True|

|------|---|---|

|version=gate30|(44699-8501)|8501|

| version=gate40|(45489-8279)|8279|






```python
import scipy as sp
obs1 = np.array([[20119, (45489-20119)], [20034, (44699-20034)]])
sp.stats.chi2_contingency(obs1)
```

<pre>
(3.1698355431707994,
 0.07500999897705699,
 1,
 array([[20252.35970417, 25236.64029583],
        [19900.64029583, 24798.35970417]]))
</pre>
* The significance probability of the chi-square independent test is 7.5%.

* That is, 𝑋 and 𝑌 cannot be said to be correlated.



```python
obs2 = np.array([[8501, (44699-8501)], [8279, (45489-8279)]])
sp.stats.chi2_contingency(obs2)
```

<pre>
(9.915275528905669,
 0.0016391259678654423,
 1,
 array([[ 8316.50796115, 36382.49203885],
        [ 8463.49203885, 37025.50796115]]))
</pre>
* The significance probability of the chi-square independent test is 0.1%.

* In other words, we can say that 𝑋 and 𝑌 are correlated.

* Retention after 7 days is correlated with whether the gate is at 30 or 40.

* Gate must be kept at 30 to maintain retention after 7 days.


-----


### conclusion

The gate should be kept at 30.



### More to think about

* Actually, there are various metrics to consider other than retention.

* In-app purchases, number of games played, referrer due to friend invitation, etc.

* In this data, only retention is given, so we focused on one thing and analyzed it.

* It is important for service operators and planners to determine really important metrics and evaluate test results based on them.

