## Data Cleaning and Plotting in Python

#### About Dataset


The data is succesfully scrapped from imdb top netflix movies and tvshows.This dataset need clever programming knowledge for feature extraction also you can build a RECOMMENDATION system either GENRE prediction model




```python
#Import Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
#Open a csv file
df = pd.read_csv('movies.csv')
#Check the info of the csv file
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9999 entries, 0 to 9998
    Data columns (total 9 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   MOVIES    9999 non-null   object 
     1   YEAR      9355 non-null   object 
     2   GENRE     9919 non-null   object 
     3   RATING    8179 non-null   float64
     4   ONE-LINE  9999 non-null   object 
     5   STARS     9999 non-null   object 
     6   VOTES     8179 non-null   object 
     7   RunTime   7041 non-null   float64
     8   Gross     460 non-null    object 
    dtypes: float64(2), object(7)
    memory usage: 703.2+ KB



```python
#Overview of the table
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
      <th>MOVIES</th>
      <th>YEAR</th>
      <th>GENRE</th>
      <th>RATING</th>
      <th>ONE-LINE</th>
      <th>STARS</th>
      <th>VOTES</th>
      <th>RunTime</th>
      <th>Gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>(2021)</td>
      <td>\nAction, Horror, Thriller</td>
      <td>6.1</td>
      <td>\nA woman with a mysterious illness is forced ...</td>
      <td>\n    Director:\nPeter Thorwarth\n| \n    Star...</td>
      <td>21,062</td>
      <td>121.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>(2021– )</td>
      <td>\nAnimation, Action, Adventure</td>
      <td>5.0</td>
      <td>\nThe war for Eternia begins again in what may...</td>
      <td>\n            \n    Stars:\nChris Wood, \nSara...</td>
      <td>17,870</td>
      <td>25.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>(2010–2022)</td>
      <td>\nDrama, Horror, Thriller</td>
      <td>8.2</td>
      <td>\nSheriff Deputy Rick Grimes wakes up from a c...</td>
      <td>\n            \n    Stars:\nAndrew Lincoln, \n...</td>
      <td>885,805</td>
      <td>44.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>(2013– )</td>
      <td>\nAnimation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>\nAn animated series that follows the exploits...</td>
      <td>\n            \n    Stars:\nJustin Roiland, \n...</td>
      <td>414,849</td>
      <td>23.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>(2021)</td>
      <td>\nAction, Crime, Horror</td>
      <td>NaN</td>
      <td>\nA prequel, set before the events of Army of ...</td>
      <td>\n    Director:\nMatthias Schweighöfer\n| \n  ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Make proper column 
df.columns = map(str.title, df.columns)
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
      <th>Movies</th>
      <th>Year</th>
      <th>Genre</th>
      <th>Rating</th>
      <th>One-Line</th>
      <th>Stars</th>
      <th>Votes</th>
      <th>Runtime</th>
      <th>Gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>(2021)</td>
      <td>\nAction, Horror, Thriller</td>
      <td>6.1</td>
      <td>\nA woman with a mysterious illness is forced ...</td>
      <td>\n    Director:\nPeter Thorwarth\n| \n    Star...</td>
      <td>21,062</td>
      <td>121.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>(2021– )</td>
      <td>\nAnimation, Action, Adventure</td>
      <td>5.0</td>
      <td>\nThe war for Eternia begins again in what may...</td>
      <td>\n            \n    Stars:\nChris Wood, \nSara...</td>
      <td>17,870</td>
      <td>25.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>(2010–2022)</td>
      <td>\nDrama, Horror, Thriller</td>
      <td>8.2</td>
      <td>\nSheriff Deputy Rick Grimes wakes up from a c...</td>
      <td>\n            \n    Stars:\nAndrew Lincoln, \n...</td>
      <td>885,805</td>
      <td>44.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>(2013– )</td>
      <td>\nAnimation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>\nAn animated series that follows the exploits...</td>
      <td>\n            \n    Stars:\nJustin Roiland, \n...</td>
      <td>414,849</td>
      <td>23.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>(2021)</td>
      <td>\nAction, Crime, Horror</td>
      <td>NaN</td>
      <td>\nA prequel, set before the events of Army of ...</td>
      <td>\n    Director:\nMatthias Schweighöfer\n| \n  ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Remove duplicates
df = df.drop_duplicates()
df.shape
```




    (9568, 9)




```python
#Remove White Spaces in columns
df.columns.str.strip()
```




    Index(['Movies', 'Year', 'Genre', 'Rating', 'One-Line', 'Stars', 'Votes',
           'Runtime', 'Gross'],
          dtype='object')




```python
#Remove unnessary columns
to_drop = ['One-Line', 'Stars', 'Gross']
df = df.drop(to_drop, axis = 1)
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
      <th>Movies</th>
      <th>Year</th>
      <th>Genre</th>
      <th>Rating</th>
      <th>Votes</th>
      <th>Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>(2021)</td>
      <td>\nAction, Horror, Thriller</td>
      <td>6.1</td>
      <td>21,062</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>(2021– )</td>
      <td>\nAnimation, Action, Adventure</td>
      <td>5.0</td>
      <td>17,870</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>(2010–2022)</td>
      <td>\nDrama, Horror, Thriller</td>
      <td>8.2</td>
      <td>885,805</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>(2013– )</td>
      <td>\nAnimation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>414,849</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>(2021)</td>
      <td>\nAction, Crime, Horror</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Remove special characters 
df['Year'] = df['Year'].str.replace('[^a-zA-Z0-9]', '')
df['Genre'] = df['Genre'].str.replace('\n','')
df['Votes'] = df['Votes'].str.replace('\n', '')
df['Runtime'] = df['Runtime'].replace('\n', '').astype(float)
df['Votes'] = df['Votes'].str.replace(',','').astype(float)
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
      <th>Movies</th>
      <th>Year</th>
      <th>Genre</th>
      <th>Rating</th>
      <th>Votes</th>
      <th>Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>2021</td>
      <td>Action, Horror, Thriller</td>
      <td>6.1</td>
      <td>21062.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>Animation, Action, Adventure</td>
      <td>5.0</td>
      <td>17870.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>20102022</td>
      <td>Drama, Horror, Thriller</td>
      <td>8.2</td>
      <td>885805.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>Animation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>414849.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>Action, Crime, Horror</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>Drama, Thriller</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Arcane</td>
      <td>2021</td>
      <td>Animation, Action, Adventure</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Heart of Invictus</td>
      <td>2022</td>
      <td>Documentary, Sport</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>Adventure, Drama, Fantasy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>Adventure, Drama, Fantasy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 6 columns</p>
</div>




```python
#Organize the Date column to 4 digit year
df['Year'] = df['Year'].str.extract(r'^(\d{4})', expand = False)
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
      <th>Movies</th>
      <th>Year</th>
      <th>Genre</th>
      <th>Rating</th>
      <th>Votes</th>
      <th>Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>2021</td>
      <td>Action, Horror, Thriller</td>
      <td>6.1</td>
      <td>21062.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>Animation, Action, Adventure</td>
      <td>5.0</td>
      <td>17870.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>2010</td>
      <td>Drama, Horror, Thriller</td>
      <td>8.2</td>
      <td>885805.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>Animation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>414849.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>Action, Crime, Horror</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Split the multiple genre into column
df[['Genre_1', 'Genre_2', 'Genre_3']] = df['Genre'].str.split(',', expand = True)
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
      <th>Movies</th>
      <th>Year</th>
      <th>Genre</th>
      <th>Rating</th>
      <th>Votes</th>
      <th>Runtime</th>
      <th>Genre_1</th>
      <th>Genre_2</th>
      <th>Genre_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>2021</td>
      <td>Action, Horror, Thriller</td>
      <td>6.1</td>
      <td>21062.0</td>
      <td>121.0</td>
      <td>Action</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>Animation, Action, Adventure</td>
      <td>5.0</td>
      <td>17870.0</td>
      <td>25.0</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>2010</td>
      <td>Drama, Horror, Thriller</td>
      <td>8.2</td>
      <td>885805.0</td>
      <td>44.0</td>
      <td>Drama</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>Animation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>414849.0</td>
      <td>23.0</td>
      <td>Animation</td>
      <td>Adventure</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>Action, Crime, Horror</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Action</td>
      <td>Crime</td>
      <td>Horror</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>Drama, Thriller</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Drama</td>
      <td>Thriller</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Arcane</td>
      <td>2021</td>
      <td>Animation, Action, Adventure</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Heart of Invictus</td>
      <td>2022</td>
      <td>Documentary, Sport</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Documentary</td>
      <td>Sport</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>Adventure, Drama, Fantasy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>Adventure, Drama, Fantasy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 9 columns</p>
</div>




```python
#Dropping the original genre column
df = df.drop(['Genre'], axis = 1)
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
      <th>Movies</th>
      <th>Year</th>
      <th>Rating</th>
      <th>Votes</th>
      <th>Runtime</th>
      <th>Genre_1</th>
      <th>Genre_2</th>
      <th>Genre_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>2021</td>
      <td>6.1</td>
      <td>21062.0</td>
      <td>121.0</td>
      <td>Action</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>5.0</td>
      <td>17870.0</td>
      <td>25.0</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>2010</td>
      <td>8.2</td>
      <td>885805.0</td>
      <td>44.0</td>
      <td>Drama</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>9.2</td>
      <td>414849.0</td>
      <td>23.0</td>
      <td>Animation</td>
      <td>Adventure</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Action</td>
      <td>Crime</td>
      <td>Horror</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Drama</td>
      <td>Thriller</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Arcane</td>
      <td>2021</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Heart of Invictus</td>
      <td>2022</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Documentary</td>
      <td>Sport</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 8 columns</p>
</div>




```python
#Remove NaN values
rating_mean = df['Rating'].mean()
df['Rating'] = df['Rating'].fillna(rating_mean)
df['Rating'] = df['Rating'].round(decimals=1)
df[['Year', 'Votes', 'Runtime','Genre_1']] = df[['Year', 'Votes', 'Runtime', 'Genre_1']].fillna('')
df['Runtime']= pd.to_numeric(df['Runtime'])
df['Votes'] = pd.to_numeric(df['Votes'])
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
      <th>Movies</th>
      <th>Year</th>
      <th>Rating</th>
      <th>Votes</th>
      <th>Runtime</th>
      <th>Genre_1</th>
      <th>Genre_2</th>
      <th>Genre_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>2021</td>
      <td>6.1</td>
      <td>21062.0</td>
      <td>121.0</td>
      <td>Action</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>5.0</td>
      <td>17870.0</td>
      <td>25.0</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>2010</td>
      <td>8.2</td>
      <td>885805.0</td>
      <td>44.0</td>
      <td>Drama</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>9.2</td>
      <td>414849.0</td>
      <td>23.0</td>
      <td>Animation</td>
      <td>Adventure</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Action</td>
      <td>Crime</td>
      <td>Horror</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Drama</td>
      <td>Thriller</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Arcane</td>
      <td>2021</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Heart of Invictus</td>
      <td>2022</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Documentary</td>
      <td>Sport</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 8 columns</p>
</div>




```python
#Filtering year from 2019 to 2023 for the sample plot
year_filter = df['Year'].apply(lambda x: x >= '2019')
df = df[year_filter].reset_index(drop = True)
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
      <th>Movies</th>
      <th>Year</th>
      <th>Rating</th>
      <th>Votes</th>
      <th>Runtime</th>
      <th>Genre_1</th>
      <th>Genre_2</th>
      <th>Genre_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>2021</td>
      <td>6.1</td>
      <td>21062.0</td>
      <td>121.0</td>
      <td>Action</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>5.0</td>
      <td>17870.0</td>
      <td>25.0</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Action</td>
      <td>Crime</td>
      <td>Horror</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Outer Banks</td>
      <td>2020</td>
      <td>7.6</td>
      <td>25858.0</td>
      <td>50.0</td>
      <td>Action</td>
      <td>Crime</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Last Letter from Your Lover</td>
      <td>2021</td>
      <td>6.8</td>
      <td>5283.0</td>
      <td>110.0</td>
      <td>Drama</td>
      <td>Romance</td>
      <td>None</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4022</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Drama</td>
      <td>Thriller</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4023</th>
      <td>Arcane</td>
      <td>2021</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>4024</th>
      <td>Heart of Invictus</td>
      <td>2022</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Documentary</td>
      <td>Sport</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4025</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>4026</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
<p>4027 rows × 8 columns</p>
</div>




```python
#Plotting Number of movies from year 2019-2023 by Genre

genre_counts = df['Genre_1'].explode().str.strip().value_counts()
genre_unique = df['Genre_1'].explode().str.strip().unique()
plt.figure(figsize=(10,6))
sns.barplot(x=genre_counts, y=genre_unique,  orient='h', palette='viridis')
plt.ylabel('Genre')
plt.xlabel('Number of Movies')
plt.title('Number of Movies by Genre (2019-2023)')
plt.show()
```


    
![png](README_files/README_15_0.png)
    



```python
#Plotting the number of movies by year
year_counts = df['Year'].explode().str.strip().value_counts()
year_unique = sorted(df['Year'].explode().str.strip().unique())
plt.figure(figsize=(10,6))
sns.barplot(year_unique, year_counts, palette='viridis')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Number of Movies by Year')
plt.show()

```


    
![png](README_files/README_16_0.png)
    



```python
#Making pie chart for the top 7 genre
fig, ax = plt.subplots()
ax.pie(genre_counts[:7], labels = genre_counts[:7].index, autopct='%1.0f%%', startangle = 90)
ax.axis('equal')
ax.set_title('Top 7 Movie Genre')
plt.show()
```


    
![png](README_files/README_17_0.png)
    



```python
#Heatmap
df_rating = df[['Rating', 'Runtime', 'Votes']]
cor_movie = df_rating.corr()
plt.figure(figsize=(10,6))
sns.heatmap(cor_movie, annot=True)
plt.title('Heatmap')
plt.show()

```


    
![png](README_files/README_18_0.png)
    

