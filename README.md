## Data Cleaning and Plotting in Python

#### About Dataset


The data is succesfully scrapped from imdb top netflix movies and tvshows.This dataset need clever programming knowledge for feature extraction also you can build a RECOMMENDATION system either GENRE prediction model



#### Importing Packages


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

#### Loading the dataset


```python
df = pd.read_csv('movies.csv')
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



#### Removing Duplicates


```python
df = df.drop_duplicates()
df.shape
```




    (9568, 9)




```python
df.columns.str.strip()
```




    Index(['Movies', 'Year', 'Genre', 'Rating', 'One-Line', 'Stars', 'Votes',
           'Runtime', 'Gross'],
          dtype='object')



#### Dropping unnecessary column


```python
to_drop = ['One-Line', 'Gross']
df = df.drop(to_drop, axis = 1)
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
      <th>Stars</th>
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
      <td>\n    Director:\nPeter Thorwarth\n| \n    Star...</td>
      <td>21,062</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>(2021– )</td>
      <td>\nAnimation, Action, Adventure</td>
      <td>5.0</td>
      <td>\n            \n    Stars:\nChris Wood, \nSara...</td>
      <td>17,870</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>(2010–2022)</td>
      <td>\nDrama, Horror, Thriller</td>
      <td>8.2</td>
      <td>\n            \n    Stars:\nAndrew Lincoln, \n...</td>
      <td>885,805</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>(2013– )</td>
      <td>\nAnimation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>\n            \n    Stars:\nJustin Roiland, \n...</td>
      <td>414,849</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>(2021)</td>
      <td>\nAction, Crime, Horror</td>
      <td>NaN</td>
      <td>\n    Director:\nMatthias Schweighöfer\n| \n  ...</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>Totenfrau</td>
      <td>(2022– )</td>
      <td>\nDrama, Thriller</td>
      <td>NaN</td>
      <td>\n    Director:\nNicolai Rohde\n| \n    Stars:...</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Arcane</td>
      <td>(2021– )</td>
      <td>\nAnimation, Action, Adventure</td>
      <td>NaN</td>
      <td>\n</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Heart of Invictus</td>
      <td>(2022– )</td>
      <td>\nDocumentary, Sport</td>
      <td>NaN</td>
      <td>\n    Director:\nOrlando von Einsiedel\n| \n  ...</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>The Imperfects</td>
      <td>(2021– )</td>
      <td>\nAdventure, Drama, Fantasy</td>
      <td>NaN</td>
      <td>\n    Director:\nJovanka Vuckovic\n| \n    Sta...</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>The Imperfects</td>
      <td>(2021– )</td>
      <td>\nAdventure, Drama, Fantasy</td>
      <td>NaN</td>
      <td>\n    Director:\nJovanka Vuckovic\n| \n    Sta...</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 7 columns</p>
</div>



#### Separating Directors and Stars in the Stars Column


```python
df[['Directors', 'Star']] = df['Stars'].str.split('Stars:', expand=True)
df.drop('Stars', inplace=True, axis=1)
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
      <th>Directors</th>
      <th>Star</th>
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
      <td>\n    Director:\nPeter Thorwarth\n| \n</td>
      <td>\nPeri Baumeister, \nCarl Anton Koch, \nAlexan...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>(2021– )</td>
      <td>\nAnimation, Action, Adventure</td>
      <td>5.0</td>
      <td>17,870</td>
      <td>25.0</td>
      <td>\n            \n</td>
      <td>\nChris Wood, \nSarah Michelle Gellar, \nLena ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>(2010–2022)</td>
      <td>\nDrama, Horror, Thriller</td>
      <td>8.2</td>
      <td>885,805</td>
      <td>44.0</td>
      <td>\n            \n</td>
      <td>\nAndrew Lincoln, \nNorman Reedus, \nMelissa M...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>(2013– )</td>
      <td>\nAnimation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>414,849</td>
      <td>23.0</td>
      <td>\n            \n</td>
      <td>\nJustin Roiland, \nChris Parnell, \nSpencer G...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>(2021)</td>
      <td>\nAction, Crime, Horror</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n    Director:\nMatthias Schweighöfer\n| \n</td>
      <td>\nMatthias Schweighöfer, \nNathalie Emmanuel, ...</td>
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
      <td>(2022– )</td>
      <td>\nDrama, Thriller</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n    Director:\nNicolai Rohde\n| \n</td>
      <td>\nFelix Klare, \nRomina Küper, \nAnna Maria Mü...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Arcane</td>
      <td>(2021– )</td>
      <td>\nAnimation, Action, Adventure</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Heart of Invictus</td>
      <td>(2022– )</td>
      <td>\nDocumentary, Sport</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n    Director:\nOrlando von Einsiedel\n| \n  ...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>The Imperfects</td>
      <td>(2021– )</td>
      <td>\nAdventure, Drama, Fantasy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n    Director:\nJovanka Vuckovic\n| \n</td>
      <td>\nMorgan Taylor Campbell, \nIñaki Godoy, \nRhi...</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>The Imperfects</td>
      <td>(2021– )</td>
      <td>\nAdventure, Drama, Fantasy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n    Director:\nJovanka Vuckovic\n| \n</td>
      <td>\nMorgan Taylor Campbell, \nJennifer Cheon Gar...</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 8 columns</p>
</div>



#### Remove special characters


```python
df = df.replace('\n', '', regex=True)
df['Year'] = df['Year'].str.replace('[^0-9]', '')
df['Directors'] = df['Directors'].str.replace('|', '')
df['Directors'] = df['Directors'].str.replace('Director:', '')
df['Votes'] = df['Votes'].str.replace('[^0-9]', '')

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
      <th>Directors</th>
      <th>Star</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>2021</td>
      <td>Action, Horror, Thriller</td>
      <td>6.1</td>
      <td>21062</td>
      <td>121.0</td>
      <td>Peter Thorwarth</td>
      <td>Peri Baumeister, Carl Anton Koch, Alexander Sc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>Animation, Action, Adventure</td>
      <td>5.0</td>
      <td>17870</td>
      <td>25.0</td>
      <td></td>
      <td>Chris Wood, Sarah Michelle Gellar, Lena Headey...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>20102022</td>
      <td>Drama, Horror, Thriller</td>
      <td>8.2</td>
      <td>885805</td>
      <td>44.0</td>
      <td></td>
      <td>Andrew Lincoln, Norman Reedus, Melissa McBride...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>Animation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>414849</td>
      <td>23.0</td>
      <td></td>
      <td>Justin Roiland, Chris Parnell, Spencer Grammer...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>Action, Crime, Horror</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Matthias Schweighöfer</td>
      <td>Matthias Schweighöfer, Nathalie Emmanuel, Ruby...</td>
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
      <td>Drama, Thriller</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Nicolai Rohde</td>
      <td>Felix Klare, Romina Küper, Anna Maria Mühe, Ro...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Arcane</td>
      <td>2021</td>
      <td>Animation, Action, Adventure</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>None</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Heart of Invictus</td>
      <td>2022</td>
      <td>Documentary, Sport</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Orlando von Einsiedel     Star:Prince Harry</td>
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
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Iñaki Godoy, Rhianna J...</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>Adventure, Drama, Fantasy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Jennifer Cheon Garcia,...</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 8 columns</p>
</div>



#### Formatting Year colum to 4 digit


```python
#Formatting the Year column to 4 digit 
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
      <th>Directors</th>
      <th>Star</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blood Red Sky</td>
      <td>2021</td>
      <td>Action, Horror, Thriller</td>
      <td>6.1</td>
      <td>21062</td>
      <td>121.0</td>
      <td>Peter Thorwarth</td>
      <td>Peri Baumeister, Carl Anton Koch, Alexander Sc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>Animation, Action, Adventure</td>
      <td>5.0</td>
      <td>17870</td>
      <td>25.0</td>
      <td></td>
      <td>Chris Wood, Sarah Michelle Gellar, Lena Headey...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>2010</td>
      <td>Drama, Horror, Thriller</td>
      <td>8.2</td>
      <td>885805</td>
      <td>44.0</td>
      <td></td>
      <td>Andrew Lincoln, Norman Reedus, Melissa McBride...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>Animation, Adventure, Comedy</td>
      <td>9.2</td>
      <td>414849</td>
      <td>23.0</td>
      <td></td>
      <td>Justin Roiland, Chris Parnell, Spencer Grammer...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>Action, Crime, Horror</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Matthias Schweighöfer</td>
      <td>Matthias Schweighöfer, Nathalie Emmanuel, Ruby...</td>
    </tr>
  </tbody>
</table>
</div>



#### Split the multiple genre into column


```python
df[['Genre_1', 'Genre_2', 'Genre_3']] = df['Genre'].str.split(',', expand = True)
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
      <th>Directors</th>
      <th>Star</th>
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
      <td>21062</td>
      <td>121.0</td>
      <td>Peter Thorwarth</td>
      <td>Peri Baumeister, Carl Anton Koch, Alexander Sc...</td>
      <td>Action</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>5.0</td>
      <td>17870</td>
      <td>25.0</td>
      <td></td>
      <td>Chris Wood, Sarah Michelle Gellar, Lena Headey...</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>2010</td>
      <td>8.2</td>
      <td>885805</td>
      <td>44.0</td>
      <td></td>
      <td>Andrew Lincoln, Norman Reedus, Melissa McBride...</td>
      <td>Drama</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>9.2</td>
      <td>414849</td>
      <td>23.0</td>
      <td></td>
      <td>Justin Roiland, Chris Parnell, Spencer Grammer...</td>
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
      <td>Matthias Schweighöfer</td>
      <td>Matthias Schweighöfer, Nathalie Emmanuel, Ruby...</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Nicolai Rohde</td>
      <td>Felix Klare, Romina Küper, Anna Maria Mühe, Ro...</td>
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
      <td></td>
      <td>None</td>
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
      <td>Orlando von Einsiedel     Star:Prince Harry</td>
      <td>None</td>
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
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Iñaki Godoy, Rhianna J...</td>
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
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Jennifer Cheon Garcia,...</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 10 columns</p>
</div>



#### Fill the Rating NaN values with average value


```python
rating_mean = df['Rating'].mean()
df['Rating'] = df['Rating'].fillna(rating_mean)
df['Rating'] = df['Rating'].round(decimals=1)

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
      <th>Directors</th>
      <th>Star</th>
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
      <td>21062</td>
      <td>121.0</td>
      <td>Peter Thorwarth</td>
      <td>Peri Baumeister, Carl Anton Koch, Alexander Sc...</td>
      <td>Action</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>5.0</td>
      <td>17870</td>
      <td>25.0</td>
      <td></td>
      <td>Chris Wood, Sarah Michelle Gellar, Lena Headey...</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>2010</td>
      <td>8.2</td>
      <td>885805</td>
      <td>44.0</td>
      <td></td>
      <td>Andrew Lincoln, Norman Reedus, Melissa McBride...</td>
      <td>Drama</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>9.2</td>
      <td>414849</td>
      <td>23.0</td>
      <td></td>
      <td>Justin Roiland, Chris Parnell, Spencer Grammer...</td>
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
      <td>Matthias Schweighöfer</td>
      <td>Matthias Schweighöfer, Nathalie Emmanuel, Ruby...</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>6.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Nicolai Rohde</td>
      <td>Felix Klare, Romina Küper, Anna Maria Mühe, Ro...</td>
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
      <td></td>
      <td>None</td>
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
      <td>Orlando von Einsiedel     Star:Prince Harry</td>
      <td>None</td>
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
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Iñaki Godoy, Rhianna J...</td>
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
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Jennifer Cheon Garcia,...</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 10 columns</p>
</div>



#### Removing NaN values


```python
df[['Year', 'Genre_1']] = df[['Year', 'Genre_1']].fillna('')
df[['Runtime', 'Votes']] = df[['Runtime', 'Votes']].fillna(0)
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
      <th>Directors</th>
      <th>Star</th>
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
      <td>21062</td>
      <td>121.0</td>
      <td>Peter Thorwarth</td>
      <td>Peri Baumeister, Carl Anton Koch, Alexander Sc...</td>
      <td>Action</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>5.0</td>
      <td>17870</td>
      <td>25.0</td>
      <td></td>
      <td>Chris Wood, Sarah Michelle Gellar, Lena Headey...</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>2010</td>
      <td>8.2</td>
      <td>885805</td>
      <td>44.0</td>
      <td></td>
      <td>Andrew Lincoln, Norman Reedus, Melissa McBride...</td>
      <td>Drama</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>9.2</td>
      <td>414849</td>
      <td>23.0</td>
      <td></td>
      <td>Justin Roiland, Chris Parnell, Spencer Grammer...</td>
      <td>Animation</td>
      <td>Adventure</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Matthias Schweighöfer</td>
      <td>Matthias Schweighöfer, Nathalie Emmanuel, Ruby...</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Nicolai Rohde</td>
      <td>Felix Klare, Romina Küper, Anna Maria Mühe, Ro...</td>
      <td>Drama</td>
      <td>Thriller</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Arcane</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td></td>
      <td>None</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Heart of Invictus</td>
      <td>2022</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Orlando von Einsiedel     Star:Prince Harry</td>
      <td>None</td>
      <td>Documentary</td>
      <td>Sport</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Iñaki Godoy, Rhianna J...</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Jennifer Cheon Garcia,...</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 10 columns</p>
</div>




```python
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
      <th>Directors</th>
      <th>Star</th>
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
      <td>21062</td>
      <td>121.0</td>
      <td>Peter Thorwarth</td>
      <td>Peri Baumeister, Carl Anton Koch, Alexander Sc...</td>
      <td>Action</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>5.0</td>
      <td>17870</td>
      <td>25.0</td>
      <td></td>
      <td>Chris Wood, Sarah Michelle Gellar, Lena Headey...</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Walking Dead</td>
      <td>2010</td>
      <td>8.2</td>
      <td>885805</td>
      <td>44.0</td>
      <td></td>
      <td>Andrew Lincoln, Norman Reedus, Melissa McBride...</td>
      <td>Drama</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rick and Morty</td>
      <td>2013</td>
      <td>9.2</td>
      <td>414849</td>
      <td>23.0</td>
      <td></td>
      <td>Justin Roiland, Chris Parnell, Spencer Grammer...</td>
      <td>Animation</td>
      <td>Adventure</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Matthias Schweighöfer</td>
      <td>Matthias Schweighöfer, Nathalie Emmanuel, Ruby...</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Nicolai Rohde</td>
      <td>Felix Klare, Romina Küper, Anna Maria Mühe, Ro...</td>
      <td>Drama</td>
      <td>Thriller</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Arcane</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td></td>
      <td>None</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Heart of Invictus</td>
      <td>2022</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Orlando von Einsiedel     Star:Prince Harry</td>
      <td>None</td>
      <td>Documentary</td>
      <td>Sport</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Iñaki Godoy, Rhianna J...</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Jennifer Cheon Garcia,...</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
<p>9568 rows × 10 columns</p>
</div>




```python
#### Filtering Year from 2019 to 2023 for sample plot
```


```python
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
      <th>Directors</th>
      <th>Star</th>
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
      <td>21062</td>
      <td>121.0</td>
      <td>Peter Thorwarth</td>
      <td>Peri Baumeister, Carl Anton Koch, Alexander Sc...</td>
      <td>Action</td>
      <td>Horror</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Masters of the Universe: Revelation</td>
      <td>2021</td>
      <td>5.0</td>
      <td>17870</td>
      <td>25.0</td>
      <td></td>
      <td>Chris Wood, Sarah Michelle Gellar, Lena Headey...</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Army of Thieves</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Matthias Schweighöfer</td>
      <td>Matthias Schweighöfer, Nathalie Emmanuel, Ruby...</td>
      <td>Action</td>
      <td>Crime</td>
      <td>Horror</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Outer Banks</td>
      <td>2020</td>
      <td>7.6</td>
      <td>25858</td>
      <td>50.0</td>
      <td></td>
      <td>Chase Stokes, Madelyn Cline, Madison Bailey, J...</td>
      <td>Action</td>
      <td>Crime</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Last Letter from Your Lover</td>
      <td>2021</td>
      <td>6.8</td>
      <td>5283</td>
      <td>110.0</td>
      <td>Augustine Frizzell</td>
      <td>Shailene Woodley, Joe Alwyn, Wendy Nottingham,...</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4102</th>
      <td>Totenfrau</td>
      <td>2022</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Nicolai Rohde</td>
      <td>Felix Klare, Romina Küper, Anna Maria Mühe, Ro...</td>
      <td>Drama</td>
      <td>Thriller</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4103</th>
      <td>Arcane</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td></td>
      <td>None</td>
      <td>Animation</td>
      <td>Action</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>4104</th>
      <td>Heart of Invictus</td>
      <td>2022</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Orlando von Einsiedel     Star:Prince Harry</td>
      <td>None</td>
      <td>Documentary</td>
      <td>Sport</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4105</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Iñaki Godoy, Rhianna J...</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>4106</th>
      <td>The Imperfects</td>
      <td>2021</td>
      <td>6.9</td>
      <td>0</td>
      <td>0.0</td>
      <td>Jovanka Vuckovic</td>
      <td>Morgan Taylor Campbell, Jennifer Cheon Garcia,...</td>
      <td>Adventure</td>
      <td>Drama</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
<p>4107 rows × 10 columns</p>
</div>




```python
#### Sample Plot
```


```python
#Horizontal Bar
genre_counts = df['Genre_1'].explode().str.strip().value_counts()
genre_unique = df['Genre_1'].explode().str.strip().unique()
plt.figure(figsize=(10,6))
sns.barplot(x=genre_counts, y=genre_unique,  orient='h', palette='viridis')
plt.ylabel('Genre')
plt.xlabel('Number of Movies')
plt.title('Number of Movies by Genre (2019-2023)')
plt.show()
```


    
![png](README_files/README_29_0.png)
    



```python
#Bar Graph
year_counts = df['Year'].explode().str.strip().value_counts()
year_unique = sorted(df['Year'].explode().str.strip().unique())
plt.figure(figsize=(10,6))
sns.barplot(year_unique, year_counts, palette='viridis')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Number of Movies by Year')
plt.show()

```


    
![png](README_files/README_30_0.png)
    



```python
#Pie Chary
fig, ax = plt.subplots()
ax.pie(genre_counts[:7], labels = genre_counts[:7].index, autopct='%1.0f%%', startangle = 90)
ax.axis('equal')
ax.set_title('Top 7 Movie Genre')
plt.show()
```


    
![png](README_files/README_31_0.png)
    



```python
#Heatmap
df_rating = df[['Rating', 'Runtime', 'Votes']]
cor_movie = df_rating.corr()
plt.figure(figsize=(10,6))
sns.heatmap(cor_movie, annot=True)
plt.title('Heatmap')
plt.show()

```


    
![png](README_files/README_32_0.png)
    

