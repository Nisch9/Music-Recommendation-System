import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()

data = pd.read_csv('/Users/rishi/BTECH/Programming/My_Projects/Music_Recommendation_System/Dataset/data.csv')
# data.head()
# data.info()
df = data.drop(columns=['id','name','artists','release_date'])
# df=data
df.fillna(0)
# print(df.corr())
from sklearn.preprocessing import MinMaxScaler
datatypes = ['int16','int32','int64','float16','float32','float64']
normalization = data.select_dtypes(include = datatypes)
for col in normalization.columns:
  MinMaxScaler(col)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
features = kmeans.fit_predict(normalization)
data['features'] = features
MinMaxScaler(data['features'])

class Spotify_Recommendation():
  def __init__(self, dataset):
    self.dataset = dataset
  def recommend(self,songs,amount=1):
    distance = []
    song = self.dataset[(self.dataset.name.str.lower() == songs.lower())].head(1).values[0]
    rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]
    # print(song)
    for songs in tqdm(rec.values):
      d=0
      for col in np.arange(len(rec.columns)):
        if not col in [3,8,14,16,1]:
          d=d+np.absolute(float(song[col]) - float(songs[col]))
      distance.append(d)
    rec['distance'] = distance
    rec = rec.sort_values('distance')
    columns = ['artists','name']
    return rec[columns][:amount]