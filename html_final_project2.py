from google.colab import drive
drive.mount('/content/gdrive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import csv

#train data organized
df = pd.read_csv('gdrive/My Drive/train_raw.csv')
df = df.dropna(thresh = 24) #remove some data with too many null value

df['Album_type'] = df['Album_type'].replace(['single','album', 'compilation',None, np.nan], [0, 1, 2, -1, -1])
df['Licensed'] = df['Licensed'].replace([True, False, None, np.nan], [1, -1, 0, 0])
df['official_video'] = df['official_video'].replace([True, False, None, np.nan], [1, -1, 0, 0])

#test data organized
# dt = pd.read_csv('test.csv')
dt = pd.read_csv('gdrive/My Drive/test.csv')

dt['Album_type'] = dt['Album_type'].replace(['single','album', 'compilation',None, np.nan], [0, 1, 2, -1, -1])
dt['Licensed'] = dt['Licensed'].replace([True, False, None, np.nan], [1, -1, 0, 0])
dt['official_video'] = dt['official_video'].replace([True, False, None, np.nan], [1, -1, 0, 0])

dt.info()
df["Danceability"].value_counts()

#artist
artist_dance_sum = Counter()
artist_appear_sum = Counter()
temp = df[["Artist", "Danceability"]]
temp = temp.dropna()
for l in temp.to_numpy():
  artist_list = [x.strip() for x in re.split(',', l[0])]
  for artist in artist_list:
    artist_dance_sum[artist] += float(l[1])
    artist_appear_sum[artist] += float(1)

top_artist = artist_dance_sum.most_common()
print(len(top_artist))

avg_artist = float(sum(artist_dance_sum.values())/sum(artist_appear_sum.values()))
xs = [a[0] for a in top_artist]
ys = [float(a[1]/artist_appear_sum[a[0]]) for a in top_artist]
ys, xs = zip(*sorted(zip(ys, xs), reverse = True))
z_artist = dict(zip(xs, ys))

temp = df[['Artist']]
tt = temp.to_numpy()
score = []
for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_artist)
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    num += z_artist.get(x, avg_artist)
    cnt += 1
  score.append(float(num/cnt))
df['Artist'] = df['Artist'].replace(tt, score)

temp = dt[['Artist']]
tt = temp.to_numpy()
score = []
c = float(0)
total = float(0)

for i in tt:
  if(i[0] != i[0]): #nan
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    total += z_artist.get(x, avg_artist)
    c += 1
avg_artist = float(total/c)


for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_artist)
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    num += z_artist.get(x, avg_artist)
    cnt += 1
  score.append(float(num/cnt))
dt['Artist'] = dt['Artist'].replace(tt, score)

#composer
composer_dance_sum = Counter()
composer_appear_sum = Counter()
temp = df[["Composer", "Danceability"]]
temp = temp.dropna()
for l in temp.to_numpy():
  composer_list = [x.strip() for x in re.split(',', l[0])]
  for composer in composer_list:
    composer_dance_sum[composer] += float(l[1])
    composer_appear_sum[composer] += float(1)

top_composer = composer_dance_sum.most_common()
print(len(top_composer))

avg_com = float(sum(artist_dance_sum.values())/sum(artist_appear_sum.values()))
xs = [a[0] for a in top_composer]
ys = [float(a[1]/composer_appear_sum[a[0]]) for a in top_composer]
ys, xs = zip(*sorted(zip(ys, xs), reverse = True))
z_com = dict(zip(xs, ys))

temp = df[['Composer']]
tt = temp.to_numpy()
score = []
for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]):
    score.append(avg_com)
    continue
  for x in re.split(',', i[0]):
    num += z_com.get(x, avg_com)
    cnt += 1
  score.append(float(num/cnt))
df['Composer'] = df['Composer'].replace(tt, score)

temp = dt[['Composer']]
tt = temp.to_numpy()
score = []

for i in tt:
  if(i[0] != i[0]): #nan
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    total += z_com.get(x, avg_com)
    c += 1
avg_com = float(total/c)

for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]):
    score.append(avg_com)
    continue
  for x in re.split(',', i[0]):
    num += z_com.get(x, avg_com)
    cnt += 1
  score.append(float(num/cnt))
dt['Composer'] = dt['Composer'].replace(tt, score)

#channel
channel_dance_sum = Counter()
channel_appear_sum = Counter()
temp = df[["Channel", "Danceability"]]
temp = temp.dropna()
for l in temp.to_numpy():
  channel_list = [x.strip() for x in re.split(',', l[0])]
  for channel in channel_list:
    channel_dance_sum[channel] += float(l[1])
    channel_appear_sum[channel] += 1
top_channel = channel_dance_sum.most_common()
print(len(top_channel))

avg_ch = float(sum(channel_dance_sum.values())/sum(channel_appear_sum.values()))
xs = [a[0] for a in top_channel]
ys = [float(a[1]/channel_appear_sum[a[0]]) for a in top_channel]
ys, xs = zip(*sorted(zip(ys, xs), reverse = True))
z_ch = dict(zip(xs, ys))

temp = df[['Channel']]
tt = temp.to_numpy()
score = []
for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_ch)
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    num += z_ch.get(x, avg_ch)
    cnt += 1
  score.append(float(num/cnt))
df['Channel'] = df['Channel'].replace(tt, score)

temp = dt[['Channel']]
tt = temp.to_numpy()
score = []

for i in tt:
  if(i[0] != i[0]): #nan
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    total += z_ch.get(x, avg_ch)
    c += 1
avg_ch = float(total/c)

for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_ch)
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    num += z_ch.get(x, avg_ch)
    cnt += 1
  score.append(float(num/cnt))
dt['Channel'] = dt['Channel'].replace(tt, score)

#title
title_dance_sum = Counter()
title_appear_sum = Counter()
temp = df[["Title", "Danceability"]]
temp = temp.dropna()
for l in temp.to_numpy():
  title_list = [x for x in re.split('\W+', l[0])] #only take singer name
  for title in title_list:
    title_dance_sum[title] += float(l[1])
    title_appear_sum[title] += 1
top_title = title_dance_sum.most_common()
print(len(top_title))

avg_title = float(sum(title_dance_sum.values())/sum(title_appear_sum.values()))
xs = [a[0] for a in top_title]
ys = [float(a[1]/title_appear_sum[a[0]]) for a in top_title]
ys, xs = zip(*sorted(zip(ys, xs), reverse = True))
z_title = dict(zip(xs, ys))

temp = df[['Title']]
tt = temp.to_numpy()
score = []
for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_title)
    continue
  for x in re.split('\W+', i[0]):
    #print(x)
    #x = x.strip()
    num += z_title.get(x, avg_title)
    cnt += 1
  score.append(float(num/cnt))
df['Title'] = df['Title'].replace(tt, score)

temp = dt[['Title']]
tt = temp.to_numpy()
score = []

for i in tt:
  if(i[0] != i[0]): #nan
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    total += z_title.get(x, avg_title)
    c += 1
avg_title = float(total/c)

for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_title)
    continue
  for x in re.split('\W+', i[0]):
    #print(x)
    #x = x.strip()
    num += z_title.get(x, avg_title)
    cnt += 1
  score.append(float(num/cnt))
dt['Title'] = dt['Title'].replace(tt, score)

#track
track_dance_sum = Counter()
track_appear_sum = Counter()
temp = df[["Track", "Danceability"]]
temp = temp.dropna()
for l in temp.to_numpy():
  track_list = [x.strip() for x in re.split('\W+',l[0])]
  for track in track_list:
    track_dance_sum[track] += float(l[1])
    track_appear_sum[track] += 1
top_track = track_dance_sum.most_common()
print(len(top_track))

avg_tr = float(sum(track_dance_sum.values())/sum(track_appear_sum.values()))
xs = [a[0] for a in top_track]
ys = [float(a[1]/track_appear_sum[a[0]]) for a in top_track]
ys, xs = zip(*sorted(zip(ys, xs), reverse = True))
z_tr = dict(zip(xs, ys))

temp = df[['Track']]
tt = temp.to_numpy()
score = []
for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_tr)
    continue
  for x in re.split('\W+',i[0]):
    x = x.strip()
    num += z_tr.get(x, avg_tr)
    cnt += 1
  score.append(float(num/cnt))
df['Track'] = df['Track'].replace(tt, score)

temp = dt[['Track']]
tt = temp.to_numpy()
score = []

for i in tt:
  if(i[0] != i[0]): #nan
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    total += z_tr.get(x, avg_tr)
    c += 1
avg_tr = float(total/c)

for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_tr)
    continue
  for x in re.split('\W+',i[0]):
    x = x.strip()
    num += z_tr.get(x, avg_tr)
    cnt += 1
  score.append(float(num/cnt))
dt['Track'] = dt['Track'].replace(tt, score)

#album
album_dance_sum = Counter()
album_appear_sum = Counter()
temp = df[["Album", "Danceability"]]
temp = temp.dropna()
for l in temp.to_numpy():
  album_list = [x.strip() for x in re.split('\W+',l[0])]
  for album in album_list:
    album_dance_sum[album] += float(l[1])
    album_appear_sum[album] += 1

top_album = album_dance_sum.most_common()
print(len(top_album))

avg_album = float(sum(album_dance_sum.values())/sum(album_appear_sum.values()))
xs = [a[0] for a in top_album]
ys = [float(a[1]/album_appear_sum[a[0]]) for a in top_album]
ys, xs = zip(*sorted(zip(ys, xs), reverse = True))
z_album = dict(zip(xs, ys))

temp = df[['Album']]
tt = temp.to_numpy()
score = []
for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_album)
    continue
  for x in re.split('\W+',l[0]):
    x = x.strip()
    num += z_album.get(x, avg_album)
    cnt += 1
  score.append(float(num/cnt))
df['Album'] = df['Album'].replace(tt, score)

temp = dt[['Album']]
tt = temp.to_numpy()
score = []

for i in tt:
  if(i[0] != i[0]): #nan
    continue
  for x in re.split(',', i[0]):
    x = x.strip()
    total += z_album.get(x, avg_album)
    c += 1
avg_album = float(total/c)

for i in tt:
  cnt = float(0)
  num = float(0)
  if(i[0] != i[0]): #nan
    score.append(avg_album)
    continue
  for x in re.split('\W+',l[0]):
    x = x.strip()
    num += z_album.get(x, avg_album)
    cnt += 1
  score.append(float(num/cnt))
dt['Album'] = dt['Album'].replace(tt, score)

df = df.drop(['Uri', 'Url_spotify', 'Url_youtube', 'Description'], axis = 1)
plt.figure(figsize=(20, 10))
sns.set(style="whitegrid")
corr = df.corr()
sns.heatmap(corr,annot=True, cmap="YlGnBu")


dt = dt.drop(['Uri', 'Url_spotify', 'Url_youtube', 'Description'], axis = 1)
plt.figure(figsize=(20, 10))
sns.set(style="whitegrid")
corr = dt.corr()
sns.heatmap(corr,annot=True, cmap="YlGnBu")

#train fillna
num = df['Energy'].count()
total = df['Energy'].sum()
df['Energy'] = df['Energy'].replace([np.nan, None], float(total/num))

num = df['Key'].count()
total = df['Key'].sum()
df['Key'] = df['Key'].replace([np.nan, None], float(total/num))

num = df['Loudness'].count()
total = df['Loudness'].sum()
df['Loudness'] = df['Loudness'].replace([np.nan, None], float(total/num))

num = df['Speechiness'].count()
total = df['Speechiness'].sum()
df['Speechiness'] = df['Speechiness'].replace([np.nan, None], float(total/num))

num = df['Acousticness'].count()
total = df['Acousticness'].sum()
df['Acousticness'] = df['Acousticness'].replace([np.nan, None], float(total/num))

num = df['Instrumentalness'].count()
total = df['Instrumentalness'].sum()
df['Instrumentalness'] = df['Instrumentalness'].replace([np.nan, None], float(total/num))

num = df['Liveness'].count()
total = df['Liveness'].sum()
df['Liveness'] = df['Liveness'].replace([np.nan, None], float(total/num))

num = df['Valence'].count()
total = df['Valence'].sum()
df['Valence'] = df['Valence'].replace([np.nan, None], float(total/num))

num = df['Tempo'].count()
total = df['Tempo'].sum()
df['Tempo'] = df['Tempo'].replace([np.nan, None], float(total/num))

num = df['Duration_ms'].count()
total = df['Duration_ms'].sum()
df['Duration_ms'] = df['Duration_ms'].replace([np.nan, None], float(total/num))

num = df['Views'].count()
total = df['Views'].sum()
df['Views'] = df['Views'].replace([np.nan, None], float(total/num))

num = df['Likes'].count()
total = df['Likes'].sum()
df['Likes'] = df['Likes'].replace([np.nan, None], float(total/num))

num = df['Stream'].count()
total = df['Stream'].sum()
df['Stream'] = df['Stream'].replace([np.nan, None], float(total/num))

num = df['Comments'].count()
total = df['Comments'].sum()
df['Comments'] = df['Comments'].replace([np.nan, None], float(total/num))

# df.info()
plt.figure(figsize=(20, 10))
sns.set(style="whitegrid")
corr = df.corr()
sns.heatmap(corr,annot=True, cmap="YlGnBu")

#test fillna
num = dt['Energy'].count()
total = dt['Energy'].sum()
dt['Energy'] = dt['Energy'].replace([np.nan, None], float(total/num))

num = dt['Key'].count()
total = dt['Key'].sum()
dt['Key'] = dt['Key'].replace([np.nan, None], float(total/num))

num = dt['Loudness'].count()
total = dt['Loudness'].sum()
dt['Loudness'] = dt['Loudness'].replace([np.nan, None], float(total/num))

num = dt['Speechiness'].count()
total = dt['Speechiness'].sum()
dt['Speechiness'] = dt['Speechiness'].replace([np.nan, None], float(total/num))

num = dt['Acousticness'].count()
total = dt['Acousticness'].sum()
dt['Acousticness'] = dt['Acousticness'].replace([np.nan, None], float(total/num))

num = dt['Instrumentalness'].count()
total = dt['Instrumentalness'].sum()
dt['Instrumentalness'] = dt['Instrumentalness'].replace([np.nan, None], float(total/num))

num = dt['Liveness'].count()
total = dt['Liveness'].sum()
dt['Liveness'] = dt['Liveness'].replace([np.nan, None], float(total/num))

num = dt['Valence'].count()
total = dt['Valence'].sum()
dt['Valence'] = dt['Valence'].replace([np.nan, None], float(total/num))

num = dt['Tempo'].count()
total = dt['Tempo'].sum()
dt['Tempo'] = dt['Tempo'].replace([np.nan, None], float(total/num))

num = dt['Duration_ms'].count()
total = dt['Duration_ms'].sum()
dt['Duration_ms'] = dt['Duration_ms'].replace([np.nan, None], float(total/num))

num = dt['Views'].count()
total = dt['Views'].sum()
dt['Views'] = dt['Views'].replace([np.nan, None], float(total/num))

num = dt['Likes'].count()
total = dt['Likes'].sum()
dt['Likes'] = dt['Likes'].replace([np.nan, None], float(total/num))

num = dt['Stream'].count()
total = dt['Stream'].sum()
dt['Stream'] = dt['Stream'].replace([np.nan, None], float(total/num))

num = dt['Comments'].count()
total = dt['Comments'].sum()
dt['Comments'] = dt['Comments'].replace([np.nan, None], float(total/num))

dt.info()
plt.figure(figsize=(20, 10))
sns.set(style="whitegrid")
corr = dt.corr()
sns.heatmap(corr,annot=True, cmap="YlGnBu")

from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

X = df[["Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Valence", "Track", "Title", "Channel", "Composer", "Artist"]]
Y = df["Danceability"]
testdata = dt[["Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Valence", "Track", "Title", "Channel", "Composer", "Artist"]]
id = dt["id"]
id = id.values.tolist()

#linear regression
regr = linear_model.LinearRegression()
regr.fit(X, Y)

predicttrain = regr.predict(X)
roundtrain =  [round(x) for x in predicttrain]
roundtrain = [ele if ele >= 0 else 0 for ele in roundtrain]
roundtrain = [ele if ele <= 9 else 9 for ele in roundtrain]

print(mean_absolute_error(Y, roundtrain))

res = regr.predict(testdata)
res =  [round(x) for x in res]
res = [ele if ele >= 0 else 0 for ele in res]
res = [ele if ele <= 9 else 9 for ele in res]

print(res)

head = ['id', 'Danceability']
with open('sub.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(head)
    for i in range(0, len(id)):
      data = []
      data.append(id[i])
      data.append(res[i])
      writer.writerow(data)


#random forest
from sklearn import tree
# from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib
import random
random.seed(1126)
p = np.random.permutation(13544)
# p = [i for i in range(13544)]
attr = ['Channel', 'Title', 'Track', 'Composer', 'Valence'] #, 'Acousticness', 'Loudness', 'Artist', 'Speechiness', 'Instrumentalness']
X = df[attr]
print(X.shape)
X_train = X.iloc[p[:10000]]
X_val = X.iloc[p[10000:]]
y = df["Danceability"]
y_train = y.iloc[p[:10000]]
y_val = y.iloc[p[10000:]]
testdata_random_forest = dt[attr]
print(X_train.shape, y_train.shape)
print(X_train.iloc[:2])
X_train = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train)
# transformer.transform(X_val)
X_val = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_val)
testdata_random_forest = PolynomialFeatures(degree=2, include_bias=False).fit_transform(testdata_random_forest)
# X_train.shape, y_train.shape = make_classification(n_samples=10000, n_features=100,
#               n_informative=2, n_redundant=0,
#               random_state=0, shuffle=True)
print(X_train.shape, y_train.shape)
# print(X_train.iloc[:2])

depth = []
ress_round = []
ress_no_round = []
for i in range(8, 17):
  # clf = tree.DecisionTreeClassifier(max_depth=i)
  clf = RandomForestRegressor(max_depth=i)
  clf = clf.fit(X_train, y_train)

  predictval = clf.predict(X_val)
  print("train:", mean_absolute_error(y_train, clf.predict(X_train)))
  roundtrain =  [round(x) for x in predictval]
  # roundtrain = [ele if ele >= 0 else 0 for ele in roundtrain]
  # roundtrain = [ele if ele <= 9 else 9 for ele in roundtrain]

  r = mean_absolute_error(y_val, roundtrain)

  print(r)
  depth.append(i)
  ress_round.append(r)
  r = mean_absolute_error(y_val, predictval)
  print(r)
  ress_no_round.append(r)

# print(len(ress), print(ress))
plt.scatter(depth, ress_round)
plt.show()

plt.scatter(depth, ress_no_round)
plt.show()

print(depth)
print(ress_round)


res_new = clf.predict(testdata_random_forest)
# res_new =  [round(x) for x in res]
# res_new = [ele if ele >= 0 else 0 for ele in res]
# res_new = [ele if ele <= 9 else 9 for ele in res]

print(res_new)


head = ['id', 'Danceability']
with open('sub.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(head)
    for i in range(0, len(id)):
      data = []
      data.append(id[i])
      data.append(res_new[i])
      writer.writerow(data)

print(mean_absolute_error(res_new, res))

attr = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness',
       'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo',
       'Duration_ms', 'Views', 'Likes', 'Stream', 'Album_type', 'Licensed',
       'official_video', 'id', 'Track', 'Album', 'Comments', 'Title',
       'Channel', 'Composer', 'Artist']
cor = [1, 0.044, 0.024, 0.25, 0.22, 0.29, 0.22, 0.082, 0.4, 0.092, 0.092, 0.089, 0.096, 0.071, 0.1, 0.017, 0.039, 0.19, 0.73, 0.014, 0.041, 0.76, 0.78, 0.43, 0.25]
tuplist = [[cor[i], attr[i]] for i in range(25)]
tuplist.sort()
tuplist.reverse()
print(tuplist)
print([i[0] for i in tuplist])
print([i[1] for i in tuplist])

from sklearn import tree
# from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import random
random.seed(1126)
p = np.random.permutation(13544)
# p = [i for i in range(13544)]
attr = ['Channel', 'Title', 'Track', 'Composer', 'Valence', 'Acousticness', 'Loudness', 'Artist', 'Speechiness', 'Instrumentalness']
X = df[attr]
print(X.shape)
X_train = X.iloc[p[:10000]]
X_val = X.iloc[p[10000:]]
y = df["Danceability"]
y_train = y.iloc[p[:10000]]
y_val = y.iloc[p[10000:]]
testdata_random_forest = dt[attr]
print(X_train.shape, y_train.shape)
print(X_train.iloc[:2])
X_train = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train)
# transformer.transform(X_val)
X_val = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_val)
testdata_random_forest = PolynomialFeatures(degree=2, include_bias=False).fit_transform(testdata_random_forest)
# X_train.shape, y_train.shape = make_classification(n_samples=10000, n_features=100,
#               n_informative=2, n_redundant=0,
#               random_state=0, shuffle=True)
print(X_train.shape, y_train.shape)
# print(X_train.iloc[:2])

depth = []
ress_round = []
ress_no_round = []
minR = 1e9
for i in range(13, 13):
  # clf = tree.DecisionTreeClassifier(max_depth=i)

  tmpclf = RandomForestClassifier(max_depth=i)
  tmpclf = tmpclf.fit(X_train, y_train)

  predictval = clf.predict(X_val)
  roundtrain =  [round(x) for x in predictval]
  roundtrain = [ele if ele >= 0 else 0 for ele in roundtrain]
  roundtrain = [ele if ele <= 9 else 9 for ele in roundtrain]

  r = mean_absolute_error(y_val, roundtrain)
  if(r < minR):
    clf = tmpclf
    r = minR

  print(r)
  depth.append(i)
  ress_round.append(r)
  r = mean_absolute_error(y_val, predictval)
  print(r)
  ress_no_round.append(r)

# print(len(ress), print(ress))
# plt.scatter(depth, ress_round)
# plt.show()

# plt.scatter(depth, ress_no_round)
# plt.show()

# print(depth)
# print(ress_round)


res_new = clf.predict(testdata_random_forest)
print(res_new)
res_new =  [round(x) for x in res_new]
res_new = [ele if ele >= 0 else 0 for ele in res_new]
res_new = [ele if ele <= 9 else 9 for ele in res_new]

print(res_new)


head = ['id', 'Danceability']
with open('sub.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(head)
    for i in range(0, len(id)):
      data = []
      data.append(id[i])
      data.append(res_new[i])
      writer.writerow(data)

from sklearn import tree
# from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import random
random.seed(1126)
p = np.random.permutation(13544)
# p = [i for i in range(13544)]
attr = ['Channel', 'Title', 'Track', 'Composer', 'Valence', 'Acousticness', 'Loudness', 'Artist', 'Speechiness', 'Instrumentalness']
X = df[attr]
print(X.shape)
X_train = X.iloc[p[:10000]]
X_val = X.iloc[p[10000:]]
y = df["Danceability"]
y_train = y.iloc[p[:10000]]
y_val = y.iloc[p[10000:]]
testdata_random_forest = dt[attr]
print(X_train.shape, y_train.shape)
print(X_train.iloc[:2])


#cat boost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

distribution = y.value_counts()
print(distribution)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, y, test_size=0.25, random_state=28)

# Define the hyperparameters for the CatBoost algorithm
params = {'learning_rate': 0.1, 'depth': 6,\
          'l2_leaf_reg': 3, 'iterations': 100}

# Initialize the CatBoostClassifier object
# with the defined hyperparameters and fit it on the training set
model = CatBoostClassifier(**params)
model.fit(X_train, y_train)

print(mean_absolute_error(y_train, model.predict(X_train)))
print(mean_absolute_error(y_val, model.predict(X_val)))