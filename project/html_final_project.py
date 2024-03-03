import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import csv

#train data organized
df = pd.read_csv('train.csv')
df = df.dropna(thresh = 24) #remove some data with too many null value
ori = df.copy()

df['Album_type'] = df['Album_type'].replace(['single','album', 'compilation',None, np.nan], [0, 1, 2, -1, -1])
df['Licensed'] = df['Licensed'].replace([True, False, None, np.nan], [1, -1, 0, 0])
df['official_video'] = df['official_video'].replace([True, False, None, np.nan], [1, -1, 0, 0])

#test data organized
dt = pd.read_csv('test.csv')
oritest = dt.copy()

dt['Album_type'] = dt['Album_type'].replace(['single','album', 'compilation',None, np.nan], [0, 1, 2, -1, -1])
dt['Licensed'] = dt['Licensed'].replace([True, False, None, np.nan], [1, -1, 0, 0])
dt['official_video'] = dt['official_video'].replace([True, False, None, np.nan], [1, -1, 0, 0])

dt.info()

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
    if x in z_artist:
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
    if x in z_com:
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
    if x in z_ch:
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
    if x in z_title:
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
    if x in z_tr:
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
    if x in z_album:
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

df.info()
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

df.info()

#"Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Views", "Likes", "Stream", "Composer", "Tempo", "Duration_ms"
feature = ["Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Views", "Likes", "Stream", "Composer"]
X = df[feature]
y = df["Danceability"]
testdata = dt[feature]

import numpy as np
from sklearn.model_selection import train_test_split
xn = X.to_numpy()
yn = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(xn, yn, test_size=0.33, random_state=42)

#linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)

xin = reg.predict(X_train)
xval = reg.predict(X_test)

roundxin =  [round(x) for x in xin]
roundxin = [ele if ele >= 0 else 0 for ele in roundxin]
roundxin = [ele if ele <= 9 else 9 for ele in roundxin]
roundxval =  [round(x) for x in xval]
roundxval = [ele if ele >= 0 else 0 for ele in roundxval]
roundxval = [ele if ele <= 9 else 9 for ele in roundxval]

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_train, roundxin))
print(mean_absolute_error(y_test, roundxval))

#SVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf_svm.fit(X, y)

res_svm = clf_svm.predict(testdata)

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(testdata)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y)
res_neighbor = neigh.predict(X_test)

#xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X, y)

res_xgb = xgb.predict(testdata)

#neural network
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier().fit(X, y)
res_mlp = mlp.predict(testdata)

#Adaboost
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
ada.fit(X, y)
res_ada = ada.predict(testdata)

# output result
id = dt["id"]
head = ['id', 'Danceability']
with open('svm.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(head)
    for i in range(0, len(id)):
      data = []
      data.append(id[i])
      data.append(int(res_svm[i]))
      writer.writerow(data)

head = ['id', 'Danceability']
with open('knn.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(head)
    for i in range(0, len(id)):
      data = []
      data.append(id[i])
      data.append(int(res_neighbor[i]))
      writer.writerow(data)

head = ['id', 'Danceability']
with open('xgb.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(head)
    for i in range(0, len(id)):
      data = []
      data.append(id[i])
      data.append(int(res_xgb[i]))
      writer.writerow(data)

head = ['id', 'Danceability']
with open('neural.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(head)
    for i in range(0, len(id)):
      data = []
      data.append(id[i])
      data.append(int(res_mlp[i]))
      writer.writerow(data)

head = ['id', 'Danceability']
with open('ada.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(head)
    for i in range(0, len(id)):
      data = []
      data.append(id[i])
      data.append(int(res_ada[i]))
      writer.writerow(data)


from sklearn.ensemble import VotingClassifier
clf1 = SVC()
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = XGBClassifier()
clf4 = MLPClassifier()
clf5 = AdaBoostClassifier()
eclf1 = VotingClassifier(estimators=[('svm', clf1), ('knn', clf2), ('neural', clf4), ('ada', clf5)], voting='hard')
eclf1 = eclf1.fit(X, y)
res_mix = eclf1.predict(testdata)

from collections import Counter
head = ['id', 'Danceability']
with open('mix.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(head)
    for i in range(0, len(id)):
      data = []
      data.append(id[i])
      data.append(res_mix[i])
      writer.writerow(data)