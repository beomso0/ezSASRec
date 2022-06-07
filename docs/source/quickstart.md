# **QuickStart**
example data source: [link](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

```python
import pandas as pd 
import pickle
from sasrec.util import filter_k_core, SASRecDataSet, load_model
from sasrec.model import SASREC
from sasrec.sampler import WarpSampler
import multiprocessing
```

## Preprocessing


```python
path = 'your path'
```


```python
df = pd.read_csv('ratings.csv')
df = df.rename({'userId':'userID','movieId':'itemID','timestamp':'time'},axis=1)\
       .sort_values(by=['userID','time'])\
       .drop(['rating','time'],axis=1)\
       .reset_index(drop=True)
```


```python
df.head()
```


|   | userID | itemID |
|---|--------|--------|
| 0 | 1      | 2762   |
| 1 | 1      | 54503  |
| 2 | 1      | 112552 |
| 3 | 1      | 96821  |
| 4 | 1      | 5577   |


```python
# filter data
# every user and item will appear more than 6 times in filtered_df

filtered_df = filter_k_core(df, 7)
```

    Original: 270896 users and 45115 items
    Final: 243377 users and 24068 items



```python
# make maps (encoder)

user_set, item_set = set(filtered_df['userID'].unique()), set(filtered_df['itemID'].unique())
user_map = dict()
item_map = dict()
for u, user in enumerate(user_set):
    user_map[user] = u+1
for i, item in enumerate(item_set):
    item_map[item] = i+1

maps = (user_map, item_map)   
```


```python
# Encode filtered_df

filtered_df["userID"] = filtered_df["userID"].apply(lambda x: user_map[x])
filtered_df["itemID"] = filtered_df["itemID"].apply(lambda x: item_map[x])
```

```python
# save data and maps

# save sasrec data    
filtered_df.to_csv('sasrec_data.txt', sep="\t", header=False, index=False)

# save maps
with open('maps.pkl','wb') as f:
    pickle.dump(maps, f)
```

## Load data and Train model


```python
# load data

data = SASRecDataSet('sasrec_data.txt')
data.split() # train, val, test split
              # the last interactions of each user is used for test
              # the last but one will be used for validation
              # others will be used for train
```


```python
# make model and warmsampler for batch training

max_len = 100
hidden_units = 128
batch_size = 2048

model = SASREC(
    item_num=data.itemnum,
    seq_max_len=max_len,
    num_blocks=2,
    embedding_dim=hidden_units,
    attention_dim=hidden_units,
    attention_num_heads=2,
    dropout_rate=0.2,
    conv_dims = [hidden_units, hidden_units],
    l2_reg=0.00001
)

sampler = WarpSampler(data.user_train, data.usernum, data.itemnum, batch_size=batch_size, maxlen=max_len, n_workers=multiprocessing.cpu_count())
```


```python
# train model

model.train(
          data,
          sampler,
          num_epochs=3, 
          batch_size=batch_size, 
          lr=0.001, 
          val_epoch=1,
          val_target_user_n=1000, 
          target_item_n=-1,
          auto_save=True,
          path = path,
          exp_name='exp_example',
        )
```
    epoch 1 / 3 -----------------------------

    Evaluating...    

    epoch: 1, test (NDCG@10: 0.04607630127474612, HR@10: 0.097)
    best score model updated and saved


    epoch 2 / 3 -----------------------------

    Evaluating...    

    epoch: 2, test (NDCG@10: 0.060855185638025944, HR@10: 0.118)
    best score model updated and saved


    epoch 3 / 3 -----------------------------

    Evaluating...   

    epoch: 3, test (NDCG@10: 0.07027207563856912, HR@10: 0.139)
    best score model updated and saved


## Predict


```python
# load trained model

model = load_model(path,'exp_example')
```

### get score


```python
# get user-item score

# make inv_user_map
inv_user_map = {v: k for k, v in user_map.items()}

# sample target_user
model.sample_val_users(data, 100)
encoded_users = model.val_users

# get scores
score = model.get_user_item_score(data,
                          [inv_user_map[u] for u in encoded_users], # user_list containing raw(not-encoded) userID 
                          [1,2,3], # item_list containing raw(not-encoded) itemID
                          user_map,
                          item_map,   
                          batch_size=10
                        )
```
    100%|██████████| 10/10 [00:00<00:00, 29.67batch/s]
```python
score.head()
```

|   |userID|        1 |        2 |         3|
|--:|-----:|---------:|---------:|---------:|
| 0 | 1525 | 5.596944 | 4.241653 | 3.804743 |
| 1 | 1756 | 4.535607 | 2.694459 | 0.858440 |
| 2 | 2408 | 5.883061 | 4.655960 | 4.691791 |
| 3 | 2462 | 5.084695 | 2.942075 | 2.773376 |
| 4 | 3341 | 5.532438 | 4.348150 | 4.073740 |


### get recommendation


```python
# get top N recommendation 

reco = model.recommend_item(data,
                            user_map,
                            [inv_user_map[u] for u in encoded_users],
                            is_test=True,
                            top_n=5)
```
    100%|██████████| 100/100 [00:04<00:00, 21.10it/s]

```python
# returned tuple contains topN recommendations for each user

reco
```
    {1525: [(456, 6.0680223),
      (355, 6.033769),
      (379, 5.9833336),
      (591, 5.9718275),
      (776, 5.8978705)],
     1756: [(7088, 5.735977),
      (15544, 5.5946136),
      (5904, 5.500249),
      (355, 5.492655),
      (22149, 5.4117346)],
     2408: [(456, 5.976555),
      (328, 5.8824606),
      (588, 5.8614006),
      (264, 5.7114534),
      (299, 5.649914)],
     2462: [(259, 6.3445344),
      (591, 6.2664876),
      (295, 6.105361),
      (355, 6.0698805),
      (1201, 5.8477645)],
     3341: [(110, 5.510764),
      (1, 5.4927354),
      (259, 5.4851904),
      (161, 5.467624),
      (208, 5.2486935)], ...}