# **ezSASRec: The easiest way to utilize SASRec (Self-Attentive Sequential Recommendation) for your system **


An easy and efficient tool to build sequential recommendation system utilizing SASRec


## Documentation
https://ezsasrec.netlify.app

## References

### repos
1. [kang205 SASRec](https://github.com/kang205/SASRec)
2. [nnkkmto/SASRec-tf2](https://github.com/nnkkmto/SASRec-tf2)
3. [microsoft recommenders](https://github.com/microsoft/recommenders)

### papers
1. [Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)
2. [A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models](https://www.informatik.uni-wuerzburg.de/datascience/staff/dallmann/?tx_extbibsonomycsl_publicationlist%5Baction%5D=download&tx_extbibsonomycsl_publicationlist%5Bcontroller%5D=Document&tx_extbibsonomycsl_publicationlist%5BfileName%5D=main.pdf&tx_extbibsonomycsl_publicationlist%5BintraHash%5D=23f589b27e22018936753bb64b33971d&tx_extbibsonomycsl_publicationlist%5BuserName%5D=dallmann&cHash=dd7c54126f6c20972a502e9cc223cec2)

---------------
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




<!-- 
## Introduction
This repository contains tools to train, evaluate and save SASRec model.
- - - 
Original codes and architectures are from 
  - https://github.com/kang205/SASRec
  - https://github.com/microsoft/recommenders/tree/main/recommenders/models/sasrec

## References
1. [Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)
2. [A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models](https://www.informatik.uni-wuerzburg.de/datascience/staff/dallmann/?tx_extbibsonomycsl_publicationlist%5Baction%5D=download&tx_extbibsonomycsl_publicationlist%5Bcontroller%5D=Document&tx_extbibsonomycsl_publicationlist%5BfileName%5D=main.pdf&tx_extbibsonomycsl_publicationlist%5BintraHash%5D=23f589b27e22018936753bb64b33971d&tx_extbibsonomycsl_publicationlist%5BuserName%5D=dallmann&cHash=dd7c54126f6c20972a502e9cc223cec2)

## Quick Start
### a. sas_evaluate

- added parameters
  - target_user_n=1000 : 
  - target_item_n=-1: metric 산출 시 target label(마지막 interaction) 외에 추가할 neg_candidate의 수 | -1일 경우 target label 및 해당 user의 train,valid에 활용된 아이템 제외한 모든 아이템을 neg_candidate에 포함
  - rank_threshold=10 : NDCG@k 및 HR@k metrics의 k 값
  - is_val : True -> validation score 계산 || False -> test score 계산
  </br>
- usage

  ```python
  sas_evaluate(test_model,data, target_user_n=1000, target_item_n=-1,rank_threshold=5)
  ```
### b. sas_train
- added parameters
  - target_user_n=1000 : for sas_evaluate
  - target_item_n=-1: for sas_evaluate
  - auto_save : 학습 시, best HR@10 score 모델을 자동으로 저장할지 여부 ([save_sasrec_model source code](custom_SASRec/custom_util.py))
  - path : 저장 경로
  - exp_name : 실험 이름 (저장 시 suffix)
  </br>
  
- usage

  ```python
  sas_train(test_model,data,sampler,num_epochs=num_epochs, batch_size=batch_size, learning_rate=lr, val_epoch=5, target_user_n=10000, target_item_n=-1)
  ```
### c. sas_get_prediction
- user에 대한 상위 n개 추천 아이템 산출
- parameters
  - model_ : SASRec model
  - dataset : SASRecDATASET
  - user_map_dict : {original : EncodedLabel} 형태의 dict
  - user_id_list : 추천을 받고자 하는 user id의 list
  - target_item_n : 추천 후보 수 
    - randomly sampled
    - 전체 후보 -> target_item_n = -1
  - top_n : 추천 item 수 (상위 n개)
  - exclude_purchased : 해당 user가 이미 구매한 item을 추천 후보에서 제외할지 여부
  - is_test : 각 user의 sequence에서 마지막 1개 item(test target)을 제외한 sequence를 기반으로 추천할지 여부.
- return
  ```
  {user_id : [(encoded_item_id, pred_score) ...]}
  ```
    </br>
  
- usage

  ```python
  pred = sas_get_prediction(loaded_model,data,user_map,user_id_list,is_test=True)
  ```
### d. sas_predict
- added parameters
  - neg_cand_n: test target 외에 추가된 neg_candidates의 수.
    -  행렬 연산이 끝난 뒤에 output(test_logits)을 reshape할 때 필요함
    - 학습 및 평가 시에는 sas_evaluate 함수에서 자동으로 값을 전달
    - 최종 배포 시에는 ```neg_cand_n=0```으로 지정

## 2. custom_util
save and load SASRec model
### a. save_sasrec_model
- 학습이 완료된 SASRec 객체의 weight와 args를 각각 파일로 저장함.
</br>
- usage

  ```python
  save_sasrec_model(test_model, path, exp_name='save_test')
  ```
  </br>

- parameters
  - model: SASRec 객체
  - path: 저장할 경로 (저장 시 path 아래에 exp_name의 폴더 생성)
  - exp_name: 실험(모델) 이름을 suffix로 추가
- outputs
  - {exp_name}_weights : 학습된 모델의 weights들을 담은 파일
  - {exp_name}_train_log.txt : model의 parameter와 update log 확인 가능
  - {exp_name}_model_args : SASRec 모델의 parameter dict를 담은 binary 파일

### b. load_sasrec_model
- save_sasrec_model로 저장한 SASRec 객체의 weight와 args 파일을 불러와서 SASRec 객체 생성
</br>
- usage

  ```python
  loaded_model = load_sasrec_model(path, exp_name='save_test')
  ```

- parameters
  - path: 파일**들**이 저장된 경로
  - exp_name: 실험(모델) 이름
    - save 시 지정한 suffix -->