# customized **SASRec** (based on microsoft recommenders)

## 1. custom_model
- train and evaluate SASRec model
### a. sas_evaluate

- added parameters
  - target_user_n=1000 : 
  - target_item_n=-1: metric 산출 시 target label(마지막 interaction) 외에 추가할 neg_candidate의 수 | -1일 경우 target label 및 해당 user의 train,valid에 활용된 아이템 제외한 모든 아이템을 neg_candidate에 포함
  - rank_threshold=10 : NDCG@k 및 HR@k metrics의 k 값
  </br>
- usage

```python
sas_evaluate(test_model,data, target_user_n=1000, target_item_n=-1,rank_threshold=5)
```
### b. sas_train
- added parameters
  - target_user_n=1000 : for sas_evaluate
  - target_item_n=-1: for sas_evaluate
    </br>
  - usage

```python
sas_train(test_model,data,sampler,num_epochs=num_epochs, batch_size=batch_size, learning_rate=lr, val_epoch=5, target_user_n=10000, target_item_n=-1)
```


## 2. custom_util
- save and load SASRec model
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
  - path: 저장할 경로
  - exp_name: 실험(모델) 이름을 suffix로 추가

### a. load_sasrec_model
- save_sasrec_model로 저장한 SASRec 객체의 weight와 args 파일을 불러와서 SASRec 객체 생성
</br>
- usage

```python
loaded_model = load_sasrec_model(path, exp_name='save_test')
```

  </br>

- parameters
  - path: 파일**들**이 저장된 경로
  - exp_name: 실험(모델) 이름
    - save 시 지정한 suffix


# Todo
- revise SASRecDATASET
- revise **predict**