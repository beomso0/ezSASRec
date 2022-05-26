# customized **SASRec** (based on microsoft recommenders)

## 1. custom_model
train and evaluate SASRec model
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
    - save 시 지정한 suffix


# Todo
- revise SASRecDATASET
- revise **predict**