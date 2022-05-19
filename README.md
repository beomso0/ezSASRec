# customized **SASRec** (based on microsoft recommenders)

## custom_model
- 

### sas_train
- 

### sas_evaluate
- add parameters
  - target_user_n=1000 : 
  - target_item_n=-1: metric 산출 시 target label(마지막 interaction) 외에 추가할 neg_candidate의 수 | -1일 경우 target label 및 해당 user의 train,valid에 활용된 아이템 제외한 모든 아이템을 neg_candidate에 포함
  - rank_threshold=10 : NDCG@k 및 HR@k metrics의 k 값 

## custom_util
- save and load SASRec model

# Todo
- revise SASRecDATASET
- revise **predict**