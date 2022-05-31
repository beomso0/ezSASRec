# model

SAS Rec model
Self-Attentive Sequential Recommendation Using Transformer  

**Wang-Cheng Kang, Julian McAuley (2018), Self-Attentive Sequential Recommendation. Proceedings of IEEE International Conference on Data Mining (ICDM'18)**  
*[Original source code from nnkkmto/SASRec-tf2](https://github.com/nnkkmto/SASRec-tf2)*

## sasrec.model.SASREC

SASREC model initialization.

```python
sasrec.model.SASREC(item_num, seq_max_len=100, num_blocks=2, embedding_dim=100, attention_dim=100, conv_dims=[100,100], dropout_rate=0.5, l2_reg=0.0)
```

- Args:
    - **item_num**: *(int)*, Number of items in the dataset.
    - **seq_max_len**: *(int)*, Maximum number of items in user history.
    - **num_blocks**: *(int)*, Number of Transformer blocks to be used.
    - **embedding_dim**: *(int)*, Item embedding dimension.
    - **attention_dim**: *(int)*, Transformer attention dimension.
    - **conv_dims**: *(list)*, List of the dimensions of the Feedforward layer.
    - **dropout_rate**: *(float)*, Dropout rate.
    - **l2_reg**: *(float)*, Coefficient of the L2 regularization.  

</br>

- Attributes:
  - **val_users**: *(list)*, user list to execute validation.
  - **history**: *(pd.DataFrame)*, dataframe containing validation history.

### sasrec.model.SASREC.train

High level function for model training as well as well as evaluation on the validation and test dataset

```python
sasrec.model.SASREC.train(dataset, sampler,num_epochs=10,batch_size=128,lr=0.001,val_epoch=5,val_target_user_n=1000,target_item_n=-1,auto_save=False,path='./',exp_name='SASRec_exp')
```

- Args:
  - **dataset**: *(sasrec.util.SASRecDataSet)* SASRecDataSet object
  - **sampler**: *(sasrec.sampler.WarpSampler)* WarpSampler object
  - **num_epochs**: *(int)*, number of epochs
  - **batch_size**: *(int)*, size of batch
  - **lr**: *(float)*, learning rate
  - **val_epoch**: *(int)*, validation interval
  - **val_target_user_n**: *(int)*, number of users for validation.
  - **target_item_n**: *(int)*, number of items(including negative samples) for validation. *-1* means all.
  - **auto_save**: *(bool)*, flag to automatically save best score model
  - **path**: *(str)*, path to save model, optional
  - **exp_name**: *(str)*, name of experiment, optional

### sasrec.model.SASREC.predict

### sasrec.model.SASREC.evaluate


### sasrec.model.SASREC.recommend_item


### sasrec.model.SASREC.get_user_item_score


### sasrec.model.SASREC.save


