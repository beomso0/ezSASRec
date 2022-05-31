# model

SAS Rec model
Self-Attentive Sequential Recommendation Using Transformer  

**Wang-Cheng Kang, Julian McAuley (2018), Self-Attentive Sequential Recommendation. Proceedings of IEEE International Conference on Data Mining (ICDM'18)**  
*[Original source code from nnkkmto/SASRec-tf2](https://github.com/nnkkmto/SASRec-tf2)*

## sasrec.model.SASREC

SASREC model initialization.

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

### sasrec.model.SASREC.predict


### sasrec.model.SASREC.train


### sasrec.model.SASREC.evaluate


### sasrec.model.SASREC.recommend_item


### sasrec.model.SASREC.get_user_item_score


### sasrec.model.SASREC.save


