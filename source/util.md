# util

---

## sasrec.util.SASRecDataSet

A class for creating SASRec dataset used during train, validation and testing. 

```python
sasrec.util.SASRecDataSet(filename,col_sep)
``` 

- Attributes:
  - **usernum**: *(integer)*, total number of users
  - **itemnum**: *(integer)*, total number of items
  - **User**: *(dict)*, all the users (keys) with items as values
  - **Items**: set of all the items
  - **user_train**: *(dict)*, subset of User that are used for - training
  - **user_valid**: *(dict)*, subset of User that are used for - validation
  - **user_test**: *(dict)*, subset of User that are used for testing
  - **col_sep**: column separator in the data file
  - **filename**: data filename

---

## sasrec.util.filter_k_core

Filter rating dataframe for minimum number of users and items by repeatedly applying min_rating_filter until the condition is satisfied. 

```python
sasrec.util.filter_k_core(data, core_num=0, col_user="userID", col_item="itemID")
```

- Args:
  - **data**: *(pd.DataFrame)*, dataframe to filter
  - **core_num**: *(integer)*, minimum number of appearance
  - **col_user**: *(str)*, name of user col
  - **col_item**: *(str)*, name of item col

---

## sasrec.util.load_sasrec_model

Load saved SASRec model.  

```python
sasrec.util.load_sasrec_model(path, exp_name='sas_experiment')
```

- Args:
  - **path**: *(str)*, path where model is saved
  - **exp_name**: *(str)*, exp_name suffix of the model to load