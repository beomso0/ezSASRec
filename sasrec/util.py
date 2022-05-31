from collections import defaultdict
import pickle 
import os
from .model import SASREC

class SASRecDataSet:
    """
    A class for creating SASRec specific dataset used during
    train, validation and testing.

    Attributes:
        usernum: integer, total number of users
        itemnum: integer, total number of items
        User: dict, all the users (keys) with items as values
        Items: set of all the items
        user_train: dict, subset of User that are used for training
        user_valid: dict, subset of User that are used for validation
        user_test: dict, subset of User that are used for testing
        col_sep: column separator in the data file
        filename: data filename
    """

    def __init__(self, **kwargs):
        self.usernum = 0
        self.itemnum = 0
        self.User = defaultdict(list)
        self.Items = set()
        self.user_train = {}
        self.user_valid = {}
        self.user_test = {}
        self.col_sep = kwargs.get("col_sep", " ")
        self.filename = kwargs.get("filename", None)

        if self.filename:
            with open(self.filename, "r") as fr:
                sample = fr.readline()
            ncols = sample.strip().split(self.col_sep)
            if ncols == 3:
                self.with_time = True
            else:
                self.with_time = False

    def split(self, **kwargs):
        self.filename = kwargs.get("filename", self.filename)
        if not self.filename:
            raise ValueError("Filename is required")

        if self.with_time:
            self.data_partition_with_time()
        else:
            self.data_partition()

    def data_partition(self):
        # assume user/item index starting from 1
        f = open(self.filename, "r")
        for line in f:
            u, i = line.rstrip().split(self.col_sep)
            u = int(u)
            i = int(i)
            self.usernum = max(u, self.usernum)
            self.itemnum = max(i, self.itemnum)
            self.User[u].append(i)

        for user in self.User:
            nfeedback = len(self.User[user])
            if nfeedback < 3:
                self.user_train[user] = self.User[user]
                self.user_valid[user] = []
                self.user_test[user] = []
            else:
                self.user_train[user] = self.User[user][:-2]
                self.user_valid[user] = []
                self.user_valid[user].append(self.User[user][-2])
                self.user_test[user] = []
                self.user_test[user].append(self.User[user][-1])

    def data_partition_with_time(self):
        # assume user/item index starting from 1
        f = open(self.filename, "r")
        for line in f:
            u, i, t = line.rstrip().split(self.col_sep)
            u = int(u)
            i = int(i)
            t = float(t)
            self.usernum = max(u, self.usernum)
            self.itemnum = max(i, self.itemnum)
            self.User[u].append((i, t))
            self.Items.add(i)

        for user in self.User.keys():
            # sort by time
            items = sorted(self.User[user], key=lambda x: x[1])
            # keep only the items
            items = [x[0] for x in items]
            self.User[user] = items
            nfeedback = len(self.User[user])
            if nfeedback < 3:
                self.user_train[user] = self.User[user]
                self.user_valid[user] = []
                self.user_test[user] = []
            else:
                self.user_train[user] = self.User[user][:-2]
                self.user_valid[user] = []
                self.user_valid[user].append(self.User[user][-2])
                self.user_test[user] = []
                self.user_test[user].append(self.User[user][-1])

def _get_column_name(name, col_user, col_item):
    if name == "user":
        return col_user
    elif name == "item":
        return col_item
    else:
        raise ValueError("name should be either 'user' or 'item'.")

def min_rating_filter_pandas(
    data,
    min_rating=1,
    filter_by="user",
    col_user="userID",
    col_item="itemID",
):
    """Filter rating DataFrame for each user with minimum rating.

    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.

    Args:
        data (pandas.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating,
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        pandas.DataFrame: DataFrame with at least columns of user and item that has been filtered by the given specifications.
    """
    split_by_column = _get_column_name(filter_by, col_user, col_item)

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    return data.groupby(split_by_column).filter(lambda x: len(x) >= min_rating)

def filter_k_core(data, core_num=0, col_user="userID", col_item="itemID"):
    """Filter rating dataframe for minimum number of users and items by
    repeatedly applying min_rating_filter until the condition is satisfied.

    """
    num_users, num_items = data[col_user].nunique(), data[col_item].nunique()
    print(f"Original: {num_users} users and {num_items} items")
    df_inp = data.copy()

    if core_num > 0:
        while True:
            df_inp = min_rating_filter_pandas(
                df_inp, min_rating=core_num, filter_by="item"
            )
            df_inp = min_rating_filter_pandas(
                df_inp, min_rating=core_num, filter_by="user"
            )
            count_u = df_inp.groupby(col_user)[col_item].count()
            count_i = df_inp.groupby(col_item)[col_user].count()
            if (
                len(count_i[count_i < core_num]) == 0
                and len(count_u[count_u < core_num]) == 0
            ):
                break
    df_inp = df_inp.sort_values(by=[col_user])
    num_users = df_inp[col_user].nunique()
    num_items = df_inp[col_item].nunique()
    print(f"Final: {num_users} users and {num_items} items")

    return df_inp

def load_sasrec_model(path, exp_name='sas_experiment'):
  with open(path+exp_name+'/'+exp_name+'_model_args','rb') as f:
    arg_dict = pickle.load(f)
  model = SASREC(item_num=arg_dict['item_num'],
                   seq_max_len=arg_dict['seq_max_len'],
                   num_blocks=arg_dict['num_blocks'],
                   embedding_dim=arg_dict['embedding_dim'],
                   attention_dim=arg_dict['attention_dim'],
                   attention_num_heads=arg_dict['attention_num_heads'],
                   dropout_rate=arg_dict['dropout_rate'],
                   conv_dims = arg_dict['conv_dims'],
                   l2_reg=arg_dict['l2_reg'],
                   num_neg_test=arg_dict['num_neg_test'],
    )
  model.load_weights(path+exp_name+'/'+exp_name+'_weights')
  return model