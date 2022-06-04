from collections import defaultdict
import pickle 
import os
from .model import SASREC

class SASRecDataSet:
    """A class for creating SASRec specific dataset used during
    train, validation and testing.    

    Args:
        filename (str): Data Filename.
        col_sep (str): column separator in the data file.

    Attributes:
        usernum (int): Total number of users.
        itemnum (int): Total number of items.
        User (dict): All the users (keys) with items as values.
        Items (set): Set of all the items.
        user_train (dict): Subset of User that are used for training.
        user_valid (dict): Subset of User that are used for validation.
        user_test (dict): Subset of User that are used for testing.
        filename (str): Data Filename. Defaults to None.
        col_sep (str): Column separator in the data file. Defaults to '/t'.

    Examples:
        >>> data = SASRecDataSet('filename','/t')
    """

    def __init__(self, filename=None, col_sep='\t'):
        self.usernum = 0
        self.itemnum = 0
        self.User = defaultdict(list)
        self.Items = set()
        self.user_train = {}
        self.user_valid = {}
        self.user_test = {}
        self.filename = filename
        self.col_sep = col_sep

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
        data (pd.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating,
            timestamp, etc. can be optional.
        min_rating (int): Minimum number of ratings for user or item.
        filter_by (str): Either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): Column name of user ID.
        col_item (str): Column name of item ID.

    Returns:
        pandas.DataFrame: DataFrame with at least columns of user and item that has been filtered by the given specifications.
    """
    split_by_column = _get_column_name(filter_by, col_user, col_item)

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    return data.groupby(split_by_column).filter(lambda x: len(x) >= min_rating)

def filter_k_core(data, core_num=0, col_user="userID", col_item="itemID"):

    """Filter rating dataframe for minimum number of users and items by
    # repeatedly applying min_rating_filter until the condition is satisfied.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        core_num (int, optional): Minimun number for user and item to appear on data. Defaults to 0.
        col_user (str, optional): User column name. Defaults to "userID".
        col_item (str, optional): Item column name. Defaults to "itemID".

    Returns:
        pd.DataFrame: Filtered dataframe
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

def load_model(path, exp_name='sas_experiment'):
    """Load SASRec model

    Args:
        path (str): Path where the model is saved.
        exp_name (str, optional): Experiment name (folder name). Defaults to 'sas_experiment'.

    Returns:
        model.SASREC: loaded SASRec model
    """
    with open(f'{path}/{exp_name}/{exp_name}_model_args','rb') as f:
        arg_dict = pickle.load(f)
    
    if 'history' not in arg_dict.keys():
        arg_dict['history'] = None
        
    model = SASREC(item_num=arg_dict['item_num'],
                    seq_max_len=arg_dict['seq_max_len'],
                    num_blocks=arg_dict['num_blocks'],
                    embedding_dim=arg_dict['embedding_dim'],
                    attention_dim=arg_dict['attention_dim'],
                    attention_num_heads=arg_dict['attention_num_heads'],
                    dropout_rate=arg_dict['dropout_rate'],
                    conv_dims = arg_dict['conv_dims'],
                    l2_reg=arg_dict['l2_reg'],
                    history=arg_dict['history'],
        )

    model.load_weights(f'{path}/{exp_name}/{exp_name}_weights')
    
    return model