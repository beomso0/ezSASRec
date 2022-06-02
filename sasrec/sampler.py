import multiprocessing
import numpy as np
from multiprocessing import Process, Queue
import tensorflow as tf


def random_neq(left, right, s):
    t = np.random.randint(left, right)
    while t in s:
        t = np.random.randint(left, right)
    return t


def sample_function(
    user_train, usernum, itemnum, batch_size, maxlen, result_queue, seed
):
    """Batch sampler that creates a sequence of negative items based on the
    original sequence of items (positive) that the user has interacted with.

    Args:
        user_train (dict): dictionary of training exampled for each user
        usernum (int): number of users
        itemnum (int): number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        result_queue (multiprocessing.Queue): queue for storing sample results
        seed (int): seed for random generator
    """

    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(seed)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """Sampler object that creates an iterator for feeding batch data while training.

    Attributes:
        User: dict, all the users (keys) with items as values
        usernum: integer, total number of users
        itemnum: integer, total number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        n_workers (int): number of workers for parallel execution
    """

    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        User,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        np.random.randint(2e9),
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# class PredictSampler(object):
#     """Sampler object that creates an iterator for feeding batch data while predicting.

#     Attributes:
#         User: dict, all the users (keys) with items as values
#         usernum: integer, total number of users
#         itemnum: integer, total number of items
#         batch_size (int): batch size
#         maxlen (int): maximum input sequence length
#         n_workers (int): number of workers for parallel execution
#     """

#     def __init__(self, User, user_map_dict,user_id_list, batch_size=128,n_workers=1):
        
#         self.result_queue = Queue(maxsize=n_workers * 10)
#         self.processors = []
#         mgr = Manager()
#         self.mgr_user_list = mgr.list(user_id_list)
#         for _ in range(n_workers):
#             self.processors.append(
#                 Process(
#                     target=predict_sample_function,
#                     args=(
#                         User,
#                         user_map_dict,
#                         batch_size,
#                         self.result_queue,
#                         self.mgr_user_list
#                     ),
#                 )
#             )
#             self.processors[-1].daemon = True
#             self.processors[-1].start()

#     def next_batch(self):
#         return self.result_queue.get()

#     def close(self):
#         for p in self.processors:
#             p.terminate()
#             p.join()

# def predict_sample_function(
#     user_history, user_map_dict,batch_size, result_queue, mgr_user_list
# ):
#     """Batch sampler that creates a sequence of negative items based on the
#     original sequence of items (positive) that the user has interacted with.

#     Args:
#         user_train (dict): dictionary of training exampled for each user
#         usernum (int): number of users
#         itemnum (int): number of items
#         batch_size (int): batch size
#         maxlen (int): maximum input sequence length
#         result_queue (multiprocessing.Queue): queue for storing sample results
#         seed (int): seed for random generator
#     """
    

#     def sample():
        
#         user_id = mgr_user_list.pop()
#         user = user_map_dict[user_id]
#         seq = user_history[user]

#         return (user_id, seq)

#     # original
#     while True:
#         one_batch = []
#         for i in range(batch_size):
#             try:
#                 one_batch.append(sample())
#             except IndexError:
#                 break

#         result_queue.put(zip(*one_batch))
#         print(len(mgr_user_list))
#         if len(mgr_user_list)<=0:
#             print('while loop break')
#             break
