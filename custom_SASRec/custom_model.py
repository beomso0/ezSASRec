from matplotlib.pyplot import autoscale
from recommenders.models.sasrec.model import SASREC
from recommenders.utils.timer import Timer
from .custom_util import save_sasrec_model
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf

def sas_train(model, dataset, sampler, **kwargs):
        """
        High level function for model training as well as
        evaluation on the validation and test dataset
        
        <kwargs>
        num_epochs
        batch_size
        learning_rate
        val_epoch : 몇 epoch마다 validation 진행할지
        val_target_user_n : validation 
        target_item_n
        auto_save
        path
        exp_name
        """
        num_epochs = kwargs.get("num_epochs", 10)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("learning_rate", 0.001)
        val_epoch = kwargs.get("val_epoch", 5)
        val_target_user_n =kwargs.get("val_target_user_n",1000)
        target_item_n = kwargs.get("target_item_n",-1)
        auto_save = kwargs.get("auto_save",True)
        path = kwargs.get("path",'./')
        exp_name = kwargs.get("exp_name",'SASRec_exp')
        best_score = 0

        num_steps = int(len(dataset.user_train) / batch_size)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7
        )

        loss_function = model.loss_function

        train_loss = tf.keras.metrics.Mean(name="train_loss")

        train_step_signature = [
            {
                "users": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
                "input_seq": tf.TensorSpec(
                    shape=(None, model.seq_max_len), dtype=tf.int64
                ),
                "positive": tf.TensorSpec(
                    shape=(None, model.seq_max_len), dtype=tf.int64
                ),
                "negative": tf.TensorSpec(
                    shape=(None, model.seq_max_len), dtype=tf.int64
                ),
            },
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            with tf.GradientTape() as tape:
                pos_logits, neg_logits, loss_mask = model(inp, training=True)
                loss = loss_function(pos_logits, neg_logits, loss_mask)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            return loss

        T = 0.0
        t0 = Timer()
        t0.start()

        for epoch in range(1, num_epochs + 1):

            print(f'epoch {epoch} / {num_epochs} started---------------------')

            step_loss = []
            train_loss.reset_states()
            for step in tqdm(
                range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
            ):

                u, seq, pos, neg = sampler.next_batch()

                inputs, target = model.create_combined_dataset(u, seq, pos, neg)

                loss = train_step(inputs, target)
                step_loss.append(loss)

            if epoch % val_epoch == 0:                
                print("Evaluating...")
                t_test = sas_evaluate(model,dataset,target_user_n=val_target_user_n,target_item_n=target_item_n,is_val=True)
                print(
                    f"epoch: {epoch}, time: {T},  test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})"
                )

                if auto_save:
                    if t_test[1] > best_score:
                        best_score = t_test[1]
                        save_sasrec_model(model,path,exp_name,save_info={'score':t_test[1],'epoch':epoch})
                        print('best score model updated and saved')
                    else:
                        pass
                else:
                    pass

                

def sas_evaluate(model_, dataset, target_user_n=1000, target_item_n=-1, rank_threshold=10,is_val=False):

        """
        Evaluation on the test users (users with at least 3 items)

        <kwargs>
        model_ | dataset: SASRecDataSet 객체 | target_user_n: evaluate할 user 수 | target_item_n
        """
        usernum = dataset.usernum
        itemnum = dataset.itemnum
        all = dataset.User
        train = dataset.user_train  # removing deepcopy
        valid = dataset.user_valid
        test = dataset.user_test

        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0

        if usernum > target_user_n:
            users = random.sample(range(1, usernum + 1), target_user_n)
        else:
            users = range(1, usernum + 1)
        # users = range(1,11)

        for u in tqdm(users, ncols=70, leave=False, unit="b"):

            if len(train[u]) < 1 or len(test[u]) < 1:
                continue

            seq = np.zeros([model_.seq_max_len], dtype=np.int32)
            idx = model_.seq_max_len - 1

            if is_val:                
                item_idx = [valid[u][0]]
            else:
                seq[idx] = valid[u][0]
                idx -= 1
                item_idx = [test[u][0]]

            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(all[u])

            if (target_item_n == -1):
              item_idx=item_idx+list(set(range(1,itemnum+1)).difference(rated))
            
            elif type(target_item_n)==int:
              for _ in range(target_item_n):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            
            elif type(target_item_n)==float:
              for _ in range(round(itemnum*target_item_n)):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            else:
              raise            
            
            inputs = {}
            inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
            inputs["input_seq"] = np.array([seq])
            inputs["candidate"] = np.array([item_idx])
            # print(inputs)

            # inverse to get descending sort
            predictions = -1.0 * sas_predict(model_,inputs, len(item_idx)-1)
            predictions = np.array(predictions)
            predictions = predictions[0]
            # print('predictions:', predictions)

            rank = predictions.argsort().argsort()[0]
            # print('rank:', rank)


            valid_user += 1

            if rank < rank_threshold:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

        return NDCG / valid_user, HT / valid_user

def sas_get_prediction(model_, dataset, user_map_dict,user_id_list, target_item_n=-1,top_n=10,exclude_purchased=True,is_test=False):
  all = dataset.User
  itemnum = dataset.itemnum
  users = [user_map_dict[u] for u in user_id_list]
  inv_user_map = {v: k for k, v in user_map_dict.items()}
  return_dict={}

  for u in tqdm(users):
    seq = np.zeros([model_.seq_max_len], dtype=np.int32)
    idx = model_.seq_max_len - 1

    list_to_seq = all[u] if not is_test else all[u][:-1]
    for i in reversed(list_to_seq):
      seq[idx] = i
      idx -= 1
      if idx == -1:
        break

    if exclude_purchased: 
      rated = set(all[u]) 
    else: 
      rated = set()
    
    # make empty candidate list
    item_idx=[]

    if (target_item_n == -1):
      item_idx=item_idx+list(set(range(1,itemnum+1)).difference(rated))
    
    elif type(target_item_n)==int:
      for _ in range(target_item_n):
        t = np.random.randint(1, itemnum + 1)
        while t in rated:
            t = np.random.randint(1, itemnum + 1)
        item_idx.append(t)
    
    elif type(target_item_n)==float:
      for _ in range(round(itemnum*target_item_n)):
        t = np.random.randint(1, itemnum + 1)
        while t in rated:
            t = np.random.randint(1, itemnum + 1)
        item_idx.append(t)

    else:
      raise
    
    inputs = {}
    inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
    inputs["input_seq"] = np.array([seq])
    inputs["candidate"] = np.array([item_idx])

    predictions = sas_predict(model_,inputs, len(item_idx)-1)
    predictions = np.array(predictions)
    predictions = predictions[0]

    pred_dict = {v : predictions[i] for i,v in enumerate(item_idx)}
    pred_dict = sorted(pred_dict.items(), key = lambda item: item[1], reverse = True)
    top_10_list = pred_dict[:10]

    return_dict[inv_user_map[u]] = top_10_list

  return return_dict

def sas_predict(model_, inputs,neg_cand_n):
    """Returns the logits for the test items.

    Args:
        inputs (tf.Tensor): Input tensor.

    Returns:
            tf.Tensor: Output tensor.
    """
    training = False
    input_seq = inputs["input_seq"]
    candidate = inputs["candidate"]

    mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
    seq_embeddings, positional_embeddings = model_.embedding(input_seq)
    seq_embeddings += positional_embeddings
    # seq_embeddings = model_.dropout_layer(seq_embeddings)
    seq_embeddings *= mask
    seq_attention = seq_embeddings
    seq_attention = model_.encoder(seq_attention, training, mask)
    seq_attention = model_.layer_normalization(seq_attention)  # (b, s, d)
    seq_emb = tf.reshape(
        seq_attention,
        [tf.shape(input_seq)[0] * model_.seq_max_len, model_.embedding_dim],
    )  # (b*s, d)
    candidate_emb = model_.item_embedding_layer(candidate)  # (b, s, d)
    candidate_emb = tf.transpose(candidate_emb, perm=[0, 2, 1])  # (b, d, s)

    test_logits = tf.matmul(seq_emb, candidate_emb)
    # (200, 100) * (1, 101, 100)'

    test_logits = tf.reshape(
        test_logits,
        [tf.shape(input_seq)[0], model_.seq_max_len, 1+neg_cand_n],
    )  # (1, 50, 1+neg_can)
    test_logits = test_logits[:, -1, :]  # (1, 101)
    return test_logits

