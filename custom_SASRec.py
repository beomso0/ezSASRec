import pickle 
from recommenders.models.sasrec.model import SASREC
from recommenders.utils.timer import Timer
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf

def train(model, dataset, sampler, **kwargs):
        """
        High level function for model training as well as
        evaluation on the validation and test dataset
        
        <kwargs>
        num_epochs | batch_size | learning_rate | val_epoch : 몇 epoch마다 validation 진행할지
        """
        num_epochs = kwargs.get("num_epochs", 10)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("learning_rate", 0.001)
        val_epoch = kwargs.get("val_epoch", 5)

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
                t_test = evaluate(model,dataset)
                print(
                    f"epoch: {epoch}, time: {T},  test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})"
                )

def evaluate(model_, dataset, target_user_n=1000, target_item_n=-1):
        """
        Evaluation on the test users (users with at least 3 items)

        <kwargs>
        model_ | dataset: SASRecDataSet 객체 | target_user_n: evaluate할 user 수 | target_item_n=
        """
        usernum = dataset.usernum
        itemnum = dataset.itemnum
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
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break
            rated = set(train[u])
            # print('rated', rated)
            rated.add(0)
            # print('rated2', rated)
            item_idx = [test[u][0]]
            # for _ in range(model_.num_neg_test):
            #     t = np.random.randint(1, itemnum + 1)
            #     while t in rated:
            #         t = np.random.randint(1, itemnum + 1)
            #     item_idx.append(t)
            '''
            item_num = list(range(1,11))
            rated = set([1,2,3,4])
            item_idx=[5]
            exclude_set = rated.union(set(item_idx))
            item_idx=item_idx+(list(set(item_num).difference(exclude_set)))
            '''
            exclude_set = rated.union(set(item_idx))
            item_idx=item_idx+list(set(range(1,itemnum+1)).difference(exclude_set))+[0 for _ in range(0,len(exclude_set)-2)]
            
            inputs = {}
            inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
            inputs["input_seq"] = np.array([seq])
            inputs["candidate"] = np.array([item_idx])
            # print(inputs)

            # inverse to get descending sort
            predictions = -1.0 * model_.predict(inputs)
            predictions = np.array(predictions)
            predictions = predictions[0]
            # print('predictions:', predictions)

            rank = predictions.argsort().argsort()[0]
            # print('rank:', rank)


            valid_user += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

        return NDCG / valid_user, HT / valid_user

def save_sasrec_model(model,path, exp_name='sas_experiment'):
  model.save_weights(path+exp_name+'_weights') # save trained weights
  arg_list = ['item_num','seq_max_len','num_blocks','embedding_dim','attention_dim','attention_num_heads','dropout_rate','conv_dims','l2_reg','num_neg_test']
  dict_to_save = {a: model.__dict__[a] for a in arg_list}
  with open(path+exp_name+'_model_args','wb') as f:
    pickle.dump(dict_to_save, f)

def load_sasrec_model(path, exp_name='sas_experiment'):
  with open(path+exp_name+'_model_args','rb') as f:
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
  model.load_weights(path+exp_name+'_weights')
  return model