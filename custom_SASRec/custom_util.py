import pickle 
from recommenders.models.sasrec.model import SASREC

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