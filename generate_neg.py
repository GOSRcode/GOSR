import pandas as pd
import pickle
from utils import myFloder, pickle_loader, collate, trans_to_cuda, eval_metric, collate_test, user_neg


dataset = 'Beauty'
data = pd.read_csv('./Data/' + dataset + '.csv')
user = data['user_id'].unique()
item = data['item_id'].unique()
user_num = len(user)
item_num = len(item)

data_neg = user_neg(data, item_num)
f = open(dataset+'_neg', 'wb')
pickle.dump(data_neg,f)
f.close()

