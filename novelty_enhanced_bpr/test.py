import torch
from torch.autograd import Variable

import gokart

from novelty_enhanced_bpr import TrainModel
from novelty_enhanced_bpr.data.generate_psudo_data import GeneratePsudoData
from novelty_enhanced_bpr.evaluation.metrics import recall_at_k, map_at_k


class TestModel(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'

    def requires(self):
        return dict(model=TrainModel(), data=GeneratePsudoData())

    def run(self):
        model = self.load('model')
        data = self.load('data')['clicks_test']
        for k in [5, 10]:
            recall, map = self._run(model, data, k)
            print(k, recall, map)
        import pdb; pdb.set_trace()

    @staticmethod
    def _run(model, data, k):
        user_tensor = Variable(torch.FloatTensor(data['user_id'].values)).long()
        item_tensor = Variable(torch.FloatTensor(data['item_id'].values)).long()
        scores = model(item=item_tensor, user=user_tensor)
        data['model_score'] = scores.data.numpy()

        data['rank'] = data.groupby('user_id')['model_score'].rank(ascending=False)
        print(data[data['rank'] <= k].groupby('item_type').agg({'user_id': 'nunique'}))
        print(data[data['rank'] <= k].groupby('user_id', as_index=False).agg({'item_type': 'nunique'}).groupby('item_type').agg({'user_id': 'nunique'}))
        return recall_at_k(data, k), map_at_k(data, k)
