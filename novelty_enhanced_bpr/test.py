import torch
from torch.autograd import Variable
import pandas as pd
import gokart
import luigi
import matplotlib.pyplot as plt

from novelty_enhanced_bpr import TrainModel
from novelty_enhanced_bpr.data.generate_psudo_data import GeneratePsudoData
from novelty_enhanced_bpr.evaluation.metrics import recall_at_k, map_at_k


class TestModel(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'
    k_list = luigi.ListParameter(default=[5, 10])
    rerun = True

    def output(self):
        return self.make_target('scores.csv')

    def requires(self):
        return dict(model=TrainModel(), data=GeneratePsudoData())

    def run(self):
        model = self.load('model')
        model_name = TrainModel().param_kwargs['model_name']
        data = self.load('data')['clicks_test']
        output = self._run(model, data, model_name, self.k_list)
        self.dump(output)

    @classmethod
    def _run(cls, model, data, model_name, k_list):
        user_tensor = Variable(torch.FloatTensor(data['user_id'].values)).long()
        item_tensor = Variable(torch.FloatTensor(data['item_id'].values)).long()
        scores = model(item=item_tensor, user=user_tensor)
        data['model_score'] = scores.data.numpy()
        data['rank'] = data.groupby('user_id')['model_score'].rank(ascending=False)
        cls._group_by_item_type(data, model_name)
        cls._group_by_user(data, model_name)
        return cls._calculate_metrics(data, model_name, k_list)

    @staticmethod
    def _calculate_metrics(data, model_name, k_list):
        recall_list = []
        map_list = []
        for k in k_list:
            recall_list.append(recall_at_k(data, k))
            map_list.append(map_at_k(data, k))
        evalueation_df = pd.DataFrame(dict(model=model_name, k=k_list, recall=recall_list, map=map_list))
        return evalueation_df

    @staticmethod
    def _group_by_item_type(data, model_name, k=10):
        agg_data = data[data['rank'] <= k].groupby('item_type', as_index=False).agg({'user_id': 'nunique'})
        plt.figure()
        plt.bar(agg_data['item_type'].values, agg_data['user_id'].values, tick_label=['A', 'B', 'C', 'D'],
                align='center')
        plt.xlabel("item type", fontsize=16)
        plt.ylabel("# of users", fontsize=16)
        plt.tick_params(labelsize=10)
        plt.savefig(f'resources/item_types_{model_name}.png')

    @staticmethod
    def _group_by_user(data, model_name, k=10):
        agg_data = data[data['rank'] <= k].groupby('user_id', as_index=False).agg({'item_type': 'nunique'}).groupby('item_type', as_index=False).agg({'user_id': 'nunique'})
        plt.figure()
        plt.bar(agg_data['item_type'].values, agg_data['user_id'].values, tick_label=[1, 2, 3, 4],
                align='center')
        plt.xlabel("# of item types", fontsize=16)
        plt.ylabel("# of users", fontsize=16)
        plt.tick_params(labelsize=10)
        plt.savefig(f'resources/users_{model_name}.png')
