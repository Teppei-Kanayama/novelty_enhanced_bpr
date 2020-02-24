import gokart
import luigi
import numpy as np
import torch
from torch.autograd import Variable

from novelty_enhanced_bpr.data.generate_psudo_data import GeneratePsudoData
from novelty_enhanced_bpr.data.make_paired_data import MakePairedData
from novelty_enhanced_bpr.evaluation.metrics import recall_at_k, map_at_k
from novelty_enhanced_bpr.model.matrix_factorization import MatrixFactorization, bpr_loss, data_sampler


class TrainModel(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'

    distance_threshold: float = luigi.FloatParameter(default=None)
    novelty_rate: float = luigi.FloatParameter(default=0.0)
    embedding_dim: int = luigi.IntParameter()
    lr: float = luigi.FloatParameter()
    weight_decay: float = luigi.FloatParameter()
    max_iter: int = luigi.IntParameter()
    n_users: int = luigi.IntParameter()
    n_items: int = luigi.IntParameter()

    def requires(self):
        psudo_data_task = GeneratePsudoData()
        paired_data_task = MakePairedData(click_task=psudo_data_task, distance_threshold=None)
        novelty_enhanced_paired_data_task = MakePairedData(click_task=psudo_data_task, distance_threshold=self.distance_threshold)
        return dict(data=psudo_data_task, paired_data=paired_data_task, novelty_enhanced_paired_data=novelty_enhanced_paired_data_task)

    def output(self):
        return self.make_model_target(relative_file_path='model/mf.zip',
                                      save_function=torch.save,
                                      load_function=torch.load)

    def run(self):
        paired_data = self.load_data_frame('paired_data', required_columns={'user_id', 'positive_item_id', 'negative_item_id'})
        novelty_enhanced_paired_data = self.load_data_frame('novelty_enhanced_paired_data', required_columns={'user_id', 'positive_item_id', 'negative_item_id'})
        validation_data = self.load('data')['clicks_validation']

        model = MatrixFactorization(n_items=self.n_items, n_users=self.n_users, embedding_dim=self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sampled_paried_data = data_sampler(paired_data)
        sampled_novelty_enhanced_paired_data = data_sampler(novelty_enhanced_paired_data)

        training_losses = []
        for iterations, (d1, d2) in enumerate(zip(sampled_paried_data, sampled_novelty_enhanced_paired_data)):
            d = d2 if np.random.binomial(1, self.novelty_rate, 1)[0] else d1
            predict = [model(item=d['positive_item_ids'], user=d['user_ids']),
                       model(item=d['negative_item_ids'], user=d['user_ids'])]
            loss = bpr_loss(predict)
            training_losses.append(float(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations % 100 == 0:
                validation_score = validate(model, validation_data)
                print(f'iteration: {iterations}, '
                      f'train loss: {np.array(training_losses).mean()}, '
                      f'val recall@5: {validation_score["recall"]}, '
                      f'val map@5: {validation_score["map"]}')

            if iterations > self.max_iter:
                self.dump(model)
                break


def validate(model, data):
    user_tensor = Variable(torch.FloatTensor(data['user_id'].values)).long()
    item_tensor = Variable(torch.FloatTensor(data['item_id'].values)).long()
    scores = model(item=item_tensor, user=user_tensor)
    data['model_score'] = scores.data.numpy()

    data['rank'] = data.groupby('user_id')['model_score'].rank(ascending=False)
    print(data[data['rank'] <= 5].groupby('item_type').agg({'user_id': 'nunique'}))
    return dict(recall=recall_at_k(data, k=5), map=map_at_k(data, k=5))
