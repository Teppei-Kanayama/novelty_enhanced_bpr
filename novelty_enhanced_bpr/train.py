import torch
import gokart
import luigi

from novelty_enhanced_bpr.data.generate_psudo_data import GeneratePsudoData
from novelty_enhanced_bpr.data.make_paired_data import MakePairedData
from novelty_enhanced_bpr.model.matrix_factorization import MatrixFactorization, bpr_loss, data_sampler


class TrainModel(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'

    distance_threshold: float = luigi.FloatParameter(default=1.0)
    novelty_weight: float = luigi.FloatParameter(default=0.0)
    embedding_dim: int = luigi.IntParameter(default=10)
    lr: float = luigi.FloatParameter(default=0.005)
    weight_decay: float = luigi.FloatParameter(default=0.0001)
    max_iter: int = luigi.IntParameter(default=100)

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
        data = self.load('data')['clicks']
        paired_data = self.load_data_frame('paired_data', required_columns={'user_id', 'positive_item_id', 'negative_item_id'})
        novelty_enhanced_paired_data = self.load_data_frame('novelty_enhanced_paired_data', required_columns={'user_id', 'positive_item_id', 'negative_item_id'})

        n_users = data['user_id'].max() + 1
        n_items = data['item_id'].max() + 1

        model = MatrixFactorization(n_items=n_items, n_users=n_users, embedding_dim=self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sampled_data = data_sampler(paired_data)

        training_losses = []
        for iterations, d in enumerate(sampled_data):
            predict = [model(item=d['positive_item_ids'], user=d['user_ids']),
                       model(item=d['negative_item_ids'], user=d['user_ids'])]
            loss = bpr_loss(predict)
            training_losses.append(float(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations > self.max_iter:
                self.dump(model)
                break
