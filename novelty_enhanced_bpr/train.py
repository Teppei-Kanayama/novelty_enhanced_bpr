import numpy as np
import torch
import gokart
import luigi

from novelty_enhanced_bpr.data.generate_psudo_data import GeneratePsudoData


class TrainModel(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'
    # embedding_dim = luigi.IntParameter(default=10)  # type: int
    # lr = luigi.FloatParameter(default=0.005)  # type: float
    # weight_decay = luigi.FloatParameter(default=0.0001)  # type: float
    # alpha = luigi.FloatParameter(default=0.5)  # type: float
    # loss_type = luigi.Parameter(default='view_enhanced_bpr')  # type: str

    def requires(self):
        return dict(data=GeneratePsudoData())

    def output(self):
        return self.make_model_target(relative_file_path='model/mf.zip',
                                      save_function=torch.save,
                                      load_function=torch.load)

    def run(self):
        data = self.load('data')
