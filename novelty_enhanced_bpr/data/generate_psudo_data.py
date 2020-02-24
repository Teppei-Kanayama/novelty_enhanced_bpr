from typing import List

import gokart
import luigi
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split


def cross_join(df1, df2):
    df1['key'] = 0
    df2['key'] = 0
    return df1.merge(df2, how='outer').drop('key', axis=1)


class GenerateItemEmbedVectors(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'
    mu_list: List = luigi.ListParameter()

    def run(self):
        item_embed_vector = []
        item_types = []
        for i, (x, y, n) in enumerate(self.mu_list):
            item_embed_vector.append(np.random.normal(loc=(x, y), scale=(1, 1), size=(n, 2)))
            item_types.extend([i] * n)
        item_embed_vector = np.concatenate(item_embed_vector)
        n_items = item_embed_vector.shape[0]
        item_ids = np.arange(n_items)
        df = pd.DataFrame(dict(item_id=item_ids, item_type=item_types, item_vector=list(item_embed_vector)))
        self.dump(df)


class GetItemDistance(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'
    item_embed_vector_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.item_embed_vector_task

    def run(self):
        item_embed_vector = self.load()
        item_embed_vector_x = item_embed_vector.rename(columns={'item_id': 'item_id_x', 'item_vector': 'item_vector_x'})
        item_embed_vector_y = item_embed_vector.rename(columns={'item_id': 'item_id_y', 'item_vector': 'item_vector_y'})
        item_distance_df = cross_join(item_embed_vector_x, item_embed_vector_y)

        def func(vector1, vector2):
            return np.linalg.norm(vector1 - vector2)

        item_distance_df['distance'] = item_distance_df.apply(lambda x: func(x['item_vector_x'], x['item_vector_y']), axis=1)
        self.dump(item_distance_df[['item_id_x', 'item_id_y', 'distance']])


class GenerateUserEmbedVectors(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'
    user_list: List = luigi.ListParameter()

    def run(self):
        x, y, n = self.user_list
        user_embed_vector = np.random.normal(loc=(x, y), scale=(1, 1), size=(n, 2))
        user_ids = np.arange(n)
        df = pd.DataFrame(dict(user_id=user_ids, user_vector=list(user_embed_vector)))
        self.dump(df)


class GenerateUserItemInteractions(gokart.TaskOnKart):
    item_embed_vector_task = gokart.TaskInstanceParameter()
    user_embed_vector_task = gokart.TaskInstanceParameter()

    def requires(self):
        return dict(item_embed_vector=self.item_embed_vector_task, user_embed_vector=self.user_embed_vector_task)

    def run(self):
        item_embed_vector = self.load('item_embed_vector')
        user_embed_vector = self.load('user_embed_vector')

        df = cross_join(item_embed_vector, user_embed_vector)

        def func(item_vector, user_vector):
            prob = expit(1 / np.linalg.norm(item_vector - user_vector)) - 0.5
            return np.random.binomial(1, prob, 1)[0]

        df['click'] = df.apply(lambda x: func(x['item_vector'], x['user_vector']), axis=1)
        self.dump(df[['item_id', 'item_type', 'user_id', 'click']])


class GeneratePsudoData(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'

    test_size: float = luigi.FloatParameter(default=0.3)
    validation_size: float = luigi.FloatParameter(default=0.1)

    def requires(self):
        item_embed_vector_task = GenerateItemEmbedVectors()
        user_embed_vector_task = GenerateUserEmbedVectors()
        user_item_iteraction_task = GenerateUserItemInteractions(item_embed_vector_task=item_embed_vector_task,
                                                                 user_embed_vector_task=user_embed_vector_task)
        item_distance_task = GetItemDistance(item_embed_vector_task=item_embed_vector_task)
        return dict(item_distance=item_distance_task, user_item_interaction=user_item_iteraction_task)

    def run(self):
        clicks = self.load('user_item_interaction')
        item_distance = self.load('item_distance')

        clicks_train, clicks_test = train_test_split(clicks, test_size=self.test_size)
        clicks_train, clicks_validation = train_test_split(clicks_train, test_size=self.validation_size / (1 - self.test_size))

        self.dump(dict(clicks_train=clicks_train, clicks_validation=clicks_validation, clicks_test=clicks_test, item_distance=item_distance))
