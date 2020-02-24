import gokart
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from novelty_enhanced_bpr.data.generate_psudo_data import GeneratePsudoData, GenerateItemEmbedVectors, \
    GenerateUserEmbedVectors


class VisualizePsudoData(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'
    rerun = True

    def requires(self):
        return dict(item_embed=GenerateItemEmbedVectors(), user_embed=GenerateUserEmbedVectors(), data=GeneratePsudoData())

    def run(self):
        item_embed = self.load('item_embed')
        user_embed = self.load('user_embed')
        clicks = self.load('data')['clicks_train']

        for i, c in enumerate(['r', 'g', 'b', 'c']):
            self.draw_scatter(item_embed[item_embed['item_type'] == i]['item_vector'].values, c)
        self.draw_scatter(user_embed['user_vector'].values, 'black', alpha=0.07, size=5)
        plt.savefig('resources/scatter.png')

        sum_clicks = clicks.groupby('item_id', as_index=False).agg({'click': 'sum'}).rename(columns={'click': 'sum_clicks'})
        clicks = pd.merge(clicks, sum_clicks, on='item_id', how='inner')
        average_clicks = clicks.groupby('item_type', as_index=False).agg({'sum_clicks': 'mean'}).rename(columns={'sum_clicks': 'average_clicks'})

        plt.figure()
        plt.bar(average_clicks['item_type'].values, average_clicks['average_clicks'].values)
        plt.savefig('resources/bar.png')

        self.dump('this is dummy output')

    @staticmethod
    def draw_scatter(embed, color, size=20, alpha=1.0):
        embed_matrix = np.stack(embed)
        x = embed_matrix[:, 0]
        y = embed_matrix[:, 1]
        plt.scatter(x, y, c=color, s=size, alpha=alpha)

