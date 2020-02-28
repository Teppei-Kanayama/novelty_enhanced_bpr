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

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        mapper = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        for i, c in enumerate(['r', 'g', 'b', 'c']):
            self.draw_scatter(ax, item_embed[item_embed['item_type'] == i]['item_vector'].values, c, f'item_type: {mapper[i]}')
        self.draw_scatter(ax, user_embed['user_vector'].values, 'black', 'user', alpha=0.07, size=3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper left')
        fig.savefig('resources/scatter.png')

        sum_clicks = clicks.groupby('item_id', as_index=False).agg({'click': 'sum'}).rename(columns={'click': 'sum_clicks'})
        clicks = pd.merge(clicks, sum_clicks, on='item_id', how='inner')
        average_clicks = clicks.groupby('item_type', as_index=False).agg({'sum_clicks': 'mean'}).rename(columns={'sum_clicks': 'average_clicks'})

        plt.figure()
        plt.bar(average_clicks['item_type'].values, average_clicks['average_clicks'].values, tick_label=['A', 'B', 'C', 'D'], align='center')
        plt.xlabel("item type")
        plt.ylabel("clicks")
        plt.savefig('resources/bar.png')

        self.dump('this is dummy output')

    @staticmethod
    def draw_scatter(ax, embed, color, label, size=20, alpha=1.0):
        embed_matrix = np.stack(embed)
        x = embed_matrix[:, 0]
        y = embed_matrix[:, 1]
        ax.scatter(x, y, c=color, s=size, alpha=alpha, label=label)

