import gokart
import numpy as np
import matplotlib.pyplot as plt

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
        self.draw_scatter(user_embed['user_vector'].values, 'black', size=5)
        plt.savefig('resources/scatter.png')

        ctr = clicks.groupby('item_type', as_index=False).agg({'click': 'mean'}).rename(columns={'click': 'CTR'})
        plt.figure()
        plt.bar(ctr['item_type'].values, ctr['CTR'].values)
        plt.savefig('resources/bar.png')

        self.dump('this is dummy output')

    @staticmethod
    def draw_scatter(embed, color, size=20):
        embed_matrix = np.stack(embed)
        x = embed_matrix[:, 0]
        y = embed_matrix[:, 1]
        plt.scatter(x, y, c=color, s=size)
