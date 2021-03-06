import torch
import torch.nn as nn
from torch.nn import LogSigmoid
from torch.autograd import Variable


class MatrixFactorization(nn.Module):
    def __init__(self, n_items, n_users, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding_layer = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_dim)
        self.item_embedding_layer = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_dim)

    def forward(self, user, item):
        user = self.user_embedding_layer(user)
        item = self.item_embedding_layer(item)
        return (user * item).sum(axis=1)


def data_sampler(data, batch_size=2**11, iterations=1000000):
    for i in range(0, iterations):
        batch = data.sample(batch_size)
        yield dict(
            user_ids=Variable(torch.FloatTensor(batch['user_id'].values)).long(),
            positive_item_ids=Variable(torch.FloatTensor(batch['positive_item_id'].values)).long(),
            negative_item_ids=Variable(torch.FloatTensor(batch['negative_item_id'].values)).long()
            )


def bpr_loss(x):
    return -LogSigmoid()(x[0] - x[1]).mean()
