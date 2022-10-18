
import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json

class Pooling(nn.Module):

    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode: str = None,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
                            'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls']
            pooling_mode_cls_token = (pooling_mode == 'cls')
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens,
                                       pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        modes = []
        if self.pooling_mode_cls_token:
            modes.append('cls')
        if self.pooling_mode_mean_tokens:
            modes.append('mean')
        if self.pooling_mode_max_tokens:
            modes.append('max')
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append('mean_sqrt_len_tokens')

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        # Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)


def get_feature(token_embeddings, cls_tokens, attention_mask):
    features = {}
    features.update({'token_embeddings': token_embeddings, 'cls_token_embeddings': cls_tokens,
                     'attention_mask': attention_mask})
    return features
