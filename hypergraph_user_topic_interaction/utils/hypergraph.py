import numpy as np
import scipy.sparse as sp
import torch
import pickle
import gensim.models import AuthorTopicModel
from gensim.corpora import mmcorpus
from gensim.test.utils import common_dictionary, datapath, temporary_file

class Hypergraph:
    def __init__(self, opt):
        self.dataset = opt.dataset


    def get_hyperedges(self):
        dirname = "data/"
        if self.dataset == "politifact":
            filename = "hypergraph_politifact.pkl"
        elif self.dataset == "gossipcop":
            filename = "hypergraph_gossipcop.pkl"

        with open(dirname + filename, 'rb') as handle:
            result = pickle.load(handle)

        return result



    def get_adj_matrix(self, hyperedges, nodes_seq):
        items, n_node, HT, alias_inputs, node_masks, node_dic = [], [], [], [], [], []

        node_list = nodes_seq
        node_set = list(set(node_list))
        node_dic = {node_set[i]: i for i in range(len(node_set))}

        rows = []
        cols = []
        vals = []
        max_n_node = len(node_set)
        max_n_edge = len(hyperedges)
        total_num_node = len(node_set)


        num_hypergraphs = 1
        for idx in range(num_hypergraphs):
            for hyperedge_seq, hyperedge in enumerate(hyperedges):
                for node_id in hyperedge:
                    rows.append(node_dic[node_id])
                    cols.append(hyperedge_seq)
                    vals.append(1)
            u_H = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
            HT.append(np.asarray(u_H.T.todense()))
            alias_inputs.append([j for j in range(max_n_node)])
            node_masks.append([1 for j in range(total_num_node)] + (max_n_node - total_num_node) * [0])

        return alias_inputs, HT, node_masks

    def get_at_matrix(self, result, nodes_seq):

        corpus = mmcorpus.MmCorpus(datapath('author-topic.mm'))

        with temporary_file("serialized") as s_path:
            model = AuthorTopicModel(
                corpus, author2doc=author2doc, id2word=common_dictionary, num_topics=4,
                serialized=True, serialization_path=s_path
            )
            model.update(corpus, author2doc)
            HTH_inputs = [model.get_author_topics(author) for author in model.id2author.values()]

        return HTH_inputs