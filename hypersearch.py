import optuna
import torch
from torch.nn.modules.container import ModuleList
import torch.optim as optim
import dgl
import numpy as np
import scipy
import psutil
import os

from distill_dgl import model_train, choose_model, distill_test
from data.get_cascades import load_cascades
from data.get_cascades import load_logits
from data.utils import load_tensor_data, initialize_label, set_random_seed, choose_path


class AutoML(object):
    """
    Args:
        func_search: function to obtain hyper-parameters to search
    """

    def __init__(self, kwargs, func_search):
        self.default_params = kwargs
        self.dataset = kwargs['dataset']
        self.seed = kwargs['seed']
        self.func_search = func_search
        self.n_trials = kwargs['ntrials']
        self.n_jobs = kwargs['njobs']
        # self.model = None
        self.best_results = None
        self.preds = None
        self.labels = None
        self.output = None
        self.mylr = None
        self.tlr = None
        self.t = None
        self.l = None

    def _objective(self, trials):
        params = self.default_params
        params.update(self.func_search(trials))
        lr = params['my_lr']
        tr = params['my_t_lr']
        results, self.preds, self.labels, self.output = raw_experiment(params)
        if self.best_results is None or results['ValAcc'] > self.best_results['ValAcc']:
            self.best_results = results
            self.mylr = lr
            self.tlr = tr
            self.t = params['tem']
            self.l = params['lam']
        return results['ValAcc']

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials,
                       n_jobs=self.n_jobs)
        return self.best_results, self.preds, self.labels, self.output, self.mylr, self.tlr, self.t, self.l


def raw_experiment(configs):
    print('*************************************')
    print(psutil.Process(os.getpid()).memory_info().rss)
    output_dir, cascade_dir = choose_path(configs)
    # random seed
    set_random_seed(configs['seed'])
    # load_data
    adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test, idx_my_val = \
        load_tensor_data(configs['model_name'], configs['dataset'],
                         configs['labelrate'], configs['device'])
    # scipy.sparse.save_npz('cora.npz', scipy.sparse.csr_matrix(adj.cpu()))
    labels_init = initialize_label(
        idx_train, labels_one_hot).to(configs['device'])
    idx_no_train = torch.LongTensor(
        np.setdiff1d(np.array(range(len(labels))), idx_train.cpu())).to(configs['device'])
    print(idx_no_train)
    byte_idx_train = torch.zeros_like(
        labels_one_hot, dtype=torch.bool).to(configs['device'])
    byte_idx_train[idx_train] = True
    G = dgl.graph((adj_sp.row, adj_sp.col)).to(configs['device'])
    G.ndata['feat'] = features
    G.ndata['feat'].requires_grad_()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    print('Loading cascades...')
    cas = load_cascades(cascade_dir, configs['device'], final=True)
    teacher_logits, f = load_logits(cascade_dir, configs['device'], final=True)

    model = choose_model(
        configs, G, G.ndata['feat'], labels, byte_idx_train, labels_one_hot)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs['lr'],
                           weight_decay=configs['wd'])
    acc_val = model_train(configs, model, optimizer, G, labels_init, labels_one_hot,
                          labels, idx_no_train, idx_train, idx_val, idx_test, idx_my_val, cas, teacher_logits, f, G.ndata['feat'])
    acc_test, logp, same_predict = distill_test(
        configs, model, G, labels_init, labels, idx_test, cas, f, G.ndata['feat'], adj)
    preds = logp.max(1)[1].type_as(labels).cpu().numpy()
    labels = labels.cpu().numpy()
    output = np.exp(logp.cpu().detach().numpy())
    results = dict(TestAcc=acc_test.item(),
                   ValAcc=acc_val.item(), SamePredict=same_predict)
    
    del model

    return results, preds, labels, output

    
