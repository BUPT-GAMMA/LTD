from __future__ import division
from __future__ import print_function

import time
import numpy as np
from numpy.lib.function_base import append
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from data.utils import matrix_pow, row_normalize
from models.GCN import GCN
from models.GAT import GAT

from models.MLP import MLP
from models.SGConv import SGConv

from utils.metrics import accuracy


def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str,
                        default='GCN', help='Teacher Model')
    parser.add_argument('--assistant', type=int, default=-1,
                        help='Different assistant teacher. -1. None 0. nasty 1. reborn')
    parser.add_argument('--student', type=str,
                        default='GCN', help='Student Model')
    parser.add_argument('--device', type=int, default=6, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=20, help='label rate')

    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='Weight decay')                  
    parser.add_argument('--grad', type=int, default=0,
                        help='Output Feature grad')

    return parser.parse_args()


def choose_model(conf, G, features, labels, byte_idx_train, labels_one_hot):
    if conf['model_name'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=conf['hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=1,
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])
    elif conf['model_name'] == 'GAT':
        num_heads = 8
        num_layers = 1
        num_out_heads = 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=G.ndata['feat'].shape[1],
                    num_hidden=8,
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,    
                    residual=False).to(conf['device'])
    elif conf['model_name'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=G.ndata['feat'].shape[1],
                          n_hidden=16,
                          n_classes=labels.max().item() + 1,
                          n_layers=1,
                          activation=F.relu,
                          dropout=0.5,
                          aggregator_type=conf['agg_type']).to(conf['device'])
    elif conf['model_name'] == 'APPNP':
        model = APPNP(g=G,
                      in_feats=G.ndata['feat'].shape[1],
                      hiddens=[64],
                      n_classes=labels.max().item() + 1,
                      activation=F.relu,
                      feat_drop=0.5,
                      edge_drop=0.5,
                      alpha=0.1,
                      k=10).to(conf['device'])
    elif conf['model_name'] == 'MLP':
        model = MLP(num_layers=2,
                    input_dim=G.ndata['feat'].shape[1],
                    hidden_dim=conf['hidden'],
                    output_dim=labels.max().item() + 1,
                    dropout=conf['dropout']).to(conf['device'])
    elif conf['model_name'] == 'SGC':
        model = SGConv(in_feats=features.shape[1],
                       out_feats=labels.max().item() + 1,
                       k=2,
                       cached=True,
                       bias=False).to(conf['device'])
    else:
        raise ValueError(f'Undefined Model.')
    return model


def distill_train(all_logits, dur, epoch, model, optimizer, conf, G, labels_init, labels_one_hot, labels, idx_no_train, idx_train,
                  idx_val, idx_test,my_val, cas, teacher_logits, f, adj, nei_num, 
                  nei_entropy, t_model, nei_probability, same_pro,  features, last_temp):
    t0 = time.time()
    model.train()
    if conf['model_name'] in ['GCN', 'LogReg', 'MLP']:
        logits, fs = model(G.ndata['feat'])
    if conf['model_name'] in ['APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        logits, _, fs = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])

    idx_all = torch.cat([idx_train,idx_no_train],dim=0)
    idx_all,_ = torch.sort(idx_all)
    logp = F.log_softmax(logits, dim=1) #t=1
    student_hard = F.softmax(logits, dim=1)
    with torch.no_grad():
        f_logits = torch.norm(logits, 2, dim=1)
    
    extract_x = torch.cat((logits, torch.unsqueeze(f_logits, dim=1), torch.unsqueeze(nei_entropy, dim=1)), 1) # 
    acc_train = accuracy(logp[idx_train], labels[idx_train])

    my_lr = conf['my_lr']
    my_t_lr = conf['my_t_lr']
    lam = conf['lam']
    model_dict = {}
    t_model_dict = {}
    k = conf['k']
    temparature = t_model(extract_x)
    temparature = (temparature-0.2)*k
    temparature = torch.where(abs(temparature)<0.0001, torch.full_like(temparature,0.0001), temparature)
    
    teacher_logits_t = torch.div(teacher_logits[-1].t(), torch.squeeze(temparature,dim=1)).t()
    teacher_softmax = F.softmax(teacher_logits_t, dim=1)
    hard_loss = -torch.sum((labels_one_hot[idx_train]+1e-6)*torch.log(student_hard[idx_train]+1e-6))
    soft_loss = -torch.sum((teacher_softmax[idx_all]+1e-6)*torch.log(student_hard[idx_all]+1e-6))
    g_loss = soft_loss + lam* hard_loss#10
        
    for p_name, p in model.named_parameters():
        agr = torch.autograd.grad(g_loss, p, create_graph=True)[0]
        model_dict[p_name] = p - my_lr * agr#update model
    model.load_state_dict(model_dict)
    t_loss = torch.tensor(0)
    if epoch>20 :
        if conf['model_name'] in ['GCN', 'LogReg', 'MLP']:
            logits, fs = model(G.ndata['feat'],model_dict)
        if conf['model_name'] in ['APPNP']:
            logits = model(G.ndata['feat'],model_dict)
        elif conf['model_name'] == 'GAT':
            logits, _, fs = model(G.ndata['feat'], model_dict)
        elif conf['model_name'] in ['GraphSAGE', 'SGC']:
            logits = model(G, G.ndata['feat'], model_dict)  
        student_hard = F.softmax(logits, dim=1)
        t_loss = -torch.sum((labels_one_hot[idx_val]+1e-6)*torch.log(student_hard[idx_val]+1e-6))
        for pvt_name, pt in t_model.named_parameters():
            pt.grad = torch.autograd.grad(t_loss, pt,create_graph=True)[0]
            t_model_dict[pvt_name] = pt - my_t_lr * pt.grad 

        t_model.load_state_dict(t_model_dict) 
        zero_grad(t_model)
  
    dur.append(time.time() - t0)
    model.eval()
    if conf['model_name'] in ['GCN', 'LogReg', 'MLP']:
        logits,_ = model(G.ndata['feat'])
    elif conf['model_name'] in ['APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        logits = model(G.ndata['feat'])[0]
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    logp = F.log_softmax(logits, dim=1)
 
    acc_train_2 = accuracy(logp[idx_val], labels[idx_val])
    acc_val = accuracy(logp[my_val], labels[my_val])
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    
    print('Epoch %d | Loss: %.4f | acc_train2: %.4f | acc_train: %.4f | acc_val: %.4f | acc_test: %.4f | Time(s) %.4f' % (
        epoch, g_loss.item(), acc_train_2.item(), acc_train.item(), acc_val.item(), acc_test.item(), dur[-1]))
    return acc_val, t_loss, temparature


def zero_grad(my_model):#grad->0
    with torch.no_grad():
        for p in my_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

def model_train(conf, model, optimizer, G, labels_init, labels_one_hot,
                labels, idx_no_train, idx_train, idx_val, idx_test, idx_my_val, cas, teacher_logits, f, features):
    all_logits = []
    dur = []
    best = 0
    cnt = 0
    epoch = 1
    adj = torch.tensor(G.adj(scipy_fmt='coo').todense()).to(conf['device'])
    nei_num = torch.empty(len(adj)).to(conf['device'])
    nei_num_selfloop = torch.empty(len(adj)).to(conf['device'])
    for i in range(len(adj)):
        nei_num[i] = len(adj[i].nonzero()) - 1
        nei_num_selfloop[i] = len(adj[i].nonzero())
    adj_selfloop = adj #remove self loop
    for i in range(len(adj[-1])):
        adj[i][i] = 0
    idx_all = torch.cat([idx_train,idx_no_train],dim=0)
    idx_all,_ = torch.sort(idx_all)
    
    teacher_softmax = F.softmax(teacher_logits[-1], dim=1)
    nei_logits_sum = torch.mm(adj.float(), teacher_softmax)

    nei_probability = torch.div(nei_logits_sum.t(), nei_num).t()
    nei_entropy = -torch.sum(nei_probability[idx_all]*torch.log(nei_probability[idx_all]), dim=1)
    for i in range(len(nei_entropy)):
        if nei_entropy[i] != nei_entropy[i]:
            nei_entropy[i] = 0.0001
    t_lab = torch.argmax(teacher_softmax, dim=1)
    same_pro = torch.randn(len(nei_entropy))
    for i in range(len(nei_probability)):
        same_pro[i] = nei_probability[i][t_lab[i]]

    t_model = MLP(num_layers = 2, input_dim =labels.max().item()+3, hidden_dim = 64, output_dim = 1, dropout=0.6).to(conf['device'])
    temp = 0
    while epoch < conf['max_epoch']:
        acc_val, loss_val, temp = distill_train(all_logits, dur, epoch, model, optimizer, conf, G, labels_init, labels_one_hot, labels,
                                          idx_no_train, idx_train, idx_val, idx_test, idx_my_val, cas, teacher_logits, f, adj_selfloop, 
                                          nei_num_selfloop, nei_entropy, t_model, nei_probability, same_pro, features, temp)
        epoch += 1
        if epoch > 0: 
            if acc_val >= best:
                best = acc_val
                state = dict([('model', copy.deepcopy(model.state_dict()))])
                best_epoch = epoch
                cnt = 0
            else:
                cnt += 1
            if cnt == conf['patience'] or epoch == conf['max_epoch'] or loss_val != loss_val:
                print("Stop!!!")
                print(best_epoch)
                break
    model.load_state_dict(state['model'])
    print("Optimization Finished!")

    return best


def distill_test(conf, model, G, labels_init, labels, idx_test, cas, f, features, adj):
    model.eval()
    if conf['model_name'] in ['GCN', 'LogReg', 'MLP']:
        logits,_ = model(G.ndata['feat'])
    elif conf['model_name'] in ['APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        logits, G.edata['a'], gat_f = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    
    logp = F.log_softmax(logits, dim=1)
    loss_test = F.nll_loss(logp[idx_test], labels[idx_test])
    preds = torch.argmax(logp, dim=1).cpu().detach()
    teacher_preds = torch.argmax(cas[-1], dim=1).cpu().detach()
    
    acc_test = accuracy(logp[idx_test], labels[idx_test], w=True, te=cas[-1][idx_test], g=G, idx=idx_test,student=logp, real=labels,tea=cas[-1])
    acc_teacher_test = accuracy(cas[-1][idx_test], labels[idx_test])
    same_predict = np.count_nonzero(
        teacher_preds[idx_test] == preds[idx_test]) / len(idx_test)
    acc_dis = np.abs(acc_teacher_test.item() - acc_test.item())
    print("Test set results: loss= {:.4f} acc_test= {:.4f} acc_teacher_test= {:.4f} acc_dis={:.4f} same_predict= {:.4f}".format(
        loss_test.item(), acc_test.item(), acc_teacher_test.item(), acc_dis, same_predict))

    return acc_test, logp, same_predict


def save_output(output_dir, preds, labels, output, acc_test, same_predict, G, idx_train, adj, conf):
    np.savetxt(output_dir.joinpath('preds.txt'),
               preds, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('labels.txt'),
               labels, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('output.txt'),
               output, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('test_acc.txt'),
               np.array([acc_test]), fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('same_predict.txt'),
               np.array([same_predict]), fmt='%.4f', delimiter='\t')

    if conf['grad'] == 1:
        grad = G.ndata['feat'].grad.cpu().numpy()
        np.savetxt(output_dir.joinpath('grad.txt'),
                   grad, fmt='%.4f', delimiter='\t')
