# coding=utf-8
import os
import torch.backends.cudnn as cudnn
from GT_drug import GT
import pandas as pd
from encode_GT import GCNModelVAE, InnerProductDecoder
from preprocess_smote import *
# from preprocess_pearson import *
# from preprocess_spearman import *
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import random
import torch.utils.data
import argparse
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix  


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--cuda', type=str, default="0")
parser.add_argument('--n_epoch', type=int, default=300)
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--gfeat', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--nfeat', type=float, default=2000,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--classes', type=int, default=2,
                    help='classes number')
parser.add_argument('--model_path', type=str, default='models')
parser.add_argument('--lambda_d', type=float, default=1.0,
                    help='hyperparameter for domain loss')
parser.add_argument('--lambda_r', type=float, default=0.3,
                    help='hyperparameter for reconstruction loss')
parser.add_argument('--lambda_f', type=float, default=0.001,
                    help='hyperparameter for different loss')
parser.add_argument('--early-stopping', type=bool, default=False, help='Enable early stopping')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
parser.add_argument('--drug_name','-d',help='drug name')


# drug_list = ['Gefitinib', 'Afatinib', 'AR-42', 'Cetuximab', 'Etoposide', 'NVP-TAE684', 'PLX4720', 'PLX4720_451Lu', 'Sorafenib', 'Vorinostat']
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
DRUG = args.drug_name
cuda = True
cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
manual_seed = random.randint(1, 10000)
print("manual_seed:", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# # test k
# k_values = [3, 5, 10, 15]
# for k in k_values:
#     print(f"Running model with k = {k}")
#
#     
#     adj_s, features_s, labels_s, knn_s, n_features_s = load_data_drug('source', DRUG, k)
#     adj_t, features_t, labels_t, knn_t, n_features_t = load_data_drug('target', DRUG, k)

''' Load data '''
adj_s, features_s, labels_s, knn_s, n_features_s = load_data_drug('source', DRUG)
adj_t, features_t, labels_t, knn_t, n_features_t= load_data_drug('target', DRUG)

# 0: source, 1: target
domain_labels_s = torch.zeros(features_s.shape[0], dtype=torch.long).to(device)
domain_labels_t = torch.ones(features_t.shape[0], dtype=torch.long).to(device)


adj_s = adj_s.to(device)
features_s = features_s.to(device)
labels_s = labels_s.to(device)

adj_t = adj_t.to(device)
features_t = features_t.to(device)
labels_t = labels_t.to(device)


adj_s.ndata['feat'] = features_s
adj_s.ndata['label'] = labels_s
adj_t.ndata['feat'] = features_t
adj_t.ndata['label'] = labels_t


args.nfeat = n_features_s

''' Load adj labels for reconstruction '''

adj_label_s, pos_weight_s, norm_s = load_adj_label_drug(knn_s)
adj_label_t, pos_weight_t, norm_t = load_adj_label_drug(knn_t)


def predict(adj,feature):  # # def predict(feature, adj, ppmi):
    _, basic_encoded_output, _ = shared_encoder(adj,feature)
    logits = cls_model(basic_encoded_output)
    if torch.isnan(logits).any():
        print("NaN detected in logits")
    return logits


def evaluate(preds, labels):
    accuracy1 = accuracy(preds, labels)
    return accuracy1


def test(adj,feature, label):  # def test(feature, adj, ppmi, label):
    for model in models:
        model.eval()
    logits = predict(adj,feature)   # logits = predict(feature, adj, ppmi)
    labels = label
    accuracy = evaluate(logits, labels)
    logits_proba = F.softmax(logits, dim=1).cpu().detach().numpy()
    preds = logits_proba.argmax(axis=1)
    labels_np = labels.cpu().detach().numpy()



    cm = confusion_matrix(labels_np, preds)
    tn, fp, fn, tp = cm.ravel()

    # # PRINT RESULT
    # print(f"Confusion Matrix:\n{cm}")
    # print(f"True Positives (TP): {tp}")
    # print(f"False Positives (FP): {fp}")
    # print(f"True Negatives (TN): {tn}")
    # print(f"False Negatives (FN): {fn}")

    # AUC
    auc = roc_auc_score(labels_np, logits_proba[:, 1])

    # AUPR
    aupr = average_precision_score(labels_np, logits_proba[:, 1])
    return accuracy, auc, aupr


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        self.loss = diff_loss
        return self.loss


def recon_loss(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    if cost + KLD < 0 :
        print("recon_loss is negative！！！！")
    if torch.isnan(cost).any():
        print("NaN detected in cost ")
    if torch.isnan(KLD).any():
        print("NaN detected in KLD")
    return cost + KLD


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")

''' set loss function '''
loss_diff = DiffLoss()
cls_loss = nn.CrossEntropyLoss().to(device)
domain_loss = torch.nn.NLLLoss()


''' load model '''
''' private encoder/encoder for S/T'''

private_encoder_s = GCNModelVAE(input_feat_dim=args.nfeat, hidden_dim1=args.hidden, hidden_dim2=args.gfeat,
                                  dropout=args.dropout).to(device)
private_encoder_t = GCNModelVAE(input_feat_dim=args.nfeat, hidden_dim1=args.hidden, hidden_dim2=args.gfeat,
                                  dropout=args.dropout).to(device)

''' decoder'''
decoder_s = InnerProductDecoder(dropout=args.dropout, act=lambda x: x)
decoder_t = InnerProductDecoder(dropout=args.dropout, act=lambda x: x)

''' shared encoder '''
shared_encoder = GT(nfeat=args.nfeat, nhid=args.hidden, nclass=args.gfeat, dropout=args.dropout).to(device)



''' node classifier model '''

cls_model = nn.Sequential(
    nn.Linear(args.gfeat, 2),
).to(device)

''' domain discriminator model '''
domain_model = nn.Sequential(
    GRL(),
    nn.Linear(args.gfeat, 10),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(10, 2),
).to(device)


''' the set of models used in DAGFormer '''
models = [private_encoder_s, private_encoder_t, shared_encoder, cls_model, domain_model, decoder_s, decoder_t]
params = itertools.chain(*[model.parameters() for model in models])

''' setup optimizer '''
optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)


def save_model(model, filename):
    torch.save(model.state_dict(), filename)

''' training '''
best_acc_s = 0
best_auc_s = 0
best_aupr_s = 0
best_acc_t = 0
best_auc_t = 0
best_aupr_t = 0
best_val_loss = float('inf')
patience_counter = 0
acc_t_list = []
auc_t_list = []
aupr_t_list = []

losses=[]

for epoch in range(args.n_epoch):
    embedding_dir = f"./embedding/{args.drug_name}/epoch_{epoch}"
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)


    len_dataloader = min(labels_s.shape[0], labels_t.shape[0])
    global rate
    rate = min((epoch + 1) / args.n_epoch, 0.05)

    for model in models:
        model.train()
    optimizer.zero_grad()

    if cuda:
        # adj_s = adj_s.cuda()
        # adj_t = adj_t.cuda()
        # labels_s = labels_s.cuda()
        # labels_t = labels_t.cuda()
        # X_n_s = X_n_s.cuda()
        # X_n_t = X_n_t.cuda()
        # features_s = features_s.cuda()
        # features_t = features_t.cuda()
        adj_label_s = adj_label_s.cuda()
        adj_label_t = adj_label_t.cuda()
        pos_weight_s = pos_weight_s.cuda()
        pos_weight_t = pos_weight_t.cuda()

    # S and T 's feature（features_s, features_t）以及graph（adj_s, adj_t）into private encoder，get mu , logvar, recovered feature。
    H_s_pr, mu_s, logvar_s = private_encoder_s(features_s, adj_s)
    H_t_pr, mu_t, logvar_t = private_encoder_t(features_t, adj_t)


    H_s_sh, shared_encoded_source1, shared_encoded_source2 = shared_encoder(adj_s, features_s)
    H_t_sh, shared_encoded_target1, shared_encoded_target2 = shared_encoder(adj_t, features_t)


    ''' compute encoder difference loss for S and T '''
    diff_loss_s = loss_diff(mu_s, shared_encoded_source1)
    diff_loss_t = loss_diff(mu_t, shared_encoded_target1)
    diff_loss_all = diff_loss_s + diff_loss_t

    ''' compute decoder reconstruction loss for S and T '''
    z_cat_s = torch.cat((H_s_pr, H_s_sh), 1)
    z_cat_t = torch.cat((H_t_pr, H_t_sh), 1)
    recovered_cat_s = decoder_s(z_cat_s)
    recovered_cat_t = decoder_t(z_cat_t)
    mu_cat_s = torch.cat((mu_s, shared_encoded_source1), 1)
    mu_cat_t = torch.cat((mu_t, shared_encoded_target1), 1)
    logvar_cat_s = torch.cat((logvar_s, shared_encoded_source2), 1)
    logvar_cat_t = torch.cat((logvar_t, shared_encoded_target2), 1)
    recon_loss_s = recon_loss(preds=recovered_cat_s, labels=adj_label_s,
                              mu=mu_cat_s, logvar=logvar_cat_s, n_nodes=features_s.shape[0],
                              norm=norm_s, pos_weight=pos_weight_s)
    recon_loss_t = recon_loss(preds=recovered_cat_t, labels=adj_label_t,
                              mu=mu_cat_t, logvar=logvar_cat_t, n_nodes=features_t.shape[0] * 2,
                              norm=norm_t, pos_weight=pos_weight_t)
    recon_loss_all = recon_loss_s + recon_loss_t

    ''' compute drug predictor loss for S '''
    source_logits = cls_model(shared_encoded_source1)
    cls_loss_source = cls_loss(source_logits, labels_s)
    source_acc = evaluate(source_logits, labels_s)

    ''' compute domain classifier loss for both S and T '''
    domain_output_s = domain_model(shared_encoded_source1)
    domain_output_t = domain_model(shared_encoded_target1)
    err_s_domain = cls_loss(domain_output_s,
                            torch.zeros(domain_output_s.size(0)).type(torch.LongTensor).to(device))
    err_t_domain = cls_loss(domain_output_t,
                            torch.ones(domain_output_t.size(0)).type(torch.LongTensor).to(device))
    loss_grl = err_s_domain + err_t_domain


    ''' compute entropy loss for T '''
    target_logits = cls_model(shared_encoded_target1)
    target_probs = F.softmax(target_logits, dim=-1)
    loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

    ''' compute overall loss '''
    loss = cls_loss_source  + args.lambda_d * loss_grl + args.lambda_r * recon_loss_all + args.lambda_f * diff_loss_all + loss_entropy * (
                epoch / args.n_epoch * 0.01)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch == args.n_epoch - 1 or True:
        save_dir = f"./pri_share_embed/{DRUG}"
        os.makedirs(save_dir, exist_ok=True)

        # embedding to numpy
        H_s_pr_np = H_s_pr.detach().cpu().numpy()
        H_t_pr_np = H_t_pr.detach().cpu().numpy()
        H_s_sh_np = H_s_sh.detach().cpu().numpy()
        H_t_sh_np = H_t_sh.detach().cpu().numpy()

        domain_s_np = domain_labels_s.cpu().numpy()
        domain_t_np = domain_labels_t.cpu().numpy()

        # construct DataFrame  and preserve private encoder
        df_private = pd.DataFrame(np.vstack([H_s_pr_np, H_t_pr_np]))
        df_private['domain'] = np.concatenate([domain_s_np, domain_t_np])
        df_private.to_csv(f"./pri_share_embed/{DRUG}/epoch_{epoch}_private_domain.csv", index=False)

        # construct DataFrame  and preserve shared encoder
        df_shared = pd.DataFrame(np.vstack([H_s_sh_np, H_t_sh_np]))
        df_shared['domain'] = np.concatenate([domain_s_np, domain_t_np])
        df_shared.to_csv(f"./pri_share_embed/{DRUG}/epoch_{epoch}_shared_domain.csv", index=False)

        print("the data preserved {pri_share_embed}/")

    if (epoch + 1) % 1 == 0:
        # Inside the training loop:
        if args.early_stopping:
            current_val_loss = loss.item()
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > args.patience:
                    print("Early stopping triggered")
                    break
            print(str(patience_counter) + '\t' + str(current_val_loss))

        acc_s, auc_s, aupr_s = test(adj_s, features_s, labels_s)
        if acc_s > best_acc_s:
            best_acc_s = acc_s
        if auc_s >best_auc_s:
            best_auc_s = auc_s
        if aupr_s > best_aupr_s:
            best_aupr_s = aupr_s
        acc_t, auc_t, aupr_t = test(adj_t, features_t, labels_t)
        acc_t_list.append(acc_t.item())
        auc_t_list.append(auc_t)
        aupr_t_list.append(aupr_t)
        if acc_t > best_acc_t:
            best_acc_t = acc_t
        if auc_t > best_auc_t:
            best_auc_t = auc_t
        if aupr_t > best_aupr_t:
            best_aupr_t = aupr_t
        print(
            'epoch: {},acc_source: {},AUC_source:{},AUPR_source:{},acc_target: {},AUC_target:{},AUPR_target:{},loss_class:{},loss_domain:{},loss_recon:{},loss_diff:{}'.format(
                epoch,
                acc_s.item(),
                auc_s,
                aupr_s,
                acc_t.item(),
                auc_t,
                aupr_t,
                cls_loss_source.item(),
                args.lambda_d * loss_grl.item(),
                args.lambda_r * recon_loss_all.item(),
                args.lambda_f * diff_loss_all.item()))
        losses.append([epoch,
                acc_t.item(),
                auc_t,
                aupr_t,
                cls_loss_source.item(),
                args.lambda_d * loss_grl.item(),
                args.lambda_r * recon_loss_all.item(),
                args.lambda_f * diff_loss_all.item()])

# import pandas as pd
# pd.DataFrame(losses).to_csv("loss.csv")
# if acc_t_list:
#     average_acc_t = sum(acc_t_list) / len(acc_t_list)
#     average_auc_t = sum(auc_t_list) / len(auc_t_list)
#     average_aupr_t = sum(aupr_t_list) / len(aupr_t_list)
#     print('Average target accuracy:', average_acc_t)
#     print('Average target AUC:', average_auc_t)
#     print('Average target AUPR:', average_aupr_t)
# else:
#     print('No target accuracies to average')
print('source best acc :{}'.format(best_acc_s))
print('source best AUC :{}'.format(best_auc_s))
print('source best AUPR :{}'.format(best_aupr_s))
print('target best acc :{}'.format(best_acc_t))
print('target best AUC :{}'.format(best_auc_t))
print('target best AUPR :{}'.format(best_aupr_t))
print('done')
print('lr:{},d:{},r:{},f:{}'.format(args.lr, args.lambda_d, args.lambda_r,
                                    args.lambda_f))

