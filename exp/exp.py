import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from models import DeepRicci, RiemannianEmbeds
from backbone import GCN, SpGAT, GAT, GraphSAGE
from utils import cal_accuracy, cluster_metrics, cal_F1, graph_top_K
from data_factory import load_data
from sklearn.cluster import KMeans
from logger import create_logger
from geoopt.optim import RiemannianAdam


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def load_data(self):
        features, in_features, labels, adj, masks, n_classes = load_data(self.configs)
        return features, in_features, labels, adj, masks, n_classes

    def cal_cls_loss(self, model, mask, adj, features, labels):
        out = model(features, adj)
        loss = F.cross_entropy(out[mask], labels[mask])
        acc = cal_accuracy(out[mask], labels[mask])
        weighted_f1, macro_f1 = cal_F1(out[mask].detach().cpu(), labels[mask].detach().cpu())
        return loss, acc, weighted_f1, macro_f1

    def select_backbone_model(self, in_features, n_classes):
        if self.configs.backbone == 'gcn':
            model = GCN(in_features, self.configs.hidden_features_cls, n_classes, self.configs.n_layers_cls,
                        self.configs.dropout_node_cls, self.configs.dropout_edge_cls)
        elif self.configs.backbone == 'sage':
            model = GraphSAGE(in_features, self.configs.hidden_features_cls, n_classes, self.configs.n_layers_cls,
                              self.configs.dropout_node_cls, self.configs.dropout_edge_cls)
        elif self.configs.backbone == 'gat':
            model = GAT(in_features, self.configs.hidden_features_cls, n_classes,
                        self.configs.dropout_node_cls, self.configs.dropout_edge_cls,
                        alpha=self.configs.alpha_gat, n_heads=self.configs.n_heads_gat)
        elif self.configs.backbone == 'spgat':
            model = SpGAT(in_features, self.configs.hidden_features_cls, n_classes,
                          self.configs.dropout_node_cls, self.configs.dropout_edge_cls,
                          alpha=self.configs.alpha_gat, n_heads=self.configs.n_heads_gat)
        else:
            raise NotImplementedError
        return model

    def evaluate_adj_by_cls(self, adj, features, in_features, labels, n_classes, masks):
        """masks = (train, val, test)"""
        device = self.device
        model = self.select_backbone_model(in_features, n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), betas=self.configs.betas, lr=self.configs.lr_cls,
                                     weight_decay=self.configs.w_decay_cls)

        best_acc = 0.
        best_weighted_f1, best_macro_f1 = 0., 0.
        early_stop_count = 0
        best_model = None

        for epoch in range(1, self.configs.epochs_cls + 1):
            model.train()
            loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model, masks[0], adj, features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}")

            if epoch % 10 == 0:
                model.eval()
                val_loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model, masks[1], adj, features, labels)
                # print(f"Epoch {epoch}: val_loss={val_loss.item()}, val_accuracy={acc}")
                if acc > best_acc:
                    early_stop_count = 0
                    best_acc = acc
                    best_weighted_f1, best_macro_f1 = weighted_f1, macro_f1
                    best_model = model
                else:
                    early_stop_count += 1
                if early_stop_count >= self.configs.patience_cls:
                    break
        best_model.eval()
        test_loss, test_acc, test_weighted_f1, test_macro_f1 = self.cal_cls_loss(best_model, masks[2], adj, features,
                                                                                 labels)
        return best_acc, test_acc, best_model, test_weighted_f1, test_macro_f1

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        features, in_features, labels, init_adj, masks, n_classes = self.load_data()
        init_adj = init_adj + torch.eye(features.shape[0])

        if self.configs.downstream_task == 'clustering':
            self.configs.exp_iters = 1

        best_vals = []
        best_tests = []
        best_weighted_f1s = []
        best_macro_f1s = []
        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")
            Riemann_embeds_getter = RiemannianEmbeds(features.shape[0], self.configs.d_riemann,
                                                     self.configs.cur_h, self.configs.cur_s).to(device)
            model = DeepRicci(backbone=self.configs.backbone, n_layers=self.configs.n_layers,
                              n_heads=self.configs.n_heads,
                              in_features=in_features, d_riemann=self.configs.d_riemann, d_hyla=self.configs.d_hyla,
                              embed_features=self.configs.embed_features, hidden_features=self.configs.hidden_features,
                              dropout=self.configs.dropout_node, dropout_edge=self.configs.dropout_edge,
                              init_adj=init_adj,
                              scale=self.configs.scale, cur_h=self.configs.cur_h, cur_s=self.configs.cur_s,
                              gamma=self.configs.gamma, top_k=self.configs.topk,
                              act=self.configs.act_func, wq=self.configs.wq_type, act_adj=self.configs.act_adj,
                              eps=self.configs.eps, temperature=self.configs.temperature,
                              backbone_topk=self.configs.backbone_topk,
                              s=self.configs.s, r=self.configs.r, alpha_gat=self.configs.alpha_gat,
                              n_heads_gat=self.configs.n_heads_gat, device=device).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            r_optim = RiemannianAdam(Riemann_embeds_getter.parameters(), lr=self.configs.lr,
                                     weight_decay=self.configs.w_decay,
                                     stabilize=100)

            init_adj = init_adj.to(device)
            features = features.to(device)
            labels = labels.to(device)

            logger.info("--------------------------Training Start-------------------------")
            best_val = 0.
            best_val_test = 0.
            best_weighted_f1 = 0.
            best_macro_f1 = 0.
            best_cluster = {'acc': 0, 'nmi': 0, 'f1': 0, 'ari': 0}
            best_cluster_result = {}
            n_cluster_trials = self.configs.n_cluster_trials
            for epoch in range(1, self.configs.epochs + 1):
                model.train()
                Riemann_embeds_getter.train()
                update_adj_ratio = self.configs.update_adj_ratio
                if update_adj_ratio > 0.:
                    update_adj_ratio = 1 - np.sin(((epoch / self.configs.epochs) * np.pi) / 2) * update_adj_ratio
                else:
                    update_adj_ratio = 0.
                _, new_feature, adj, loss = model(features, init_adj, update_adj_ratio, Riemann_embeds_getter)

                r_optim.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                r_optim.step()
                optimizer.step()

                if (1 - self.configs.tau) and (self.configs.iterations == 0 or epoch % self.configs.iterations == 0):
                    init_adj = self.configs.tau * init_adj + (1 - self.configs.tau) * adj.detach()

                logger.info(f"Epoch {epoch}: train_loss={loss.item()}")

                if epoch % self.configs.eval_freq == 0:
                    logger.info("---------------Evaluation Start-----------------")
                    model.eval()
                    Riemann_embeds_getter.eval()
                    if self.configs.downstream_task == 'classification':
                        edge = adj.detach()
                        if self.configs.backbone in ['gat', 'spgat', 'sage']:
                            edge = graph_top_K(edge, self.configs.topk)
                        val_acc, test_acc, _, test_weighted_f1, test_macro_f1 = self.evaluate_adj_by_cls(edge,
                                                                                                         new_feature.detach(),
                                                                                                         in_features + 2 * self.configs.d_riemann,
                                                                                                         labels,
                                                                                                         n_classes,
                                                                                                         masks)
                        logger.info(
                            f"Epoch {epoch}: val_accuracy={val_acc.item() * 100: .2f}%, test_accuracy={test_acc * 100: .2f}%")
                        logger.info(
                            f"\t\t weighted_f1={test_weighted_f1.item() * 100: .2f}%, macro_f1={test_macro_f1 * 100: .2f}%")
                        logger.info("-------------------------------------------------------------------------")
                        if val_acc > best_val:
                            best_val = val_acc
                            best_val_test = test_acc
                            best_weighted_f1 = test_weighted_f1
                            best_macro_f1 = test_macro_f1

                    elif self.configs.downstream_task == 'clustering':
                        embedding, _, _, _ = model(features, init_adj, update_adj_ratio, Riemann_embeds_getter)
                        embedding = embedding.detach().cpu().numpy()
                        acc, nmi, f1, ari = [], [], [], []
                        for step in range(n_cluster_trials):
                            kmeans = KMeans(n_clusters=n_classes, random_state=step)
                            predicts = kmeans.fit_predict(embedding)
                            metrics = cluster_metrics(labels.cpu().numpy(), predicts)
                            acc_, nmi_, f1_, ari_ = metrics.evaluateFromLabel()
                            acc.append(acc_)
                            nmi.append(nmi_)
                            f1.append(f1_)
                            ari.append(ari_)
                        acc, nmi, f1, ari = np.mean(acc), np.mean(nmi), np.mean(f1), np.mean(ari)
                        if acc > best_cluster['acc']:
                            best_cluster['acc'] = acc
                            best_cluster_result['acc'] = [acc, nmi, f1, ari]
                        if nmi > best_cluster['nmi']:
                            best_cluster['nmi'] = nmi
                            best_cluster_result['nmi'] = [acc, nmi, f1, ari]
                        if f1 > best_cluster['f1']:
                            best_cluster['f1'] = f1
                            best_cluster_result['f1'] = [acc, nmi, f1, ari]
                        if ari > best_cluster['ari']:
                            best_cluster['ari'] = ari
                            best_cluster_result['ari'] = [acc, nmi, f1, ari]
                        logger.info(f"Epoch {epoch}: ACC: {acc}, NMI: {nmi}, F1: {f1}, ARI: {ari}")
                        logger.info("-------------------------------------------------------------------------")
                    else:
                        raise NotImplementedError

            if self.configs.downstream_task == 'classification':
                logger.info(
                    f"best_val_accuracy={best_val.item() * 100: .2f}%, best_test_accuracy={best_val_test * 100: .2f}%")
                logger.info(
                    f"weighted_f1={test_weighted_f1.item() * 100: .2f}%, macro_f1={test_macro_f1 * 100: .2f}%")
                best_vals.append(best_val.item())
                best_tests.append(best_val_test.item())
                best_weighted_f1s.append(best_weighted_f1.item())
                best_macro_f1s.append(best_macro_f1.item())

            if self.configs.downstream_task == 'clustering':
                for k, result in best_cluster_result.items():
                    acc, nmi, f1, ari = result
                    logger.info(f"Best Results according to {k}: ACC: {acc}, NMI: {nmi}, F1: {f1}, ARI: {ari} \n")
        if self.configs.downstream_task == 'classification':
            logger.info(f"best valid results: {np.max(best_vals)}")
            logger.info(f"best test results: {np.max(best_tests)}")
            logger.info(f"valid results: {np.mean(best_vals)}~{np.std(best_vals)}")
            logger.info(f"test results: {np.mean(best_tests)}~{np.std(best_tests)}")
            logger.info(f"test weighted-f1: {np.mean(best_weighted_f1s)}~{np.std(best_weighted_f1s)}")
            logger.info(f"test macro-f1: {np.mean(best_macro_f1s)}~{np.std(best_macro_f1s)}")
