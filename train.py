
import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluation import evaluate
from clustering import clustering
from utils import compare_score


def train(model, optimizer, data_loader, config, logger, seed, device):

    loss_metrics = {'loss': [], 'loss_rec': [], 'loss_cvc': [], 'loss_ncl': [],
                    'acc': [], 'nmi': [], 'ari': [], 'pur': []}

    is_best = False
    best_fu = (0.0, 0.0, 0.0, 0.0, is_best)

    # training
    for epoch in tqdm(range(config['epochs'])):
        loss_, loss_rec_, loss_cvc_, loss_ncl_ = 0., 0., 0., 0.
        feat_fu, feat1, feat2, label1_list, label2_list, flag_list = [], [], [], [], [], []
        for i, (x1, x2, label1, label2, flag_batch, indices_batch) in enumerate(data_loader):
            x1, x2, flag_batch = x1.to(device), x2.to(device), flag_batch.to(device)
            # Forward
            model.train()
            emb1, emb2, x1_hat, x2_hat = model(x1, x2)

            # Within-view Reconstruction
            loss_rec = model.reconstruction_loss(x1, x2, x1_hat, x2_hat)

            # View Distribution Alignment
            score_all = model.cross_corr_coef_matrix(emb1, emb2)
            score_aligned = score_all[flag_batch, :]
            score_aligned = score_aligned[: , flag_batch]
            loss_cvc = model.cross_corr_coef_loss(emb1[flag_batch],
                                                  emb2[flag_batch],
                                                  corr_coef_matrix=score_aligned
                                                  )

            # Semantic Matching Contrastive Learning
            score_aligned_diag = torch.diag(score_aligned.detach().clone())
            # adj with no gradient here
            adj = model.get_weighted_adj(score_all.detach().clone(),
                                         flag_batch,
                                         score_aligned_diag
                                         )
            H1, H2 = model.high_level_project(emb1, emb2)
            loss_ncl = model.cross_view_ncl(H1, H2, adj, tau=config['tau'])

            loss = loss_rec + loss_cvc * config['lambda1'] + loss_ncl * config['lambda2']

            loss_rec_ += loss_rec.item()
            loss_cvc_ += loss_cvc.item()
            loss_ncl_ += loss_ncl.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ += loss.item()

            H2_hat = torch.matmul(adj, H2.detach())
            H2_hat = F.normalize(H2_hat, p=2, dim=-1)
            feat_fu.append(torch.concat((H1.detach(), H2_hat), dim=-1))
            label1_list.append(label1)
            label2_list.append(label2)
            flag_list.append(flag_batch.cpu())

        if torch.isnan(torch.tensor(loss_)):
            break
        loss_metrics['loss'].append(loss_)

        feature_fu = torch.cat(feat_fu, dim=0)
        label_true1 = torch.cat(label1_list).numpy()
        label_true2 = torch.cat(label2_list).numpy()

        label_pred, _, _ = clustering(feature=feature_fu,
                                      cluster_num=config['n_classes'],
                                      device=device)
        acc, nmi, ari, pur = evaluate(label_true1, label_pred)
        loss_metrics['acc'].append(acc)
        loss_metrics['nmi'].append(nmi)
        loss_metrics['ari'].append(ari)
        loss_metrics['pur'].append(pur)

        is_best = False
        score = (acc, nmi, ari, pur, is_best)
        best_fu = compare_score(score, best_fu)
        if best_fu[-1]:
            best_epoch = epoch+1

        # if epoch == 0 or (epoch + 1) % 20 == 0:
        #     logger.info(f"[Epoch {epoch+1:<3}] loss: {loss_:.6f}, rec: {loss_rec_:.6f}, cvc: {loss_cvc_:.6f}, ncl: {loss_ncl_:.6f}")
        #     logger.info(f'   ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, PUR: {pur:.4f}')
    logger.write(f'  ACC {best_fu[0]:.4f}, NMI {best_fu[1]:.4f}, ARI {best_fu[2]:.4f}, PUR {best_fu[3]:.4f}')

    return best_fu
