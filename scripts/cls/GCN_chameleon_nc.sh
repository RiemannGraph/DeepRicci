python3 main.py \
--downstream_task classification \
--dataset chameleon  \
--eval_freq 20  \
--exp_iters 5   \
--epochs 100  \
--hidden_features 512   \
--embed_features 32   \
--update_adj_ratio 0.1 \
--act_adj elu \
--act_func relu \
--wq_type mlp \
--n_layers 2    \
--dropout_node 0.5  \
--dropout_edge 0.25  \
--lr 0.01   \
--w_decay 0.0   \
--n_heads 4 \
--d_riemann 16  \
--d_hyla 256    \
--scale 0.1 \
--cur_h -1.0 \
--cur_s 1.0  \
--s 1.  \
--r 2.  \
--gamma 0.05 \
--eps 0.01  \
--topk 30   \
--temperature 0.2   \
--backbone gcn  \
--backbone_topk 10  \
--n_heads_gat 8 \
--alpha_gat 0.2 \
--hidden_features_cls 32    \
--dropout_node_cls 0.2  \
--dropout_edge_cls 0.2 \
--n_layers_cls 2    \
--lr_cls 0.001  \
--w_decay_cls 5e-5  \
--epochs_cls 200    \
--patience_cls 10   \
--tau 0.9999    \
--iterations 0