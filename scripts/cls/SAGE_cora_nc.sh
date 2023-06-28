python3 main.py \
--downstream_task classification \
--dataset Cora  \
--eval_freq 50  \
--exp_iters 5   \
--epochs 2000  \
--hidden_features 512   \
--embed_features 32    \
--update_adj_ratio 0.1 \
--act_adj gumbel \
--act_func relu \
--wq_type linear \
--n_layers 2    \
--dropout_node 0.75  \
--dropout_edge 0.75  \
--lr 0.01   \
--w_decay 0.   \
--n_heads 8 \
--d_riemann 32  \
--d_hyla 256    \
--scale 0.1 \
--cur_h -1.0 \
--cur_s 1.0  \
--s 1.0  \
--r 2.0  \
--gamma 5. \
--eps 0.01  \
--topk 60   \
--temperature 0.2   \
--backbone sage  \
--backbone_topk 10  \
--n_heads_gat 8 \
--alpha_gat 0.2 \
--hidden_features_cls 32    \
--dropout_node_cls 0.8  \
--dropout_edge_cls 0.8 \
--n_layers_cls 2    \
--lr_cls 0.001  \
--w_decay_cls 0.0005  \
--epochs_cls 200    \
--patience_cls 15   \
--tau 0.9999    \
--iterations 0