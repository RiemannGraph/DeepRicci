python3 main.py \
--downstream_task clustering \
--dataset Cora  \
--eval_freq 100  \
--exp_iters 10   \
--epochs 2000  \
--hidden_features 512   \
--embed_features 4    \
--update_adj_ratio 0.1 \
--act_adj gumbel \
--act_func relu \
--wq_type linear \
--n_layers 2    \
--dropout_node 0.5  \
--dropout_edge 0.5  \
--lr 0.0001   \
--w_decay 0.0   \
--n_heads 8 \
--d_riemann 32  \
--d_hyla 128    \
--scale 0.1 \
--cur_h -1.0 \
--cur_s 1.0  \
--gamma 0.05 \
--eps 0.01  \
--topk 30   \
--temperature 0.2   \
--hidden_features_cls 32    \
--dropout_node_cls 0.5  \
--dropout_edge_cls 0.75 \
--n_layers_cls 2    \
--lr_cls 0.001  \
--w_decay_cls 0.0005  \
--epochs_cls 200    \
--patience_cls 10   \
--tau 0.9999    \
--iterations 0