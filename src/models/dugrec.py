import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

from common.abstract_recommender import GeneralRecommender
from types import SimpleNamespace

class DuGRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DuGRec, self).__init__(config, dataset)
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.decay_base = config['decay_base']
        self.decay_weight = config['decay_weight']
        self.cur_epoch = 0
        self.enable_residual = config['enable_residual']
        self.item_co_topk = config['item_co_topk']  
        self.alpha_item_co = config['alpha_item_co']  
        self.n_nodes = self.n_users + self.n_items
        self.cli_lambda = config['cli_lambda']

        self.cui_topk =  config['cui_topk']
        self.cui_theta =  config['cui_theta']  
        self.cui_gamma =  config['cui_gamma']  

        self.gpert_on = bool(getattr(config, 'gpert_on', True))
        self.gpert_lambda = config['gpert_lambda']
        self.gpert_tau = 0.2
        self.gpert_mask_p = config['gpert_mask_p']
        self.gpert_mode = 'mask' # 'mask' or 'gauss'
        self.lambda_spec = float(getattr(config, 'lambda_spec', 0.1))

        self.dct_keep_ratio = float(getattr(config, 'dct_keep_ratio', 0.2))
        self.beta_lf = float(getattr(config, 'beta_lf', 0.5))
        self.beta_hf = float(getattr(config, 'beta_hf', 0.1))
        self.beta_mix = float(getattr(config, 'beta_mix', 0.4))
        self.spec_gate_tau = float(getattr(config, 'spec_gate_tau', 1))
        self.enable_spec_gate = bool(getattr(config, 'enable_spec_gate', True))

        self.mod_item_proj = nn.Linear(3 * self.embedding_dim, self.embedding_dim, bias=False)
        self.id_item_proj = nn.Identity()

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.build_norm_bipartite_from_ui(self.interaction_matrix.tocoo())
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        v_feat_dim, t_feat_dim = 0, 0
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            v_feat_dim = self.v_feat.shape[1]

            self.v_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.embedding_dim), dtype=torch.float32, requires_grad=True), gain=1).to(
                self.device))

            self.v_MLP = nn.Linear(v_feat_dim, 4 * self.embedding_dim)
            self.v_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            t_feat_dim = self.t_feat.shape[1]
            self.t_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.embedding_dim), dtype=torch.float32, requires_grad=True), gain=1).to(
                self.device))
            self.t_MLP = nn.Linear(t_feat_dim, 4 * self.embedding_dim)
            self.t_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)

        self.id_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(self.n_users, self.embedding_dim), dtype=torch.float32, requires_grad=True), gain=1).to(
            self.device))
        self.s_MLP = nn.Linear(t_feat_dim + v_feat_dim, 4 * self.embedding_dim)
        self.s_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)


        w_t = dct.dct(self.t_feat, norm='ortho')
        w_v = dct.dct(self.v_feat, norm='ortho')
        self.interleaved_feat = torch.cat((w_v, w_t), 1)

        cache_dir = os.path.abspath(config['data_path'] + config['dataset'])
        self.build_and_cache_item_graphs_schemeA(
            k_sem=self.knn_k,
            k_co=self.item_co_topk,
            alpha_item_co=self.alpha_item_co,
            cache_dir=cache_dir
        )
        self.loss = SimpleNamespace(
            clu_tau=0.2,  
            clu_lambda=config['clu_lambda'],  
            pairs=set('vt,ts'.split(',')), # ,vs
        )

        self.u_proj_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.u_proj_t = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.u_proj_s = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self._att_prior = None  # [n_users,3]
        self._att_ema = None  # [n_users,3]
        self._att_used = None  # [n_users,3] 
        self._zu_v = self._zu_t = self._zu_s = None 
        self._att_prior = self._build_modality_prior().to(self.device)
        self._att_ema = self._att_prior.clone()
        self._att_used = self._att_prior.clone()

        self.i_proj_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.i_proj_t = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.i_proj_s = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        if not hasattr(self, 'loss'):
            self.loss = SimpleNamespace()
        self.loss.cli_tau = 0.2
        self.loss.cli_lambda = self.cli_lambda
        self.loss.i_pairs = getattr(self.loss, 'i_pairs', set('vt,ts'.split(',')))

        self.result_embed = None
        self.adj_cui = self.build_cui_ui().to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def _apply_dropedge(self, adj: torch.Tensor) -> torch.Tensor:
        if self.dropout <= 0.0:
            return adj
        adj = adj.coalesce()
        idx = adj.indices();
        val = adj.values()
        m = int(val.size(0) * (1. - self.dropout))
        prob = (val / (val.sum() + 1e-12)).clamp_min(1e-12)
        keep = torch.multinomial(prob, m, replacement=False)
        k_idx = idx[:, keep];
        k_val = val[keep]
        return torch.sparse_coo_tensor(k_idx, k_val, size=adj.shape, device=adj.device).coalesce()

    def pre_epoch_processing(self):
        self.cur_epoch += 1
        if getattr(self, 'adj_cui', None) is None:
            raise RuntimeError("adj_cui is None; build_cui_ui() must be called in __init__.")
        self.masked_adj = self._apply_dropedge(self.adj_cui)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, _unused_adj=None):

        tmp_v_feat = self.v_MLP_1(F.leaky_relu(self.v_MLP(self.v_feat)))
        tmp_t_feat = self.t_MLP_1(F.leaky_relu(self.t_MLP(self.t_feat)))
        tmp_s_feat = self.s_MLP_1(F.leaky_relu(self.s_MLP(self.interleaved_feat)))

        rep_uv = torch.cat((self.v_preference, tmp_v_feat), dim=0)
        rep_ut = torch.cat((self.t_preference, tmp_t_feat), dim=0)
        rep_sh = torch.cat((self.id_preference, tmp_s_feat), dim=0)
        v_x = torch.cat((F.normalize(rep_uv), F.normalize(rep_ut), F.normalize(rep_sh)), 1)

        user_rep, item_rep = self._lightgcn_propagate(self.masked_adj, v_x)


        d = self.embedding_dim
        v_i = item_rep[:, :d]
        t_i = item_rep[:, d:2 * d]
        s_i = item_rep[:, 2 * d:]

        self._zi_iv = F.normalize(self.i_proj_v(v_i), dim=1)  # [n_items, d]
        self._zi_it = F.normalize(self.i_proj_t(t_i), dim=1)
        self._zi_is = F.normalize(self.i_proj_s(s_i), dim=1)


        h = item_rep
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        item_rep = item_rep + h


        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        return user_rep, item_rep

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        reg_u_loss = (ua_embeddings ** 2).mean()
        reg_i_loss = (ia_embeddings ** 2).mean()

        reg_loss = self.reg_weight * (reg_u_loss + reg_i_loss)
        u_reg, u_stat = self._user_cl_total(users)

        batch_items = torch.unique(torch.cat([pos_items, neg_items], dim=0))
        i_mod_reg, i_mod_stat = self._item_mod_cl_total(batch_items)


        batch_items = torch.unique(torch.cat([pos_items, neg_items], dim=0))
        gpert_reg, gpert_stat = self._graph_perturb_ssl(users, batch_items, ua_embeddings, ia_embeddings)
        return batch_mf_loss + reg_loss + u_reg + i_mod_reg  + gpert_reg

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def _cl_fuse_users(self, v_rep, t_rep, s_rep):
        z_v = F.normalize(self.u_proj_v(v_rep), dim=1)
        z_t = F.normalize(self.u_proj_t(t_rep), dim=1)
        z_s = F.normalize(self.u_proj_s(s_rep), dim=1)

        sc_v = (z_v * z_t).sum(1, keepdim=True) + (z_v * z_s).sum(1, keepdim=True)
        sc_t = (z_t * z_v).sum(1, keepdim=True) + (z_t * z_s).sum(1, keepdim=True)
        sc_s = (z_s * z_v).sum(1, keepdim=True) + (z_s * z_t).sum(1, keepdim=True)
        att_new = F.softmax(torch.cat([sc_v, sc_t, sc_s], 1) / self.loss.clu_tau, dim=1)  # [n_users,3]

        beta_e = self.decay_weight * (1.0 - (self.decay_base ** max(1, self.cur_epoch)))
        beta_e = float(max(0.0, min(1.0, beta_e)))
        self._att_ema = (1.0 - beta_e) * self._att_ema + beta_e * att_new.detach()
        self._att_used = self._att_ema

        fused = torch.cat([
            self._att_used[:, [0]] * v_rep,
            self._att_used[:, [1]] * t_rep,
            self._att_used[:, [2]] * s_rep
        ], dim=1)

        self._zu_v, self._zu_t, self._zu_s = z_v, z_t, z_s
        return fused

    def _nce(self, A, B, tau):
        logits = (A @ B.t()) / tau
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)

    def _user_cl_total(self, users):
        device = users.device
        total = torch.tensor(0.0, device=device);
        stats = {}

        if self.loss.clu_lambda > 0:
            zv, zt, zs = self._zu_v[users], self._zu_t[users], self._zu_s[users]
            cl = torch.tensor(0.0, device=device)
            if 'vt' in self.loss.pairs: cl = cl + self._nce(zv, zt, self.loss.clu_tau)
            if 'vs' in self.loss.pairs: cl = cl + self._nce(zv, zs, self.loss.clu_tau)
            if 'ts' in self.loss.pairs: cl = cl + self._nce(zt, zs, self.loss.clu_tau)
            total += self.loss.clu_lambda * cl;
            stats['cl_u'] = float(cl.item())



        return total, stats

    def _build_modality_prior(self):
        ui: sp.coo_matrix = self.interaction_matrix.tocsr().astype(np.float32)
        deg = np.maximum(ui.getnnz(axis=1), 1).astype(np.float32)

        v_norm = np.linalg.norm(self.v_feat.cpu().numpy(), axis=1) if getattr(self, 'v_feat',
                                                                              None) is not None else None
        t_norm = np.linalg.norm(self.t_feat.cpu().numpy(), axis=1) if getattr(self, 't_feat',
                                                                              None) is not None else None

        nU = self.n_users
        sv = np.ones(nU, dtype=np.float32)
        st = np.ones(nU, dtype=np.float32)
        if v_norm is not None: sv = (ui @ v_norm) / deg
        if t_norm is not None: st = (ui @ t_norm) / deg

        ss = 0.5 * (sv + st)
        prior = torch.tensor(np.stack([sv, st, ss], axis=1), dtype=torch.float32)
        return torch.softmax(prior / 1.0, dim=1)  


    def _row_normalize_sparse(self, indices, values, shape, eps: float = 1e-12):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        if not torch.is_tensor(values):
            values = torch.tensor(values, dtype=torch.float32, device=self.device)
        adj = torch.sparse_coo_tensor(indices, values, size=shape, device=self.device).coalesce()
        rows = adj.indices()[0];
        vals = adj.values()
        row_sum = torch.zeros(shape[0], device=self.device, dtype=vals.dtype)
        row_sum.index_add_(0, rows, vals)
        norm_vals = vals / (row_sum[rows] + eps)
        return torch.sparse_coo_tensor(adj.indices(), norm_vals, size=shape, device=self.device).coalesce()

    def _row_softmax_sparse(self, indices, values, shape, eps: float = 1e-12):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        if not torch.is_tensor(values):
            values = torch.tensor(values, dtype=torch.float32, device=self.device)
        adj = torch.sparse_coo_tensor(indices, values, size=shape, device=self.device).coalesce()
        rows = adj.indices()[0];
        vals = adj.values()
        try:
            max_per_row = torch.full((shape[0],), -1e9, device=self.device, dtype=vals.dtype)
            max_per_row.scatter_reduce_(0, rows, vals, reduce='amax', include_self=True)
        except Exception:
            import numpy as _np
            r_cpu = rows.detach().cpu().numpy();
            v_cpu = vals.detach().cpu().numpy()
            mp = _np.full((shape[0],), -1e9, dtype=_np.float32)
            for r, v in zip(r_cpu, v_cpu):
                if v > mp[r]: mp[r] = v
            max_per_row = torch.tensor(mp, device=self.device, dtype=vals.dtype)
        e = torch.exp(vals - max_per_row[rows])
        denom = torch.zeros(shape[0], device=self.device, dtype=vals.dtype)
        denom.index_add_(0, rows, e)
        soft = e / (denom[rows] + eps)
        return torch.sparse_coo_tensor(adj.indices(), soft, size=shape, device=self.device).coalesce()

    def build_item_semantic_graph(self, k: int = None, alpha_image: float = None):
        k = self.knn_k if k is None else int(k)
        alpha_image = self.mm_image_weight if alpha_image is None else float(alpha_image)
        parts = []
        # Vision
        if getattr(self, 'v_feat', None) is not None and hasattr(self, 'image_embedding'):
            emb = self.image_embedding.weight.detach()
            emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            sim = torch.mm(emb, emb.t());
            sim.fill_diagonal_(-1.0)
            _, knn_ind = torch.topk(sim, k, dim=-1)
            rows = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1).expand(-1, k).reshape(-1)
            cols = knn_ind.reshape(-1)
            indices = torch.stack([rows, cols], dim=0)
            values = torch.ones(rows.numel(), device=self.device, dtype=torch.float32)
            G_img = self._row_normalize_sparse(indices, values, (self.n_items, self.n_items))
            parts.append(G_img);
            del sim
        # Text
        if getattr(self, 't_feat', None) is not None and hasattr(self, 'text_embedding'):
            emb = self.text_embedding.weight.detach()
            emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            sim = torch.mm(emb, emb.t());
            sim.fill_diagonal_(-1.0)
            _, knn_ind = torch.topk(sim, k, dim=-1)
            rows = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1).expand(-1, k).reshape(-1)
            cols = knn_ind.reshape(-1)
            indices = torch.stack([rows, cols], dim=0)
            values = torch.ones(rows.numel(), device=self.device, dtype=torch.float32)
            G_txt = self._row_normalize_sparse(indices, values, (self.n_items, self.n_items))
            parts.append(G_txt);
            del sim
        if len(parts) == 0:
            raise RuntimeError("No modality features to build semantic graph.")
        G_sem = parts[0] if len(parts) == 1 else (alpha_image * parts[0] + (1.0 - alpha_image) * parts[1])
        return G_sem.coalesce()

    def build_item_semantic_graph_joint(self, k: int = None, alpha_image: float = None):
        k = self.knn_k if k is None else int(k)
        alpha_image = self.mm_image_weight if alpha_image is None else float(alpha_image)
        G_spatial = self.build_item_semantic_graph(k=k, alpha_image=alpha_image)
        G_spec = self.build_item_semantic_graph_spectral(k=k, alpha_image=alpha_image)
        lam = float(self.lambda_spec)
        G_sem_joint = (1.0 - lam) * G_spatial + lam * G_spec
        return G_sem_joint.coalesce()

    def build_item_co_graph(self, topk: int = 100):
        ui: sp.coo_matrix = self.interaction_matrix 
        R = ui.tocsr()
        C = (R.T @ R).tocsr().astype(np.float32)
        C.setdiag(0.0);
        C.eliminate_zeros()
        rows, cols, vals = [], [], []
        n = C.shape[0]
        for i in range(n):
            start, end = C.indptr[i], C.indptr[i + 1]
            if start == end: continue
            row_idx = C.indices[start:end];
            row_val = C.data[start:end]
            if row_idx.size > topk:
                loc = np.argpartition(row_val, -topk)[-topk:]
                row_idx = row_idx[loc];
                row_val = row_val[loc]
            rows.extend([i] * row_idx.size);
            cols.extend(row_idx.tolist());
            vals.extend(row_val.tolist())
        if len(rows) == 0:
            indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            values = torch.empty((0,), dtype=torch.float32, device=self.device)
            return torch.sparse_coo_tensor(indices, values, size=(self.n_items, self.n_items), device=self.device)
        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.tensor(vals, dtype=torch.float32, device=self.device)
        return self._row_softmax_sparse(indices, values, (self.n_items, self.n_items)).coalesce()

    def build_and_cache_item_graphs_schemeA(self, k_sem, k_co, alpha_item_co, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        fused_path = os.path.join(
            cache_dir,
            f'Gii_fused_ksem{k_sem}_kco{k_co}_aco{int(alpha_item_co * 100)}'
            f'_aimg{int(self.mm_image_weight * 100)}_lspec{int(self.lambda_spec * 100)}.pt'
        )
        if os.path.exists(fused_path):
            self.mm_adj = torch.load(fused_path, map_location=self.device).coalesce().to(self.device)
            return
        G_sem = self.build_item_semantic_graph_joint(k=k_sem, alpha_image=self.mm_image_weight)
        G_co = self.build_item_co_graph(topk=k_co)

        if self.enable_spec_gate:
            with torch.no_grad():
                W = torch.cat([dct.dct(self.v_feat, norm='ortho'),
                               dct.dct(self.t_feat, norm='ortho')], dim=1)
                lf, _ = self._split_dct_bands(W, self.dct_keep_ratio)
                rho = (lf.norm(p=2, dim=1) / (W.norm(p=2, dim=1) + 1e-12))
            G_co = G_co.coalesce()
            ii = G_co.indices();
            vv = G_co.values()
            gate = torch.exp(- (rho[ii[0]] - rho[ii[1]]).abs() / max(1e-6, self.spec_gate_tau))
            G_co = torch.sparse_coo_tensor(ii, vv * gate, size=G_co.shape, device=G_co.device).coalesce()

        self.mm_adj = (alpha_item_co * G_co + (1.0 - alpha_item_co) * G_sem).coalesce().to(self.device)
        torch.save(self.mm_adj.cpu(), fused_path)

    def build_norm_bipartite_from_ui(self, ui_coo: sp.coo_matrix) -> torch.sparse.FloatTensor:
        n_u, n_i = self.n_users, self.n_items
        ui = ui_coo.tocoo()
        Rt = ui.transpose().tocoo()
        A = sp.dok_matrix((n_u + n_i, n_u + n_i), dtype=np.float32)
        data_dict = dict(zip(zip(ui.row, ui.col + n_u), [1] * ui.nnz))
        data_dict.update(dict(zip(zip(Rt.row + n_u, Rt.col), [1] * Rt.nnz)))
        A._update(data_dict)
        A = A.tocsr()
        deg = np.asarray((A > 0).sum(axis=1)).ravel().astype(np.float32) + 1e-7
        inv_sqrt = np.power(deg, -0.5)
        D = sp.diags(inv_sqrt)
        L = (D @ A @ D).tocoo()

        idx = torch.LongTensor(np.vstack([L.row, L.col]))
        val = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(idx, val, torch.Size((n_u + n_i, n_u + n_i))).to(self.device)

    def torch_sparse_to_scipy_csr(self, T: torch.Tensor) -> sp.csr_matrix:
        T = T.coalesce().cpu()
        row, col = T.indices().numpy()
        data = T.values().numpy()
        return sp.csr_matrix((data, (row, col)), shape=T.shape)

    def scipy_csr_to_torch_sparse(self, M: sp.csr_matrix) -> torch.Tensor:
        M = M.tocoo()
        idx = torch.LongTensor(np.vstack([M.row, M.col])).to(self.device)
        val = torch.FloatTensor(M.data).to(self.device)
        return torch.sparse_coo_tensor(idx, val, size=M.shape, device=self.device).coalesce()

    def _filter_sparse_topk_threshold(self, S: torch.Tensor, topk: int, theta: float) -> torch.Tensor:
        S = S.coalesce()
        rows, cols, vals = S.indices()[0], S.indices()[1], S.values()
        n_rows = S.size(0)
        buckets = [[] for _ in range(n_rows)]
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
            if v >= theta:
                buckets[r].append((c, v))
        f_rows, f_cols, f_vals = [], [], []
        for r, lst in enumerate(buckets):
            if not lst: continue
            if len(lst) > topk:
                lst = sorted(lst, key=lambda x: x[1], reverse=True)[:topk]
            for c, v in lst:
                f_rows.append(r);
                f_cols.append(c);
                f_vals.append(v)
        if not f_rows:
            return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long, device=self.device),
                                           torch.empty((0,), dtype=torch.float32, device=self.device),
                                           size=S.shape, device=self.device)
        idx = torch.tensor([f_rows, f_cols], dtype=torch.long, device=self.device)
        val = torch.tensor(f_vals, dtype=torch.float32, device=self.device)
        return self._row_normalize_sparse(idx, val, S.shape).coalesce()

    def build_cui_ui(self, use_semantic_only: bool = False) -> torch.Tensor:
        S = self.mm_adj 
        S_f = self._filter_sparse_topk_threshold(S, topk=self.cui_topk, theta=self.cui_theta)  

        R = self.interaction_matrix.tocsr()  # [U, I]
        S_csr = self.torch_sparse_to_scipy_csr(S_f)  # [I, I]
        R_tilde = (R @ S_csr).tocsr()  # [U, I]
        if self.cui_gamma is not None:
            R_mix = self.cui_gamma * R + (1.0 - self.cui_gamma) * R_tilde
        else:
            R_mix = R_tilde

        return self.build_norm_bipartite_from_ui(R_mix.tocoo())

    def _lightgcn_propagate(self, adj: torch.Tensor, v_x: torch.Tensor):
        ego = v_x
        all_embeddings = [ego]
        if self.enable_residual:
            for _ in range(self.n_ui_layers):
                side = torch.sparse.mm(adj, ego)
                _w = F.cosine_similarity(side, ego, dim=-1)
                side = torch.einsum('a,ab->ab', _w, side)
                ego = side
                all_embeddings.append(ego)
        else:
            for _ in range(self.n_ui_layers):
                ego = torch.sparse.mm(adj, ego)
                all_embeddings.append(ego)
        H = torch.stack(all_embeddings, dim=0).sum(dim=0)
        item_rep = H[self.n_users:]
        uvw = H[:self.n_users]
        v_rep = uvw[:, :self.embedding_dim]
        t_rep = uvw[:, self.embedding_dim: 2 * self.embedding_dim]
        s_rep = uvw[:, 2 * self.embedding_dim:]
        user_rep = self._cl_fuse_users(v_rep, t_rep, s_rep)
        return user_rep, item_rep

    def _item_mod_cl_total(self, items_unique: torch.Tensor):
        device = items_unique.device
        total = torch.tensor(0.0, device=device);
        stats = {}

        if getattr(self.loss, 'cli_lambda', 0.0) <= 0:
            return total, stats

        z_v_all = getattr(self, '_zi_iv', None)
        z_t_all = getattr(self, '_zi_it', None)
        z_s_all = getattr(self, '_zi_is', None)
        if (z_v_all is None) or (z_t_all is None) or (z_s_all is None):
            return total, stats

        zv = z_v_all[items_unique]  # [B_i, d]
        zt = z_t_all[items_unique]
        zs = z_s_all[items_unique]

        cl = torch.tensor(0.0, device=device)
        pairs = getattr(self.loss, 'i_pairs', set(('vt', 'ts')))

        if 'vt' in pairs:
            cl = cl + self._nce(zv, zt, getattr(self.loss, 'cli_tau', 0.2))
        if 'ts' in pairs:
            cl = cl + self._nce(zt, zs, getattr(self.loss, 'cli_tau', 0.2))
        if 'vs' in pairs:
            cl = cl + self._nce(zv, zs, getattr(self.loss, 'cli_tau', 0.2))

        total = total + self.loss.cli_lambda * cl
        stats['cl_i_mod'] = float(cl.item())
        return total, stats


    def _graph_perturb_ssl(self, users, batch_items, ua_embeddings, ia_embeddings):
        if not getattr(self, 'gpert_on', False) or self.gpert_lambda <= 0:
            return torch.tensor(0., device=ua_embeddings.device), {}

        def _mk_views(x):
            if self.gpert_mode == 'gauss':
                sigma = 0.1
                e1 = F.normalize(x + sigma * torch.randn_like(x), dim=1)
                e2 = F.normalize(x + sigma * torch.randn_like(x), dim=1)
            else:
                p = self.gpert_mask_p
                m1 = (torch.rand_like(x) > p).float();
                m2 = (torch.rand_like(x) > p).float()
                keep1 = (1.0 - p);
                keep2 = (1.0 - p)
                e1 = F.normalize(x * m1 / keep1, dim=1)
                e2 = F.normalize(x * m2 / keep2, dim=1)
            return e1, e2

        u = ua_embeddings[users]  # [B, D]
        i = ia_embeddings[batch_items]  # [B_i, D]
        u1, u2 = _mk_views(u);
        i1, i2 = _mk_views(i)

        ssl_u = self._nce(u1, u2, self.gpert_tau)
        ssl_i = self._nce(i1, i2, self.gpert_tau)
        reg = self.gpert_lambda * (ssl_u + ssl_i)
        return reg, {'gpert_u': float(ssl_u.item()), 'gpert_i': float(ssl_i.item())}

    def _split_dct_bands(self, W, keep_ratio):
        D = W.size(1);
        k = max(1, int(D * keep_ratio))
        return W[:, :k], W[:, k:]

    def _knn_from_feat(self, F, k, shape):
        F = F / (F.norm(p=2, dim=1, keepdim=True) + 1e-12)
        sim = torch.mm(F, F.t());
        sim.fill_diagonal_(-1.0)
        _, nn_idx = torch.topk(sim, k, dim=-1)
        rows = torch.arange(nn_idx.size(0), device=F.device).unsqueeze(1).expand(-1, k).reshape(-1)
        cols = nn_idx.reshape(-1)
        idx = torch.stack([rows, cols], 0)
        val = torch.ones(rows.numel(), device=F.device, dtype=torch.float32)
        return self._row_normalize_sparse(idx, val, shape).coalesce()

    def build_item_semantic_graph_spectral(self, k: int = None, alpha_image: float = None):
        k = self.knn_k if k is None else int(k)
        alpha_image = self.mm_image_weight if alpha_image is None else float(alpha_image)
        shape = (self.n_items, self.n_items)
        w_v = dct.dct(self.v_feat, norm='ortho') if getattr(self, 'v_feat', None) is not None else None
        w_t = dct.dct(self.t_feat, norm='ortho') if getattr(self, 't_feat', None) is not None else None
        w_vt = self.interleaved_feat 
        G_v_lf = G_v_hf = G_t_lf = G_t_hf = None
        if w_v is not None:
            v_lf, v_hf = self._split_dct_bands(w_v, self.dct_keep_ratio)
            G_v_lf = self._knn_from_feat(v_lf, k, shape)
            G_v_hf = self._knn_from_feat(v_hf, k, shape)
        if w_t is not None:
            t_lf, t_hf = self._split_dct_bands(w_t, self.dct_keep_ratio)
            G_t_lf = self._knn_from_feat(t_lf, k, shape)
            G_t_hf = self._knn_from_feat(t_hf, k, shape)


        G_mix = self._knn_from_feat(w_vt, k, shape)

        def _sum(A, B, ai):
            if A is None: return B
            if B is None: return A
            return ai * A + (1.0 - ai) * B

        G_lf = _sum(G_v_lf, G_t_lf, alpha_image)
        G_hf = _sum(G_v_hf, G_t_hf, alpha_image)
        G_spec = None
        if G_lf is not None: G_spec = self.beta_lf * G_lf if G_spec is None else G_spec + self.beta_lf * G_lf
        if G_hf is not None: G_spec = self.beta_hf * G_hf if G_spec is None else G_spec + self.beta_hf * G_hf
        if G_mix is not None: G_spec = self.beta_mix * G_mix if G_spec is None else G_spec + self.beta_mix * G_mix
        if G_spec is None: raise RuntimeError("No modality to build spectral semantic graph.")
        return G_spec.coalesce()















