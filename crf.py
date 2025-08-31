import numpy as np
import pickle
from collections import defaultdict

def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))

class FeatureIndexer:
    def __init__(self):
        self.feat2id = {"__bias__": 0}
        self.id2feat = ["__bias__"]

    def fit(self, sequences):
        # sequences: list of list-of-dicts
        for seq in sequences:
            for feats in seq:
                for k in feats.keys():
                    if k not in self.feat2id:
                        self.feat2id[k] = len(self.id2feat)
                        self.id2feat.append(k)
        return self

    def transform_seq(self, seq):
        # Return sparse representation: (idxs_list, vals_list) per timestep
        idxs, vals = [], []
        for feats in seq:
            ii = [0]  # bias
            vv = [1.0]
            for k, v in feats.items():
                if k == "__bias__":
                    continue
                j = self.feat2id.get(k)
                if j is not None:
                    ii.append(j)
                    vv.append(float(v))
            idxs.append(np.array(ii, dtype=np.int32))
            vals.append(np.array(vv, dtype=np.float32))
        return idxs, vals

    @property
    def n_features(self):
        return len(self.id2feat)

class LinearChainCRF:
    """
    Linear-chain CRF with:
      score(y|x) = sum_t [ W[y_t]·x_t + b[y_t] ] + sum_{t>0} T[y_{t-1}, y_t]
    We roll b[y] into W via a bias feature "__bias__".
    API:
      - fit(seqs_X, seqs_y, labels=None, epochs=20, lr=0.01, l2=1e-4, shuffle=True, verbose=1)
      - predict(seq_X) -> list[str]
      - predict_batch(seqs_X) -> list[list[str]]
      - predict_single(list_of_feature_dicts) -> list[str]  # convenience
      - save(path) / load(path)
    """
    def __init__(self):
        self.labels = []            # list[str]
        self.lab2id = {}            # str->int
        self.W = None               # (L, F)
        self.T = None               # (L, L)
        self.feats = FeatureIndexer()
        # Adam state
        self._mW = self._vW = None
        self._mT = self._vT = None
        self._t_adam = 0

    # ---------- utilities ----------
    def _ensure_labels(self, seqs_y, labels):
        if labels is None:
            labs = sorted({y for seq in seqs_y for y in seq})
        else:
            labs = list(labels)
        self.labels = labs
        self.lab2id = {lab:i for i,lab in enumerate(self.labels)}

    def _init_params(self, F, L):
        rng = np.random.RandomState(0)
        self.W = rng.normal(scale=0.01, size=(L, F)).astype(np.float32)
        self.T = rng.normal(scale=0.01, size=(L, L)).astype(np.float32)
        self._mW = np.zeros_like(self.W); self._vW = np.zeros_like(self.W)
        self._mT = np.zeros_like(self.T); self._vT = np.zeros_like(self.T)
        self._t_adam = 0

    def _seq_to_sparse(self, seq_X):
        return self.feats.transform_seq(seq_X)

    def _node_scores(self, idxs, vals):
        """
        idxs/vals: lists length T; each is np.array of feature indices/values
        returns node potentials S of shape (T, L): S[t, y] = W[y]·x_t
        """
        Tlen = len(idxs)
        L = self.W.shape[0]
        S = np.zeros((Tlen, L), dtype=np.float32)
        # For each timestep, accumulate
        for t in range(Tlen):
            ii, vv = idxs[t], vals[t]
            # W[:, ii] @ vv
            S[t] = (self.W[:, ii] * vv[np.newaxis, :]).sum(axis=1)
        return S

    # ---------- forward-backward ----------
    def _forward_backward(self, node_scores):
        """
        node_scores: (T, L)
        returns:
          logZ: float
          log_alpha: (T, L)
          log_beta:  (T, L)
        """
        Tlen, L = node_scores.shape
        # Forward
        log_alpha = np.full((Tlen, L), -np.inf, dtype=np.float32)
        log_alpha[0] = node_scores[0]
        for t in range(1, Tlen):
            # broadcast: prev (L,) + trans (L,L) -> (L,L) over prev->curr
            m = log_alpha[t-1][:, None] + self.T
            log_alpha[t] = _logsumexp(m, axis=0) + node_scores[t]
        logZ = _logsumexp(log_alpha[-1], axis=0)

        # Backward
        log_beta = np.full((Tlen, L), 0.0, dtype=np.float32)
        for t in range(Tlen-2, -1, -1):
            # next (L,) + trans (L,L) -> from t label to t+1 label
            m = self.T + (node_scores[t+1] + log_beta[t+1])[None, :]
            log_beta[t] = _logsumexp(m, axis=1)
        return logZ, log_alpha, log_beta

    def _marginals(self, node_scores, log_alpha, log_beta, logZ):
        """
        node_scores: (T, L)
        returns:
          p_t(y): (T, L) node marginals
          p_t_pair(i,j): list length T-1 of (L,L) pairwise marginals
        """
        Tlen, L = node_scores.shape
        # node marginals
        log_node = log_alpha + log_beta
        node_marg = np.exp(log_node - logZ)

        pair_margs = []
        for t in range(1, Tlen):
            # log p(y_{t-1}=i, y_t=j | x) ∝
            #   log_alpha[t-1,i] + T[i,j] + node_scores[t,j] + log_beta[t,j]
            M = (log_alpha[t-1][:, None] + self.T) + \
                (node_scores[t][None, :] + log_beta[t][None, :])
            M = M - _logsumexp(M, axis=None)
            pair_margs.append(np.exp(M))
        return node_marg, pair_margs

    # ---------- loss & gradient ----------
    def _seq_loss_grad(self, idxs, vals, y_ids, l2):
        """
        Returns negative log-likelihood and gradients dW, dT for one sequence.
        Gradients include L2 terms (on W and T).
        """
        Tlen = len(idxs)
        L, F = self.W.shape

        node_scores = self._node_scores(idxs, vals)
        logZ, log_alpha, log_beta = self._forward_backward(node_scores)
        node_marg, pair_margs = self._marginals(node_scores, log_alpha, log_beta, logZ)

        # Empirical score
        gold_score = 0.0
        for t in range(Tlen):
            y = y_ids[t]
            ii, vv = idxs[t], vals[t]
            gold_score += (self.W[y, ii] * vv).sum()
            if t > 0:
                gold_score += self.T[y_ids[t-1], y]

        nll = float(logZ - gold_score)

        # Gradients: expected - empirical (for NLL)
        dW = np.zeros_like(self.W)
        dT = np.zeros_like(self.T)

        # Emissions
        for t in range(Tlen):
            ii, vv = idxs[t], vals[t]
            # expected: sum_y p_t(y) * x_t
            # accumulate per label
            for y in range(L):
                if node_marg[t, y] != 0.0:
                    dW[y, ii] += node_marg[t, y] * vv
            # empirical: subtract x_t for gold y
            dW[y_ids[t], ii] -= vv

        # Transitions
        for t in range(1, Tlen):
            # expected pairwise
            dT += pair_margs[t-1]
            # empirical subtract one-hot of gold transition
            dT[y_ids[t-1], y_ids[t]] -= 1.0

        # L2
        dW += l2 * self.W
        dT += l2 * self.T

        return nll, dW, dT

    # ---------- optimizer (Adam) ----------
    def _adam_step(self, gW, gT, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t_adam += 1
        t = self._t_adam
        self._mW = beta1*self._mW + (1-beta1)*gW
        self._vW = beta2*self._vW + (1-beta2)*(gW*gW)
        mW_hat = self._mW / (1 - beta1**t)
        vW_hat = self._vW / (1 - beta2**t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        self._mT = beta1*self._mT + (1-beta1)*gT
        self._vT = beta2*self._vT + (1-beta2)*(gT*gT)
        mT_hat = self._mT / (1 - beta1**t)
        vT_hat = self._vT / (1 - beta2**t)
        self.T -= lr * mT_hat / (np.sqrt(vT_hat) + eps)

    # ---------- public API ----------
    def fit(self, seqs_X, seqs_y, labels=None, epochs=20, lr=0.01, l2=1e-4, shuffle=True, verbose=1, clip=5.0):
        """
        seqs_X: list of sequences, each sequence is a list of dict features
        seqs_y: list of sequences, each sequence is a list of label strings
        """
        assert len(seqs_X) == len(seqs_y)
        self._ensure_labels(seqs_y, labels)
        # feature indexer
        self.feats.fit(seqs_X)
        L = len(self.labels); F = self.feats.n_features
        self._init_params(F, L)

        # map y to ids once
        seqs_y_ids = [[self.lab2id[y] for y in ys] for ys in seqs_y]
        n = len(seqs_X)

        order = np.arange(n)
        for ep in range(1, epochs+1):
            if shuffle:
                np.random.shuffle(order)
            total_loss = 0.0
            for i in order:
                idxs, vals = self._seq_to_sparse(seqs_X[i])
                y_ids = seqs_y_ids[i]
                nll, dW, dT = self._seq_loss_grad(idxs, vals, y_ids, l2)
                # clip
                if clip is not None:
                    norm = np.sqrt((dW*dW).sum() + (dT*dT).sum())
                    if norm > clip:
                        dW *= clip / (norm + 1e-12)
                        dT *= clip / (norm + 1e-12)
                self._adam_step(dW, dT, lr)
                total_loss += nll
            if verbose:
                print(f"[CRF] epoch {ep}/{epochs}  avg NLL={total_loss/max(1,n):.4f}")
        return self

    def predict(self, seq_X):
        # Viterbi
        idxs, vals = self._seq_to_sparse(seq_X)
        node_scores = self._node_scores(idxs, vals)  # (T,L)
        Tlen, L = node_scores.shape
        if Tlen == 0:
            return []
        delta = np.zeros_like(node_scores)
        psi = np.full((Tlen, L), -1, dtype=np.int32)
        delta[0] = node_scores[0]
        for t in range(1, Tlen):
            # for each curr y: max over prev
            scores = delta[t-1][:, None] + self.T  # (L,L)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = node_scores[t] + np.max(scores, axis=0)
        # backtrace
        yseq = np.zeros(Tlen, dtype=np.int32)
        yseq[-1] = np.argmax(delta[-1])
        for t in range(Tlen-2, -1, -1):
            yseq[t] = psi[t+1, yseq[t+1]]
        return [self.labels[i] for i in yseq]

    def predict_batch(self, seqs_X):
        return [self.predict(seq) for seq in seqs_X]

    def predict_single(self, list_of_feature_dicts):
        """Convenience: accepts a (possibly length-1) sequence like your current code.
           Returns a list of labels (length = len(input seq))."""
        return self.predict(list_of_feature_dicts)

    def save(self, path):
        blob = {
            'labels': self.labels,
            'lab2id': self.lab2id,
            'W': self.W,
            'T': self.T,
            'feat2id': self.feats.feat2id,
            'id2feat': self.feats.id2feat,
        }
        with open(path, 'wb') as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            blob = pickle.load(f)
        m = cls()
        m.labels = blob['labels']
        m.lab2id = blob['lab2id']
        m.W = blob['W'].astype(np.float32)
        m.T = blob['T'].astype(np.float32)
        m.feats.feat2id = blob['feat2id']
        m.feats.id2feat = blob['id2feat']
        # init Adam slots
        m._mW = np.zeros_like(m.W); m._vW = np.zeros_like(m.W)
        m._mT = np.zeros_like(m.T); m._vT = np.zeros_like(m.T)
        m._t_adam = 0
        return m
