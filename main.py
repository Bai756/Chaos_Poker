# /// script
# dependencies = [
# "sklearn_crfsuite",
# "numpy"
# ]
# ///
import pickle
import asyncio
import pygame as pg
import os
import random
import numpy as np
import secrets
# from base import evaluate_hand, Deck, hand_name_from_rank, Player, Card
# from sklearn_crf import mc_win_prob, board_texture, draws_flags, event_to_features

# ------- crf and base since pygbag imports don't work ----------------
# ------------
# ----------
# ------
from itertools import combinations
from collections import Counter

SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

HAND_ORDER = [
    "7-card straight-flush",
    "Quad set (4 + 3)",
    "6-card straight-flush",
    "7-card flush",
    "5-card straight-flush",
    "Quad House (4 + 2 + 1)",
    "Super set (3 + 3 + 1)",
    "7-card straight",
    "Mega full house (3 + 2 + 2)",
    "Quads",
    "6-card flush",
    "6-card straight",
    "3 pair (2 + 2 + 2 + 1)",
    "Full House",
    "5-card flush",
    "Trips",
    "5-card straight",
    "2 pair",
    "1 pair",
    "High card"
]
HAND_RANK = {name: i for i, name in enumerate(HAND_ORDER[::-1])}

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    def __str__(self):
        return f"{self.rank}{self.suit}"

class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for s in SUITS for r in RANKS]
        random.shuffle(self.cards)
    def deal(self, n):
        dealt, self.cards = self.cards[:n], self.cards[n:]
        return dealt
    def shuffle(self):
        random.shuffle(self.cards)

class Player:
    def __init__(self, name, chips=500):
        self.name = name
        self.chips = chips
        self.hand = []
        self.active = True
        self.current_bet = 0
        self.acted = False
        self.all_in = False

def evaluate_hand(cards):

    def is_joker(c):
        return isinstance(c.rank, str) and c.rank.lower() in ("joker", "jk")

    # Separate jokers from concrete cards
    non_jokers = [c for c in cards if not is_joker(c)]
    jokers = [c for c in cards if is_joker(c)]
    num_jokers = len(jokers)

    def eval_no_joker(card_list):
        rank_map = {r: i for i, r in enumerate(RANKS, 2)}
        values = sorted([rank_map[c.rank] for c in card_list], reverse=True)
        suits = [c.suit for c in card_list]
        counts = Counter(values)
        suit_counts = Counter(suits)

        flush_suit = None
        for suit, count in suit_counts.items():
            if count >= 5:
                flush_suit = suit
                break
        flush_cards = sorted([rank_map[c.rank] for c in card_list if c.suit == flush_suit], reverse=True) if flush_suit else []

        def get_straight(vals, length=5):
            vals = sorted(set(vals), reverse=True)
            for i in range(len(vals) - length + 1):
                window = vals[i:i+length]
                if window[0] - window[-1] == length - 1:
                    return window[0]

            # Check for wheel straights (Ace as low)
            if length == 5 and set([14, 5, 4, 3, 2]).issubset(vals):
                return 5
            elif length == 6 and set([14, 6, 5, 4, 3, 2]).issubset(vals):
                return 6
            elif length == 7 and set([14, 7, 6, 5, 4, 3, 2]).issubset(vals):
                return 7

            return None

        quads = [v for v, c in counts.items() if c == 4]
        trips = [v for v, c in counts.items() if c == 3]
        pairs = [v for v, c in counts.items() if c == 2]
        singles = [v for v, c in counts.items() if c == 1]

        # 1. 7-card straight-flush (exact)
        if flush_suit and len(flush_cards) == 7:
            sf = get_straight(flush_cards, 7)
            if sf:
                return (HAND_RANK["7-card straight-flush"], sf)

        # 2. Quad set (4 + 3)
        if quads and trips:
            return (HAND_RANK["Quad set (4 + 3)"], max(quads), max(trips))

        # 3. 6-card straight-flush (exact)
        if flush_suit and len(flush_cards) >= 6:
            sf = get_straight(flush_cards, 6)
            if sf and not get_straight(flush_cards, 7):
                return (HAND_RANK["6-card straight-flush"], sf)

        # 4. 7-card flush
        if flush_suit and len(flush_cards) == 7:
            return (HAND_RANK["7-card flush"], *flush_cards)

        # 5. 5-card straight-flush (exact)
        if flush_suit and len(flush_cards) >= 5:
            sf = get_straight(flush_cards, 5)
            if sf and not get_straight(flush_cards, 6) and not get_straight(flush_cards, 7):
                return (HAND_RANK["5-card straight-flush"], sf)

        # 6. Quad House (4 + 2 + 1)
        if quads and pairs and singles:
            return (HAND_RANK["Quad House (4 + 2 + 1)"], max(quads), max(pairs), max(singles))

        # 7. Super set (3 + 3 + 1)
        if len(trips) >= 2 and singles:
            t1, t2 = sorted(trips, reverse=True)[:2]
            return (HAND_RANK["Super set (3 + 3 + 1)"], t1, t2, max(singles))

        # 8. 7-card straight
        if len(set(values)) == 7:
            s7 = get_straight(values, 7)
            if s7:
                return (HAND_RANK["7-card straight"], s7)

        # 9. Mega full house (3 + 2 + 2)
        if trips and len(pairs) >= 2:
            return (HAND_RANK["Mega full house (3 + 2 + 2)"], max(trips), *sorted(pairs, reverse=True)[:2])

        # 10. Quads
        if quads and len(singles) >= 3:
            return (HAND_RANK["Quads"], max(quads), *sorted(singles, reverse=True)[:3])

        # 11. 6-card flush
        if flush_suit and len(flush_cards) == 6:
            return (HAND_RANK["6-card flush"], *flush_cards)

        # 12. 6-card straight
        s6 = get_straight(values, 6)
        if s6 and not get_straight(values, 7):
            return (HAND_RANK["6-card straight"], s6)

        # 13. 3 pair (2 + 2 + 2 + 1)
        if len(pairs) >= 3 and singles:
            return (HAND_RANK["3 pair (2 + 2 + 2 + 1)"], *sorted(pairs, reverse=True)[:3], max(singles))

        # 14. Full House
        if trips and pairs and len(singles) >= 2:
            return (HAND_RANK["Full House"], max(trips), max(pairs), *sorted(singles, reverse=True)[:2])

        # 15. 5-card flush
        if flush_suit and len(flush_cards) == 5:
            return (HAND_RANK["5-card flush"], *flush_cards)

        # 16. Trips
        if trips and len(singles) >= 4:
            return (HAND_RANK["Trips"], max(trips), *sorted(singles, reverse=True)[:4])

        # 17. 5-card straight
        s5 = get_straight(values, 5)
        if s5 and not get_straight(values, 6) and not get_straight(values, 7):
            return (HAND_RANK["5-card straight"], s5)

        # 18. 2 pair
        if len(pairs) >= 2 and len(singles) >= 3:
            return (HAND_RANK["2 pair"], *sorted(pairs, reverse=True)[:2], *sorted(singles, reverse=True)[:3])

        # 19. 1 pair
        if pairs and len(singles) >= 5:
            return (HAND_RANK["1 pair"], max(pairs), *sorted(singles, reverse=True)[:5])

        # 20. High card
        return (HAND_RANK["High card"], *values[:7])

    if num_jokers == 0:
        return eval_no_joker(cards)

    present = {(c.rank, c.suit) for c in non_jokers}
    all_possible = [Card(r, s) for s in SUITS for r in RANKS if (r, s) not in present]

    best_score = None

    # Iterate all combinations of distinct replacement cards for jokers
    for combo in combinations(all_possible, num_jokers):
        trial_cards = non_jokers + list(combo)
        score = eval_no_joker(trial_cards)
        if best_score is None or score > best_score:
            best_score = score

    return best_score

def hand_name_from_rank(rank):
    return HAND_ORDER[::-1][rank]

RANK_TO_VAL = {r: i for i, r in enumerate(RANKS, start=2)}

def board_texture(board):
    """Simple texture features."""
    vals = [RANK_TO_VAL.get(c.rank, 0) for c in board]
    suits = [c.suit for c in board]
    suit_cnt = Counter(suits)
    paired = any(v >= 2 for v in Counter(vals).values())

    # flush draw: 4 of a suit on board+hand will be handled elsewhere; here, only board
    max_suit_on_board = max(suit_cnt.values()) if suits else 0

    # straighty board
    uniq = sorted(set(vals))
    straighty = 0
    for i in range(max(0, len(uniq)-4)):
        if uniq[i+4] - uniq[i] <= 5:
            straighty = 1
            break
    return {
        "board_paired": int(paired),
        "board_flushy": int(max_suit_on_board >= 3),
        "board_straighty": int(straighty),
    }

def draws_flags(hand, board):
    allc = hand + board
    suits = [c.suit for c in allc]
    suit_cnt = Counter(suits)
    flush_draw = any(v == 4 for v in suit_cnt.values())

    vals = sorted(set([RANK_TO_VAL.get(c.rank, 0) for c in allc]))
    # crude straight draw: any 4-gap window with length <= 5 and at least 4 cards
    ooe = 0
    for comb in combinations(vals, 4):
        if comb[-1] - comb[0] <= 5:
            ooe = 1
            break
    return {"flush_draw": int(flush_draw), "straight_draw": int(ooe)}

def mc_win_prob(hero_hand, board, deck_cards, opp_count=1, trials=200):
    """Estimate hero's win probability via Monte Carlo simulation."""
    if trials <= 0:
        return 0.0

    wins = ties = 0
    used = hero_hand + board
    
    for _ in range(trials):
        # Filter out cards that are already in play
        available_cards = [c for c in deck_cards 
                         if not any(c.rank == card.rank and c.suit == card.suit 
                                    for card in used)]
        random.shuffle(available_cards)

        # Opponents' hole cards
        opp_holes = []
        idx = 0
        for _o in range(opp_count):
            opp_holes.append([available_cards[idx], available_cards[idx+1]])
            idx += 2

        # Complete board to 5 cards
        need = 5 - len(board)
        draw_board = board + available_cards[idx:idx+need]
        idx += need

        hero_eval = evaluate_hand(hero_hand + draw_board)
        opp_evals = [evaluate_hand(h + draw_board) for h in opp_holes]

        best_opp = max(opp_evals)
        if hero_eval > best_opp:
            wins += 1
        elif hero_eval == best_opp:
            ties += 1

    return (wins + 0.5 * ties) / trials

def event_to_features(event):
    ev = (event or "Normal").lower()

    feats = {
        "ev_normal": int(ev == "normal"),
        "ev_war": int("war" in ev),
        "ev_joker_wild": int("joker" in ev),
        "ev_only_face": int("only face" in ev),
        "ev_no_face": int("no face" in ev),
        "ev_one_suit": int("only" in ev and "and" not in ev and "suit" in ev),
        "ev_two_suits": int("only" in ev and "and" in ev and "suit" in ev),
        "ev_rankings_reversed": int("ranking" in ev and "revers" in ev),
    }

    # Banned ranks
    banned_present = False
    banned_rank = None
    for r in RANKS:
        token = f"no {r.lower()}s"
        if token in ev:
            banned_present = True
            banned_rank = r
            break
    feats["ev_banned_rank_present"] = int(banned_present)
    for r in RANKS:
        feats[f"ev_banned_rank_{r}"] = int(banned_rank == r)

    # Suits
    suits_map = {"♠": "spades", "♥": "hearts", "♦": "diamonds", "♣": "clubs"}
    # Detect "only X suit" or "only X and Y"
    allowed_suits = []
    for name in suits_map.keys():
        if f"only {name}" in ev or name in ev and "only" in ev:
            allowed_suits.append(name)

    for nm in suits_map.values():
        feats[f"ev_allowed_suit_{nm}"] = int(nm in [suits_map[s] for s in allowed_suits])

    feats["ev_label_"+ev.replace(" ", "_")] = 1
    return feats

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

# ----- end of crf and base functions ------
# -----------

_seed = int.from_bytes(secrets.token_bytes(8), "big")
random.seed(_seed)
np.random.seed(_seed % (2**32))

BTN_X = 750
BTN_W = 130
BTN_H = 40
BTN_SPACING = 70
BTN_Y_START = 180

INFOBOX_W = 220
INFOBOX_H = 60
INFOBOX_PLAYER_X = 40
INFOBOX_PLAYER_Y = 470
INFOBOX_OPP_X = 40
INFOBOX_OPP_Y = 30

CARD_W = 60
CARD_H = 90
CARD_SPACING = 80

SLIDER_X = 300
SLIDER_Y = 420
SLIDER_W = 300
SLIDER_H = 8

PRESET_BW = 125
PRESET_BH = 32
PRESET_BY = 480

OK_BTN_RECT = pg.Rect(700, 480, 70, 32)
CANCEL_BTN_RECT = pg.Rect(190, 480, 100, 32)

pg.init()
pg.font.init()

FONT = pg.font.Font("assets/fonts/arial.ttf", 24)
BIG_FONT = pg.font.Font("assets/fonts/arialbd.ttf", 28)
CARD_FONT = pg.font.Font("assets/fonts/arialbd.ttf", 32)

GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
RED = (220, 0, 0)
GOLD = (255, 215, 0)

WIDTH, HEIGHT = 900, 600
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Chaos Poker")

players = []
community_cards = []
deck = None
pot = 0
round_stage = "preflop"
message = "Welcome to Chaos Poker!"
winner = None
show_bet_slider = False
bet_slider_value = 0
bet_slider_min = 0
bet_slider_max = 0
bet_slider_preset = []
slider_dragging = False
dealer_position = 0
waiting_for_action = False
player_contributions = [0, 0]

game_event = ""

try:
    model_path = os.path.join("assets", "crf_custom_v1.pkl")
    crf_model = LinearChainCRF.load(model_path)
    print("CRF AI model loaded successfully")

except Exception as e:
    print(f"Error loading CRF model: {e}")
    crf_model = None

RANK_TO_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                 '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

class Button:
    def __init__(self, text, x, y, w, h, color, action):
        self.rect = pg.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.action = action
        self.visible = True

    def draw(self, surface):
        pg.draw.rect(surface, self.color, self.rect, border_radius=6)
        txt = FONT.render(self.text, True, BLACK)
        surface.blit(txt, (self.rect.centerx - txt.get_width() // 2,
                           self.rect.centery - txt.get_height() // 2))

    def click(self):
        self.action()

def open_bet_slider():
    global show_bet_slider, bet_slider_min, bet_slider_max, bet_slider_value, bet_slider_preset
    opponent_bet = players[1].current_bet
    player_bet = players[0].current_bet
    min_bet = opponent_bet - player_bet + 10 if opponent_bet > player_bet else 10
    max_bet = players[0].chips
    pot_size = min(pot, max_bet)
    show_bet_slider = True
    bet_slider_min = min_bet
    bet_slider_max = max_bet
    bet_slider_value = min_bet
    bet_slider_preset = [min_bet, pot_size, max_bet]
    update_buttons()

def update_buttons():
    opponent_bet = players[1].current_bet
    player_bet = players[0].current_bet

    if opponent_bet > player_bet:
        buttons[0].text = f"Call ({opponent_bet - player_bet})"
        buttons[1].text = "Raise"
    else:
        buttons[0].text = "Check"
        buttons[1].text = "Bet"

def draw_info_box(x, y, player):
    pg.draw.rect(screen, (30, 30, 30), (x, y, INFOBOX_W, INFOBOX_H), border_radius=10)
    info = f"Chips: {player.chips}  Bet: {player.current_bet}"
    info_text = FONT.render(info, True, WHITE)
    screen.blit(info_text, (x + 10, y + 15))

def draw_hand(cards, x, y, highlight=False, hide=False):
    for i, card in enumerate(cards):
        display_card = "" if hide else card
        draw_card(screen, display_card, x + i * CARD_SPACING, y, highlight)

def draw_card(surface, card, x, y, highlight=False):
    pg.draw.rect(surface, (60, 60, 60), (x+4, y+6, CARD_W, CARD_H), border_radius=8)
    border_color = GOLD if highlight else WHITE
    pg.draw.rect(surface, border_color, (x, y, CARD_W, CARD_H), border_radius=8)
    pg.draw.rect(surface, (240, 240, 240), (x+3, y+3, CARD_W-6, CARD_H-6), border_radius=6)

    dx = 10
    if not card:
        card_str = ""
        color = GRAY
    elif card.rank == "JOKER":
        pg.draw.rect(surface, (250, 245, 200), (x+6, y+6, CARD_W-12, CARD_H-12), border_radius=6)
        card_str = "JK"
        color = GOLD
    else:
        card_str = f"{card.rank}{card.suit}"
        if card.rank == "10":
            dx = 3
        color = BLACK if card.suit in ['♠', '♣'] else RED

    text = CARD_FONT.render(card_str, True, color)
    surface.blit(text, (x + dx, y + 30))

def draw_bet_slider():
    pg.draw.rect(screen, (40, 40, 40), (250, 350, 400, 140), border_radius=12)
    pg.draw.rect(screen, GOLD, (250, 350, 400, 140), 3, border_radius=12)
    txt = BIG_FONT.render("Choose Bet Amount", True, WHITE)
    screen.blit(txt, (WIDTH//2 - txt.get_width()//2, 360))

    pg.draw.rect(screen, WHITE, (SLIDER_X, SLIDER_Y, SLIDER_W, SLIDER_H), border_radius=4)

    # Slider handle
    pos = SLIDER_X if bet_slider_max == bet_slider_min else int(SLIDER_X + (bet_slider_value - bet_slider_min) / (bet_slider_max - bet_slider_min) * SLIDER_W)
    pg.draw.circle(screen, GOLD, (pos, SLIDER_Y + SLIDER_H//2), 14)

    val_txt = FONT.render(f"{bet_slider_value} chips", True, GOLD)
    screen.blit(val_txt, (WIDTH//2 - val_txt.get_width()//2, 440))

    preset_labels = ["Min", "Pot", "All In"]
    for i, val in enumerate(bet_slider_preset):
        bx = 300 + i*130
        pg.draw.rect(screen, GRAY, (bx, PRESET_BY, PRESET_BW, PRESET_BH), border_radius=8)
        label = FONT.render(f"{preset_labels[i]} ({val})", True, BLACK)
        screen.blit(label, (bx + PRESET_BW//2 - label.get_width()//2, PRESET_BY + PRESET_BH//2 - label.get_height()//2))

    pg.draw.rect(screen, GOLD, OK_BTN_RECT, border_radius=8)
    ok_txt = FONT.render("OK", True, BLACK)
    screen.blit(ok_txt, (OK_BTN_RECT.centerx - ok_txt.get_width()//2, OK_BTN_RECT.centery - ok_txt.get_height()//2))
    pg.draw.rect(screen, RED, CANCEL_BTN_RECT, border_radius=8)
    cancel_txt = FONT.render("Cancel", True, WHITE)
    screen.blit(cancel_txt, (CANCEL_BTN_RECT.centerx - cancel_txt.get_width()//2, CANCEL_BTN_RECT.centery - cancel_txt.get_height()//2))

def handle_bet_slider_event(mx, my):
    global bet_slider_value, waiting_for_action
    if SLIDER_Y <= my <= SLIDER_Y + SLIDER_H and SLIDER_X <= mx <= SLIDER_X + SLIDER_W:
        rel = (mx - SLIDER_X) / SLIDER_W
        bet_slider_value = int(bet_slider_min + rel * (bet_slider_max - bet_slider_min))

    # Preset buttons
    for i, val in enumerate(bet_slider_preset):
        bx = 300 + i*130
        if bx <= mx <= bx+PRESET_BW and PRESET_BY <= my <= PRESET_BY+PRESET_BH:
            bet_slider_value = val

    if OK_BTN_RECT.collidepoint(mx, my):
        bet_action(bet_slider_value)
        close_bet_slider()
        return True

    if CANCEL_BTN_RECT.collidepoint(mx, my):
        close_bet_slider()

    return False

def new_deck():
    global deck
    deck = Deck()
    choose_random_event()

def choose_random_event():
    global deck, game_event
    events = ["Cards are only 1 suit", "Cards are only 2 suits",
              "No face cards", "Only face cards", "War", "Normal",
              "Rankings are reversed", "Banned Rank", "Joker Wilds"]
    game_event = random.choice(events)

    if "suit" in game_event:
        suits = ['♠', '♥', '♦', '♣']
        suit = random.choice(suits)
        if "1" in game_event:
            deck.cards = [c for c in deck.cards if c.suit == suit]
            game_event = "Cards are only " + suit
        else:
            suit2 = random.choice([s for s in suits if s != suit])
            deck.cards = [c for c in deck.cards if c.suit == suit or c.suit == suit2]
            game_event = "Cards are only " + suit + " and " + suit2
    elif "face" in game_event:
        if "No" in game_event:
            deck.cards = [c for c in deck.cards if c.rank not in ['J', 'Q', 'K', 'A']]
        else:
            deck.cards = [c for c in deck.cards if c.rank in ['J', 'Q', 'K', 'A']]
    elif "Banned" in game_event:
        banned = random.choice(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'])
        deck.cards = [c for c in deck.cards if c.rank != banned]
        game_event = "No " + banned + "s"
    elif "Joker Wilds" in game_event:
        deck.cards.append(Card('JOKER', 'Red'))
        deck.cards.append(Card('JOKER', 'Black'))
        deck.cards.append(Card('JOKER', 'Red'))
        deck.cards.append(Card('JOKER', 'Black'))
        deck.cards.append(Card('JOKER', 'Red'))
        deck.cards.append(Card('JOKER', 'Black'))
        deck.shuffle()

def start_new_game():
    global players, community_cards, pot, round_stage, message, winner, dealer_position, player_contributions
    new_deck()

    player_contributions = [0, 0]

    # Check if this is a completely new game or a new hand
    if not players:
        players = [Player("Player"), Player("Opponent")]
    else:
        # Check for busted players and give them 500 chips if needed
        for p in players:
            if p.chips <= 0:
                p.chips = 500
                message = f"{p.name} rebought for 500 chips"

        # Reset player state for new hand but keep chips
        for p in players:
            p.active = True
            p.acted = False
            p.current_bet = 0
            p.hand = []
            p.all_in = False

    pot = 0
    round_stage = "preflop"
    winner = None

    dealer_position = (dealer_position + 1) % 2

    # Set blinds based on dealer position
    if dealer_position == 0:
        # Make sure players can afford the blinds
        players[0].chips -= min(5, players[0].chips)
        players[0].current_bet = min(5, players[0].chips)
        players[1].chips -= min(10, players[1].chips)
        players[1].current_bet = min(10, players[1].chips)
        message = "You are dealer (small blind)"
        player_contributions = [min(5, players[0].chips), min(10, players[1].chips)]
    else:
        players[1].chips -= min(5, players[1].chips)
        players[1].current_bet = min(5, players[1].chips)
        players[0].chips -= min(10, players[0].chips)
        players[0].current_bet = min(10, players[0].chips)
        message = "Opponent is dealer (small blind)"
        player_contributions = [min(10, players[0].chips), min(5, players[1].chips)]

    if players[0].chips == 0:
        players[0].all_in = True
    if players[1].chips == 0:
        players[1].all_in = True
        
    # Calculate pot
    pot = player_contributions[0] + player_contributions[1]

    # Deal
    for p in players:
        p.hand = deck.deal(2)

    community_cards.clear()
    update_buttons()
    draw()

def deal_community(n):
    global community_cards
    community_cards.extend(deck.deal(n))

def bet_action(amount):
    global pot, message, waiting_for_action, player_contributions
    player = players[0]
    if winner or player.chips < amount:
        return

    if amount == player.chips:
        player.all_in = True

    player.chips -= amount
    player.current_bet += amount
    pot += amount
    player_contributions[0] += amount
    player.acted = True
    players[1].acted = False

    update_buttons()
    waiting_for_action = False

def check_action():
    global message, pot, waiting_for_action, player_contributions
    opponent_bet = players[1].current_bet
    player_bet = players[0].current_bet

    if opponent_bet > player_bet:
        to_call = opponent_bet - player_bet
        call_amt = min(to_call, players[0].chips)

        if call_amt == players[0].chips:
            players[0].all_in = True

        players[0].chips -= call_amt
        players[0].current_bet += call_amt
        pot += call_amt
        player_contributions[0] += call_amt

    players[0].acted = True
    update_buttons()
    waiting_for_action = False

def fold_action():
    global message, winner, round_stage
    players[0].active = False
    players[0].acted = True
    winner = "Opponent"
    message = "You folded."
    round_stage = "showdown"
    players[1].chips += pot

def advance_round():
    global round_stage, message, winner
    if winner:
        return

    for p in players:
        p.acted = False
        p.current_bet = 0

    if round_stage == "preflop":
        deal_community(3)
        round_stage = "flop"
    elif round_stage == "flop":
        deal_community(1)
        round_stage = "turn"
    elif round_stage == "turn":
        deal_community(1)
        round_stage = "river"
    elif round_stage == "river":
        round_stage = "showdown"
        determine_winner()
    else:
        start_new_game()
    update_buttons()

def make_buttons():
    return [
        Button("Check", BTN_X, BTN_Y_START, BTN_W, BTN_H, GRAY, check_action),
        Button("Bet", BTN_X, BTN_Y_START + BTN_SPACING, BTN_W, BTN_H, GRAY, open_bet_slider),
        Button("Fold", BTN_X, BTN_Y_START + 2*BTN_SPACING, BTN_W, BTN_H, RED, fold_action),
        Button("Next Hand", WIDTH//2 - 75, HEIGHT//2 + 50, 150, 40, GOLD, start_new_game)
    ]

buttons = make_buttons()

def determine_winner():
    global winner, message, pot
    player_best = evaluate_hand(players[0].hand + community_cards)
    opponent_best = evaluate_hand(players[1].hand + community_cards)
    player_name = hand_name_from_rank(player_best[0])
    opponent_name = hand_name_from_rank(opponent_best[0])

    if not players[0].active:
        winner = "Opponent"
        players[1].chips += pot
        message = f"Winner: {winner} (Player folded)"
        return
    elif not players[1].active:
        winner = "Player"
        players[0].chips += pot
        message = f"Winner: {winner} (Opponent folded)"
        return

    if game_event == "War":
        p0_vals = sorted((RANK_TO_VALUE[c.rank] for c in (players[0].hand + community_cards)), reverse=True)
        p1_vals = sorted((RANK_TO_VALUE[c.rank] for c in (players[1].hand + community_cards)), reverse=True)

        for v0, v1 in zip(p0_vals, p1_vals):
            if v0 > v1:
                winner = "Player"
                players[0].chips += pot
                message = f"Winner (War): {winner} (high card {v0} vs {v1})"
                return
            if v1 > v0:
                winner = "Opponent"
                players[1].chips += pot
                message = f"Winner (War): {winner} (high card {v1} vs {v0})"
                return

        winner = "Tie"
        split = pot // 2
        players[0].chips += split
        players[1].chips += pot - split
        message = f"Winner: {winner} (War tie)"
        return
    elif game_event == "Rankings are reversed":
        player_best, opponent_best = opponent_best, player_best

    # Handle side pots when someone is all-in
    if players[0].all_in or players[1].all_in:
        min_contrib = min(player_contributions[0], player_contributions[1])
        main_pot = min_contrib * 2

        side_pot = pot - main_pot

        if player_best > opponent_best:
            winner = "Player"
            players[0].chips += main_pot
            # If player contributed more, they also get side pot
            if player_contributions[0] > player_contributions[1]:
                players[0].chips += side_pot
            else:
                players[1].chips += side_pot
        elif player_best < opponent_best:
            winner = "Opponent"
            players[1].chips += main_pot
            # If opponent contributed more, they also get side pot
            if player_contributions[1] > player_contributions[0]:
                players[1].chips += side_pot
            else:
                players[0].chips += side_pot
        else:  # Tie
            winner = "Tie"
            split = main_pot // 2
            players[0].chips += split
            players[1].chips += split

            if player_contributions[0] > player_contributions[1]:
                players[0].chips += side_pot
            else:
                players[1].chips += side_pot

        message = f"Winner: {winner} ({player_name} vs {opponent_name})"
        return

    if player_best > opponent_best:
        winner = "Player"
        players[0].chips += pot
    elif player_best < opponent_best:
        winner = "Opponent"
        players[1].chips += pot
    else:  # Tie
        winner = "Tie"
        split = pot // 2
        players[0].chips += split
        players[1].chips += split

    message = f"Winner: {winner} ({player_name} vs {opponent_name})"

def close_bet_slider():
    global show_bet_slider
    show_bet_slider = False

def draw():
    screen.fill(GREEN)
    pg.draw.ellipse(screen, (0, 100, 0), (80, 120, WIDTH-160, HEIGHT-240))

    pot_text = BIG_FONT.render(f"Pot: {pot}", True, GOLD)
    screen.blit(pot_text, (50, 110))

    ev_x, ev_y = 40, 200
    ev_w, ev_h = 200, 100
    pg.draw.rect(screen, (30, 30, 30), (ev_x, ev_y, ev_w, ev_h), border_radius=10)
    pg.draw.rect(screen, GOLD, (ev_x, ev_y, ev_w, ev_h), 2, border_radius=10)
    title_txt = FONT.render("Event", True, GOLD)
    screen.blit(title_txt, (ev_x + 12, ev_y + 8))

    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            test = cur + (" " if cur else "") + w
            if font.size(test)[0] <= max_width - 20:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    ev_lines = wrap_text(game_event, FONT, ev_w)
    for i, line in enumerate(ev_lines[:6]):
        line_surf = FONT.render(line, True, WHITE)
        screen.blit(line_surf, (ev_x + 12, ev_y + 36 + i * 22))

    draw_hand(community_cards, 260, 180)
    draw_info_box(INFOBOX_PLAYER_X, INFOBOX_PLAYER_Y, players[0])
    draw_info_box(INFOBOX_OPP_X, INFOBOX_OPP_Y, players[1])

    highlight = (winner == "Player" and round_stage == "showdown")
    draw_hand(players[0].hand, 400, 500, highlight)
    highlight = (winner == "Opponent" and round_stage == "showdown")
    hide = round_stage != "showdown"
    draw_hand(players[1].hand, 400, 30, highlight, hide)

    if round_stage == "showdown":
        msg_bg = pg.Surface((WIDTH, 80), pg.SRCALPHA)
        msg_bg.fill((0, 0, 0, 180))
        screen.blit(msg_bg, (0, HEIGHT // 2 - 40))
        msg_text = BIG_FONT.render(message, True, GOLD)
        screen.blit(msg_text, (WIDTH // 2 - msg_text.get_width() // 2, HEIGHT // 2 - msg_text.get_height() // 2))
        buttons[3].visible = True
    else:
        msg_text = FONT.render(message, True, WHITE)
        screen.blit(msg_text, (50, 140))
        buttons[3].visible = False

    # Show/hide buttons
    opponent_bet = players[1].current_bet
    player_bet = players[0].current_bet

    if round_stage != "showdown" and players[0].active and players[1].active:
        buttons[0].visible = True
        buttons[1].visible = True

        if opponent_bet > player_bet:
            buttons[0].text = f"Call ({opponent_bet - player_bet})"
            buttons[1].text = "Raise"
        else:
            buttons[0].text = "Check"
            buttons[1].text = "Bet"

        buttons[2].visible = True
    else:
        buttons[0].visible = False
        buttons[1].visible = False
        buttons[2].visible = False

    for b in buttons:
        if getattr(b, "visible", True):
            b.draw(screen)

    if show_bet_slider:
        draw_bet_slider()

    pg.display.flip()

def opponent_action():
    global waiting_for_action, pot, message, winner, round_stage, player_contributions

    opp = players[1]
    player = players[0]
    to_call = player.current_bet - opp.current_bet
    bet_amount = 0

    print(f"\n--- AI Opponent's turn ---")
    print(f"Your bet: {player.current_bet}, AI bet: {opp.current_bet}, To call: {to_call}")

    if not crf_model:
        raise Exception("CRF model not loaded. Cannot make AI decision.")
    if opp.all_in:
        opp.acted = True
        waiting_for_action = True
        return

    # Extract features for the AI decision
    round_idx = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}[round_stage]

    deck_cards = [deck_card for deck_card in deck.cards]

    # Calculate win probability using Monte Carlo simulation
    mc_trials = 100
    win_prob = mc_win_prob(opp.hand, community_cards, deck_cards, opp_count=1, trials=mc_trials)
    if game_event == "Rankings are reversed":
        win_prob = 1.0 - win_prob
    win_prob_int = int(win_prob * 1000)

    # Moderate the AI's aggressiveness based on chip stack ratio
    stack_ratio = opp.chips / max(1, player.chips)
    aggression_dampener = min(1.0, 0.5 + stack_ratio/2)

    # Get board texture features directly from CRF functions
    texture_feats = board_texture(community_cards)
    draw_feats = draws_flags(opp.hand, community_cards)

    features = {
        "round": round_idx,
        "to_call": min(to_call, 2000),
        "pot": min(pot, 5000),
        "hero_stack": min(opp.chips, 5000),
        "opp_stack": min(player.chips, 5000),
        "in_position": 1 if (round_stage == "preflop" and dealer_position == 1) or
                          (round_stage != "preflop" and dealer_position == 0) else 0,
        "pot_odds_x100": int((to_call / max(1, pot + to_call)) * 100),
        "winp_x1000": int(win_prob_int * aggression_dampener),
        **texture_feats,
        **draw_feats,
        **event_to_features(game_event),
    }

    print(f"Win probability: {win_prob:.2f}")

    # Get AI decision
    action = crf_model.predict_single([features])[0]
    print(f"AI chooses: {action}")

    if action == "FOLD":
        opp.active = False
        opp.acted = True
        message = "AI opponent folds"
        winner = "Player"
        round_stage = "showdown"

    elif action == "CHECK" and to_call == 0:
        message = "AI opponent checks"
        opp.acted = True

    elif action == "CALL" or action == "CHECK":
        call_amt = min(to_call, opp.chips)

        if call_amt == opp.chips:
            opp.all_in = True

        opp.chips -= call_amt
        opp.current_bet += call_amt
        pot += call_amt
        player_contributions[1] += call_amt
        opp.acted = True

        if opp.all_in:
            message = f"AI opponent calls all-in with {call_amt}"
        else:
            message = f"AI opponent calls {call_amt}"

    elif action in ["BET_MIN", "BET_QUARTER", "BET_HALF", "BET_THREE_QUARTERS", "BET_POT", "ALLIN"]:
        # Calculate bet sizes
        min_bet = max(10, to_call + 10)
        quarter_pot = max(10, int(pot * 0.25))
        half_pot = max(10, int(pot * 0.5))
        three_quarter_pot = max(10, int(pot * 0.75))
        pot_bet = max(10, pot)

        quarter_pot = max(min_bet, quarter_pot)
        half_pot = max(min_bet, half_pot)
        three_quarter_pot = max(min_bet, three_quarter_pot)
        pot_bet = max(min_bet, pot_bet)

        # Check if AI can't afford the minimum raise/bet
        if to_call > 0 and opp.chips <= to_call:
            # AI can't even call, so go all-in
            call_amt = opp.chips
            opp.all_in = True
            opp.chips = 0
            pot += call_amt
            opp.current_bet += call_amt
            player_contributions[1] += call_amt
            opp.acted = True
            message = f"AI opponent calls all-in with {call_amt}"
        elif to_call == 0 and opp.chips < 10:
            # AI can't afford minimum bet, so check
            message = "AI opponent checks"
            opp.acted = True
        else:
            # Choose bet size based on action
            if action == "BET_MIN":
                bet_amount = min(min_bet, opp.chips, player.chips + player.current_bet)
            elif action == "BET_QUARTER":
                bet_amount = min(quarter_pot, opp.chips, player.chips + player.current_bet)
            elif action == "BET_HALF":
                bet_amount = min(half_pot, opp.chips, player.chips + player.current_bet)
            elif action == "BET_THREE_QUARTERS":
                bet_amount = min(three_quarter_pot, opp.chips, player.chips + player.current_bet)
            elif action == "BET_POT":
                bet_amount = min(pot_bet, opp.chips, player.chips + player.current_bet)
            else:
                bet_amount = opp.chips

            if to_call > 0:  # Raising
                min_raise = to_call + 10

                if opp.chips < to_call:
                    # Not enough to call, go all-in
                    call_amt = opp.chips
                    opp.chips = 0
                    pot += call_amt
                    opp.current_bet += call_amt
                    player_contributions[1] += call_amt
                    message = f"AI opponent calls all-in with {call_amt}"
                elif opp.chips < min_raise:
                    # Enough to call but not raise, just call
                    call_amt = to_call
                    opp.chips -= call_amt
                    pot += call_amt
                    opp.current_bet += call_amt
                    player_contributions[1] += call_amt
                    message = f"AI opponent calls {call_amt}"
                elif bet_amount < player.current_bet:
                    # Bet is smaller than min raise
                    call_amt = to_call
                    opp.chips -= call_amt
                    pot += call_amt
                    opp.current_bet += call_amt
                    player_contributions[1] += call_amt
                    message = f"AI opponent calls {call_amt}"
                else:
                    if bet_amount >= opp.chips + opp.current_bet:
                        # All-in raise
                        additional_chips = opp.chips
                        opp.current_bet += additional_chips
                        pot += additional_chips
                        player_contributions[1] += additional_chips
                        opp.chips = 0
                        opp.all_in = True
                        message = f"AI opponent raises all-in to {opp.current_bet}"
                    else:
                        # Normal raise
                        additional_chips = bet_amount - opp.current_bet
                        opp.chips -= additional_chips
                        pot += additional_chips
                        player_contributions[1] += additional_chips
                        opp.current_bet = bet_amount
                        message = f"AI opponent raises to {bet_amount}"
            else:  # Betting
                if bet_amount < min_bet and bet_amount < opp.chips:
                    # If less than min bet and AI can afford min bet, force min bet
                    bet_amount = min_bet

                if bet_amount >= opp.chips:
                    bet_amount = opp.chips
                    opp.all_in = True
                    message = f"AI opponent bets all-in ({bet_amount})"
                else:
                    message = f"AI opponent bets {bet_amount}"

                opp.chips -= bet_amount
                pot += bet_amount
                opp.current_bet = bet_amount
                player_contributions[1] += bet_amount

        opp.acted = True

        # Mark player as not acted since they need to respond to a raise/bet
        if to_call == 0 or bet_amount > player.current_bet:
            if player.chips > 0:
                players[0].acted = False

    update_buttons()
    waiting_for_action = True

def get_action_order():
    if round_stage == "preflop":
        return [players[0], players[1]] if dealer_position == 0 else [players[1], players[0]]
    else:
        return [players[1], players[0]] if dealer_position == 0 else [players[0], players[1]]

async def game_loop_async():
    global slider_dragging, bet_slider_value, winner, round_stage

    all_in_players = [p for p in players if p.all_in and p.active]
    if len(all_in_players) > 0:
        if all(p.acted or p.all_in for p in players if p.active):
            while round_stage not in ["showdown"]:
                advance_round()
                draw()
                await asyncio.sleep(0)
            return

    while not winner and round_stage != "showdown":
        action_order = get_action_order()

        # First player to act
        first_player = action_order[0]
        if not first_player.acted:
            if first_player == players[0]:
                await handle_player_action()
            else:
                opponent_action()
            update_buttons()
            draw()

            if any(p.all_in for p in players if p.active):
                if all(p.acted or p.all_in for p in players if p.active):
                    while round_stage not in ["showdown"]:
                        advance_round()
                        draw()
                        await asyncio.sleep(0)
                    return

        if check_round_over():
            advance_round()
            draw()
            await asyncio.sleep(0)
            continue

        if winner or round_stage == "showdown" or not (players[0].active and players[1].active):
            continue

        # Second player to act
        second_player = action_order[1]
        if not second_player.acted:
            if second_player == players[0]:
                await handle_player_action()
            else:
                opponent_action()
            update_buttons()
            draw()

            if any(p.all_in for p in players if p.active):
                if all(p.acted or p.all_in for p in players if p.active):
                    while round_stage not in ["showdown"]:
                        advance_round()
                        draw()
                        await asyncio.sleep(0)
                    return

        if check_round_over():
            advance_round()
            draw()
            await asyncio.sleep(0)

        if not players[0].active or not players[1].active:
            determine_winner()
            draw()

        # yield to event loop to keep UI responsive
        await asyncio.sleep(0)

    return

def check_round_over():
    active_players = [p for p in players if p.active]

    # If there's only one active player, the round is over
    if len(active_players) <= 1:
        return True

    for p in active_players:
        if not p.acted and not p.all_in:
            return False

    max_bet = max(p.current_bet for p in active_players)
    for p in active_players:
        if p.current_bet < max_bet and not p.all_in:
            return False

    return True

async def handle_player_action():
    global slider_dragging, bet_slider_value, waiting_for_action
    waiting_for_action = True

    if players[0].acted:
        waiting_for_action = False
        return

    while waiting_for_action:
        draw()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return
            elif event.type == pg.MOUSEBUTTONDOWN:
                if show_bet_slider:
                    mx, my = event.pos
                    handle_bet_slider_mouse_down(mx, my)
                else:
                    for b in buttons:
                        if getattr(b, "visible", True) and b.rect.collidepoint(event.pos):
                            b.click()
                            if b.text in ["Check", "Fold"] or b.text.startswith("Call ("):
                                waiting_for_action = False
            elif event.type == pg.MOUSEBUTTONUP:
                slider_dragging = False
            elif event.type == pg.MOUSEMOTION:
                if show_bet_slider and slider_dragging:
                    mx, my = event.pos
                    mx = max(SLIDER_X, min(mx, SLIDER_X + SLIDER_W))
                    rel = (mx - SLIDER_X) / SLIDER_W
                    bet_slider_value = int(bet_slider_min + rel * (bet_slider_max - bet_slider_min))
        await asyncio.sleep(0)

def handle_bet_slider_mouse_down(mx, my):
    global slider_dragging, bet_slider_value, waiting_for_action
    if SLIDER_X <= mx <= SLIDER_X + SLIDER_W and SLIDER_Y - 10 <= my <= SLIDER_Y + SLIDER_H + 10:
        rel = (mx - SLIDER_X) / SLIDER_W
        bet_slider_value = int(bet_slider_min + rel * (bet_slider_max - bet_slider_min))
        slider_dragging = True
        return

    if handle_bet_slider_event(mx, my):
        waiting_for_action = False

async def main():
    start_new_game()

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                for b in buttons:
                    if getattr(b, "visible", True) and b.rect.collidepoint(event.pos):
                        b.click()

        if round_stage != "showdown":
            await game_loop_async()
        else:
            draw()

        await asyncio.sleep(0)

    pg.quit()

if __name__ == "__main__":
    asyncio.run(main())
