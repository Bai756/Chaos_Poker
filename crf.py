import random
import argparse
import pickle
from collections import Counter
from itertools import combinations
import math

# -------------------------------
# Deck & hand evaluation
# -------------------------------

SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
RANK_TO_VAL = {r: i for i, r in enumerate(RANKS, start=2)}
VAL_TO_RANK = {v: r for r, v in RANK_TO_VAL.items()}

def new_deck():
    return [r + s for s in SUITS for r in RANKS]

def deal(cards, n):
    return [cards.pop() for _ in range(n)]

def card_val(card):
    return RANK_TO_VAL[card[0]]

def card_suit(card):
    return card[1]

def is_straight(vals):
    """Return top straight value or 0."""
    uniq = sorted(set(vals))
    # wheel
    if set([14, 5, 4, 3, 2]).issubset(uniq):
        return 5
    for i in range(len(uniq)-4):
        if uniq[i+4] - uniq[i] == 4 and uniq[i:i+5] == list(range(uniq[i], uniq[i]+5)):
            return uniq[i+4]
    return 0

def evaluate_7(cards):
    """
    Returns a hand-rank tuple comparable with Python ordering.
    Higher tuple is better.
    Ranking: (8 sf, 7 quads, 6 full, 5 flush, 4 straight, 3 trips, 2 two pair, 1 pair, 0 high)
    tie-breakers appended in descending order.
    """
    vals = sorted([RANK_TO_VAL[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]

    # flush?
    suit_cnt = Counter(suits)
    flush_suit = next((s for s, k in suit_cnt.items() if k >= 5), None)
    flush_vals = []
    if flush_suit:
        flush_vals = sorted([RANK_TO_VAL[c[0]] for c in cards if c[1] == flush_suit], reverse=True)
        top5_flush = flush_vals[:5]
    else:
        top5_flush = []

    # straight?
    straight_top = is_straight(vals)

    # straight flush?
    sf_top = 0
    if flush_suit:
        sf_top = is_straight(flush_vals)

    if sf_top:
        return (8, sf_top)

    cnt = Counter(vals)
    groups = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    # quads, trips, pairs separated
    four = [v for v, c in cnt.items() if c == 4]
    trips = [v for v, c in cnt.items() if c == 3]
    pairs = [v for v, c in cnt.items() if c == 2]
    singles = sorted([v for v, c in cnt.items() if c == 1], reverse=True)

    if four:
        k = max(four)
        kickers = [v for v in vals if v != k][:1]
        return (7, k, kickers[0])

    if trips and pairs:
        t = max(trips)
        p = max(pairs)
        return (6, t, p)

    if flush_suit:
        return (5, *top5_flush)

    if straight_top:
        return (4, straight_top)

    if trips:
        t = max(trips)
        kicks = [v for v in vals if v != t][:2]
        return (3, t, *kicks)

    if len(pairs) >= 2:
        p1, p2 = sorted(pairs, reverse=True)[:2]
        kick = max([v for v in vals if v not in (p1, p2)])
        return (2, p1, p2, kick)

    if pairs:
        p = max(pairs)
        kicks = [v for v in vals if v != p][:3]
        return (1, p, *kicks)

    return (0, *vals[:5])


# -------------------------------
# State features
# -------------------------------

def board_texture(board):
    """Simple texture features."""
    vals = [RANK_TO_VAL[c[0]] for c in board]
    suits = [c[1] for c in board]
    suit_cnt = Counter(suits)
    paired = any(v >= 2 for v in Counter(vals).values())

    # flush draw: 4 of a suit on board+hand will be handled elsewhere; here, only board
    max_suit_on_board = max(suit_cnt.values()) if suits else 0

    # straighty board (gives a coarse indicator)
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
    suits = [c[1] for c in allc]
    suit_cnt = Counter(suits)
    flush_draw = any(v == 4 for v in suit_cnt.values())

    vals = sorted(set([RANK_TO_VAL[c[0]] for c in allc]))
    # crude straight draw: any 4-gap window with length <= 5 and at least 4 cards
    ooe = 0
    for comb in combinations(vals, 4):
        if comb[-1] - comb[0] <= 5:
            ooe = 1
            break
    return {"flush_draw": int(flush_draw), "straight_draw": int(ooe)}


# -------------------------------
# Monte Carlo win probability
# -------------------------------

def mc_win_prob(hero_hand, board, deck, opp_count=1, trials=200):
    """
    Estimate hero's win probability vs opp_count random opponents.
    Heads-up default (opp_count=1).
    """
    if trials <= 0:
        return 0.0

    wins = ties = 0
    used = set(hero_hand + board)
    for _ in range(trials):
        cards = [c for c in deck if c not in hero_hand and c not in board]
        random.shuffle(cards)

        # Opponents' hole
        opp_holes = []
        idx = 0
        for _o in range(opp_count):
            opp_holes.append([cards[idx], cards[idx+1]])
            idx += 2

        # Complete board to 5
        need = 5 - len(board)
        draw_board = board + cards[idx:idx+need]
        idx += need

        hero_eval = evaluate_7(hero_hand + draw_board)
        opp_evals = [evaluate_7(h + draw_board) for h in opp_holes]

        best_opp = max(opp_evals)
        if hero_eval > best_opp:
            wins += 1
        elif hero_eval == best_opp:
            ties += 1

    return (wins + 0.5 * ties) / trials


# -------------------------------
# CRF utilities
# -------------------------------

# To avoid importing heavy libs unless needed:
def _import_crf():
    from sklearn_crfsuite import CRF
    return CRF

ACTIONS = ["FOLD", "CHECK_CALL", "BET_HALF", "BET_POT", "ALLIN"]

def discretize_bet_sizes(pot, stack, to_call):
    """
    Return concrete amounts for BET_HALF, BET_POT, ALLIN given pot and stack.
    For CHECK_CALL, amount is min(to_call, stack).
    """
    half = max(1, int(0.5 * pot))
    pot_bet = max(1, int(1.0 * pot))
    allin = stack
    # Ensure legal after calling: in no-limit, raise replaces to current amount.
    return max(0, to_call), half, pot_bet, allin

def pot_odds(to_call, pot):
    return to_call / max(1, (pot + to_call))

def features_for_state(round_idx, hero_hand, board, pot, hero_stack, opp_stack,
                       to_call, in_position, opp_count=1, mc_samples=120):
    """Build a feature dict for CRF."""
    deck = new_deck()
    for c in hero_hand + board:
        deck.remove(c)

    winp = mc_win_prob(hero_hand, board, deck, opp_count=opp_count, trials=mc_samples)

    feats = {
        "round": round_idx,                      # 0,1,2,3 (pre, flop, turn, river)
        "to_call": min(to_call, 2000),          # clipped
        "pot": min(pot, 5000),
        "hero_stack": min(hero_stack, 5000),
        "opp_stack": min(opp_stack, 5000),
        "in_position": int(in_position),
        "pot_odds_x100": int(pot_odds(to_call, pot) * 100),
        "winp_x1000": int(winp * 1000),
        **board_texture(board),
        **draws_flags(hero_hand, board),
    }
    # CRF expects categorical or numeric; sklearn-crfsuite handles ints fine.
    return feats

def sample_action_from_policy(features, faced_bet):
    """
    Heuristic policy used to generate synthetic labels (adds randomness).
    Returns one of ACTIONS.
    """
    winp = features["winp_x1000"] / 1000.0
    odds = features["pot_odds_x100"] / 100.0
    rnd = random.random()

    # Basic EV-ish gating
    if faced_bet:
        # Bluff-catch / fold
        if winp < odds * 0.9 and rnd > 0.2:
            return "FOLD"
        if winp < 0.25 and rnd > 0.5:
            return "FOLD"
        # Mix calls/raises
        if winp > 0.65 and rnd < 0.35:
            return "BET_POT"
        if winp > 0.55 and rnd < 0.6:
            return "BET_HALF"
        return "CHECK_CALL"
    else:
        # No bet faced: check or value bet/bluff
        if winp > 0.7:
            return "BET_POT" if rnd < 0.6 else "BET_HALF"
        if winp > 0.55:
            return "BET_HALF" if rnd < 0.6 else "CHECK_CALL"
        # Occasionally bluff semi-draws
        if features["flush_draw"] or features["straight_draw"]:
            return "BET_HALF" if rnd < 0.35 else "CHECK_CALL"
        return "CHECK_CALL"


# -------------------------------
# Hand simulation and sequence building
# -------------------------------

def play_synthetic_hand(mc_samples=80):
    """
    Simulate a heads-up hand (very simplified betting to 1 action per street).
    Returns a list of (feature_dict, label_action) for rounds actually played.
    """
    deck = new_deck()
    random.shuffle(deck)
    hero = deal(deck, 2)
    opp = deal(deck, 2)

    # blinds
    pot = 15
    hero_stack = 990
    opp_stack = 995
    to_call_hero = 5   # hero posted small blind 5, opp big blind 10
    to_call_opp = 0

    # Action order: preflop (hero first as SB), postflop (IP/OOP swap each street)
    board = []
    seq_feats = []
    seq_labels = []

    def one_round(round_idx, first_is_hero):
        nonlocal pot, hero_stack, opp_stack, to_call_hero, to_call_opp, board

        # FIRST player decision
        if first_is_hero:
            faced_bet = to_call_hero > 0
            feats = features_for_state(round_idx, hero, board, pot, hero_stack, opp_stack,
                                       to_call_hero, in_position=0, opp_count=1, mc_samples=mc_samples)
            label = sample_action_from_policy(feats, faced_bet)
            seq_feats.append(feats); seq_labels.append(label)
            pot, hero_stack, opp_stack, to_call_hero, to_call_opp, ended = apply_action(
                label, pot, hero_stack, opp_stack, to_call_hero, to_call_opp)
            if ended:
                return True  # hand ended
        else:
            # Opp first: generate their features, then their label
            faced_bet = to_call_opp > 0
            feats_opp = features_for_state(round_idx, opp, board, pot, opp_stack, hero_stack,
                                           to_call_opp, in_position=0, opp_count=1, mc_samples=mc_samples//2)
            opp_label = sample_action_from_policy(feats_opp, faced_bet)
            pot, opp_stack, hero_stack, to_call_opp, to_call_hero, ended = apply_action(
                opp_label, pot, opp_stack, hero_stack, to_call_opp, to_call_hero)
            if ended:
                return True

            # Then hero
            faced_bet = to_call_hero > 0
            feats = features_for_state(round_idx, hero, board, pot, hero_stack, opp_stack,
                                       to_call_hero, in_position=1, opp_count=1, mc_samples=mc_samples)
            label = sample_action_from_policy(feats, faced_bet)
            seq_feats.append(feats); seq_labels.append(label)
            pot, hero_stack, opp_stack, to_call_hero, to_call_opp, ended = apply_action(
                label, pot, hero_stack, opp_stack, to_call_hero, to_call_opp)
            if ended:
                return True

        # SECOND player decision (mirror above) if needed
        if first_is_hero:
            faced_bet = to_call_opp > 0
            feats_opp = features_for_state(round_idx, opp, board, pot, opp_stack, hero_stack,
                                           to_call_opp, in_position=1, opp_count=1, mc_samples=mc_samples//2)
            opp_label = sample_action_from_policy(feats_opp, faced_bet)
            pot, opp_stack, hero_stack, to_call_opp, to_call_hero, ended = apply_action(
                opp_label, pot, opp_stack, hero_stack, to_call_opp, to_call_hero)
            if ended:
                return True
        else:
            faced_bet = to_call_opp > 0
            # Already acted both; nothing to do if bets matched
            pass

        return False

    # Preflop (hero first)
    ended = one_round(0, first_is_hero=True)
    if ended:
        return seq_feats, seq_labels

    # Flop
    board.extend(deal(deck, 3))
    to_call_hero = to_call_opp = 0
    ended = one_round(1, first_is_hero=False)
    if ended:
        return seq_feats, seq_labels

    # Turn
    board.extend(deal(deck, 1))
    to_call_hero = to_call_opp = 0
    ended = one_round(2, first_is_hero=False)
    if ended:
        return seq_feats, seq_labels

    # River
    board.extend(deal(deck, 1))
    to_call_hero = to_call_opp = 0
    ended = one_round(3, first_is_hero=False)
    return seq_feats, seq_labels


def apply_action(label, pot, actor_stack, other_stack, to_call_actor, to_call_other):
    """
    Resolves a single action label for the acting player.
    Returns updated (pot, actor_stack, other_stack, to_call_actor, to_call_other, ended)
    """
    call_amt, half, pot_bet, allin = discretize_bet_sizes(pot, actor_stack, to_call_actor)

    if label == "FOLD":
        # Hand ends immediately; we mark ended=True.
        # We won't allocate pot to winner here because this is just for training sequences.
        return pot, actor_stack, other_stack, to_call_actor, to_call_other, True

    if label == "CHECK_CALL":
        pay = min(call_amt, actor_stack)
        actor_stack -= pay
        pot += pay
        # After calling/matching, set both to_call to 0 (bets matched)
        to_call_actor = 0
        to_call_other = 0
        return pot, actor_stack, other_stack, to_call_actor, to_call_other, False

    if label in ("BET_HALF", "BET_POT", "ALLIN"):
        target = {"BET_HALF": half, "BET_POT": pot_bet, "ALLIN": allin}[label]
        # Bet/raise amount is amount OVER current actor contribution, but we only track to_call.
        # For simplicity we make this "new bet to" equal to target, limited by stack.
        bet_amt = min(target, actor_stack)
        actor_stack -= bet_amt
        pot += bet_amt
        # Now the other player must match this bet
        to_call_other = bet_amt
        to_call_actor = 0
        return pot, actor_stack, other_stack, to_call_actor, to_call_other, False

    # Fallback to check if action unrecognized
    return pot, actor_stack, other_stack, to_call_actor, to_call_other, False


# -------------------------------
# Training & inference
# -------------------------------

def simulate_sequences(n_hands, mc_samples=60):
    X, y = [], []
    for _ in range(n_hands):
        feats, labels = play_synthetic_hand(mc_samples=mc_samples)
        if feats and labels:
            X.append(feats)
            y.append(labels)
    return X, y

def train_crf(X, y):
    CRF = _import_crf()
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        all_possible_transitions=True
    )
    crf.fit(X, y)
    return crf

def predict_actions(crf, feats_seq):
    return crf.predict_single(feats_seq)


# -------------------------------
# Human vs CRF CLI (very simple)
# -------------------------------

def human_vs_crf(model_path):
    with open(model_path, "rb") as f:
        crf = pickle.load(f)

    while True:
        print("\n=== New Hand ===")
        deck = new_deck()
        random.shuffle(deck)
        hero = deal(deck, 2)
        opp = deal(deck, 2)
        board = []

        hero_stack = 990
        opp_stack = 995
        pot = 15
        to_call_hero = 5
        to_call_opp = 0

        # Preflop (hero first)
        ended = cli_round(crf, 0, True, hero, opp, board, pot, hero_stack, opp_stack, to_call_hero, to_call_opp, deck)
        if ended:
            continue

        # Flop
        board.extend(deal(deck, 3))
        to_call_hero = to_call_opp = 0
        ended = cli_round(crf, 1, False, hero, opp, board, pot, hero_stack, opp_stack, to_call_hero, to_call_opp, deck)
        if ended:
            continue

        # Turn
        board.extend(deal(deck, 1))
        to_call_hero = to_call_opp = 0
        ended = cli_round(crf, 2, False, hero, opp, board, pot, hero_stack, opp_stack, to_call_hero, to_call_opp, deck)
        if ended:
            continue

        # River
        board.extend(deal(deck, 1))
        to_call_hero = to_call_opp = 0
        cli_round(crf, 3, False, hero, opp, board, pot, hero_stack, opp_stack, to_call_hero, to_call_opp, deck, final_showdown=True)


def cli_round(crf, round_idx, hero_first, hero, opp, board, pot, hero_stack, opp_stack, to_call_hero, to_call_opp, deck, final_showdown=False):
    """
    One street: CRF acts for the AI, you act for yourself.
    We only do up to one action each (keeps the loop minimal).
    """
    def show_state():
        print(f"\n-- Round {round_idx} ({['Preflop','Flop','Turn','River'][round_idx]}) --")
        print(f"Your hand: {hero[0]} {hero[1]}")
        print(f"Board: {' '.join(board) if board else '(none)'}")
        print(f"Pot: {pot} | Your stack: {hero_stack}, Opp stack: {opp_stack}")
        print(f"To call: {to_call_hero}")

    def crf_decide_for(whose_hand, my_stack, opp_stack, to_call, in_pos):
        feats = features_for_state(round_idx, whose_hand, board, pot, my_stack, opp_stack, to_call,
                                   in_position=in_pos, opp_count=1, mc_samples=120)
        # We need a sequence—even if length 1—for CRF predict
        return crf.predict_single([feats])[0], feats

    # First actor
    show_state()
    if hero_first:
        # Human acts first
        act, pot2, hs2, os2, tch2, tco2, ended = cli_human_action(
            pot, hero_stack, opp_stack, to_call_hero, to_call_opp
        )
        pot, hero_stack, opp_stack, to_call_hero, to_call_opp = pot2, hs2, os2, tch2, tco2
        if ended:
            print("Hand ended (you folded or all-in resolved).")
            return True

        # CRF acts second
        label, _ = crf_decide_for(opp, opp_stack, hero_stack, to_call_opp, in_pos=1)
        pot, opp_stack, hero_stack, to_call_opp, to_call_hero, ended = apply_action(
            label, pot, opp_stack, hero_stack, to_call_opp, to_call_hero
        )
        print(f"CRF action: {label}")
        if ended:
            print("CRF folded. You win the pot.")
            return True
    else:
        # CRF acts first
        label, _ = crf_decide_for(opp, opp_stack, hero_stack, to_call_opp, in_pos=0)
        pot, opp_stack, hero_stack, to_call_opp, to_call_hero, ended = apply_action(
            label, pot, opp_stack, hero_stack, to_call_opp, to_call_hero
        )
        print(f"CRF action: {label}")
        if ended:
            print("CRF folded. You win the pot.")
            return True

        # Human acts second
        show_state()
        act, pot2, hs2, os2, tch2, tco2, ended = cli_human_action(
            pot, hero_stack, opp_stack, to_call_hero, to_call_opp
        )
        pot, hero_stack, opp_stack, to_call_hero, to_call_opp = pot2, hs2, os2, tch2, tco2
        if ended:
            print("Hand ended (you folded or all-in resolved).")
            return True

    if final_showdown:
        # Reveal & evaluate
        # Complete board already has 5 cards at river
        hero_eval = evaluate_7(hero + board)
        opp_eval = evaluate_7(opp + board)
        print(f"\nSHOWDOWN: You {hero} | Opp {opp} | Board {board}")
        if hero_eval > opp_eval:
            print("You win.")
        elif hero_eval < opp_eval:
            print("CRF wins.")
        else:
            print("Tie.")
        return True

    # move to next street
    return False


def cli_human_action(pot, hero_stack, opp_stack, to_call_hero, to_call_opp):
    while True:
        if to_call_hero > 0:
            prompt = f"[fold/call/half/pot/allin] (to call {to_call_hero}): "
        else:
            prompt = f"[check/half/pot/allin]: "
        cmd = input(prompt).strip().lower()

        call_amt, half, pot_bet, allin = discretize_bet_sizes(pot, hero_stack, to_call_hero)

        if cmd == "fold":
            return "fold", pot, hero_stack, opp_stack, to_call_hero, to_call_opp, True

        if cmd == "call":
            pay = min(call_amt, hero_stack)
            hero_stack -= pay
            pot += pay
            to_call_hero = 0
            to_call_opp = 0
            return "call", pot, hero_stack, opp_stack, to_call_hero, to_call_opp, False

        if cmd == "check" and to_call_hero == 0:
            return "check", pot, hero_stack, opp_stack, to_call_hero, to_call_opp, False

        if cmd == "half":
            amt = min(half, hero_stack)
            hero_stack -= amt
            pot += amt
            to_call_opp = amt
            to_call_hero = 0
            return "bet_half", pot, hero_stack, opp_stack, to_call_hero, to_call_opp, False

        if cmd == "pot":
            amt = min(pot_bet, hero_stack)
            hero_stack -= amt
            pot += amt
            to_call_opp = amt
            to_call_hero = 0
            return "bet_pot", pot, hero_stack, opp_stack, to_call_hero, to_call_opp, False

        if cmd == "allin":
            amt = hero_stack
            hero_stack = 0
            pot += amt
            to_call_opp = amt
            to_call_hero = 0
            return "allin", pot, hero_stack, opp_stack, to_call_hero, to_call_opp, False

        print("Invalid input.")


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="CRF-based Heads-Up Hold'em (7-card showdown)")
    ap.add_argument("--simulate", type=int, default=0, help="Generate N synthetic hands for training")
    ap.add_argument("--train", type=str, default="", help="Path to save trained CRF model (.pkl)")
    ap.add_argument("--model", type=str, default="", help="Path to load trained model for play")
    ap.add_argument("--play", action="store_true", help="Play against the trained CRF")
    args = ap.parse_args()

    if args.simulate > 0:
        print(f"Simulating {args.simulate} hands...")
        X, y = simulate_sequences(args.simulate, mc_samples=50)
        print(f"Got {len(X)} sequences with average length ~{sum(len(s) for s in X)/max(1,len(X)):.2f}")
        # store temp dataset next to model if training requested
        if args.train:
            tmp = args.train + ".data.pkl"
            with open(tmp, "wb") as f:
                pickle.dump((X, y), f)
            print(f"Saved training data to {tmp}")

    if args.train:
        try:
            with open(args.train + ".data.pkl", "rb") as f:
                X, y = pickle.load(f)
        except Exception:
            print("No cached data found. Simulating 15k by default...")
            X, y = simulate_sequences(15000, mc_samples=50)
        print("Training CRF...")
        crf = train_crf(X, y)
        with open(args.train, "wb") as f:
            pickle.dump(crf, f)
        print(f"Saved CRF model to {args.train}")

    if args.play:
        model_path = args.model or args.train
        if not model_path:
            raise SystemExit("Provide --model <path> (or use --train and then --play).")
        human_vs_crf(model_path)


if __name__ == "__main__":
    main()
