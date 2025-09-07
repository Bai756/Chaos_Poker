import random
import pickle
import os
from collections import Counter
from itertools import combinations
from base import evaluate_hand, Card, SUITS, RANKS
from crf import LinearChainCRF


RANK_TO_VAL = {r: i for i, r in enumerate(RANKS, start=2)}
VAL_TO_RANK = {v: r for r, v in RANK_TO_VAL.items()}
ACTIONS = ["FOLD", "CHECK", "CALL", "BET_MIN", "BET_QUARTER", "BET_HALF", "BET_THREE_QUARTERS", "BET_POT", "ALLIN"]


def new_deck():
    return [Card(r, s) for s in SUITS for r in RANKS]

def deal(cards, n):
    return [cards.pop() for _ in range(n)]

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

def discretize_bet_sizes(pot, stack, to_call):
    min_bet = max(10, to_call + 10)
    quarter = max(1, int(0.25 * pot))
    half = max(1, int(0.5 * pot))
    three_quarters = max(1, int(0.75 * pot))
    pot_bet = max(1, int(1.0 * pot))
    allin = stack
    return max(0, to_call), min_bet, quarter, half, three_quarters, pot_bet, allin

def pot_odds(to_call, pot):
    return to_call / max(1, (pot + to_call))

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
        "ev_only_odds": int("only odds" in ev),
        "ev_only_evens": int("only evens" in ev),
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

def features_for_state(round_idx, hero_hand, board, pot, hero_stack, opp_stack,
                       to_call, in_position, opp_count=1, mc_samples=120, event=None):
    """Build a feature dict for CRF."""
    deck_cards = new_deck()
    # Remove cards that are already in play
    used_cards = hero_hand + board
    
    winp = mc_win_prob(hero_hand, board, deck_cards, opp_count=opp_count, trials=mc_samples)

    feats = {
        "round": round_idx,  # 0,1,2,3 (pre, flop, turn, river)
        "to_call": min(to_call, 2000),  # clipped
        "pot": min(pot, 5000),
        "hero_stack": min(hero_stack, 5000),
        "opp_stack": min(opp_stack, 5000),
        "in_position": int(in_position),
        "pot_odds_x100": int(pot_odds(to_call, pot) * 100),
        "winp_x1000": int(winp * 1000),
        **board_texture(board),
        **draws_flags(hero_hand, board),
        **event_to_features(event),
    }
    return feats

def sample_action_from_policy(features, faced_bet):
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
        
        # Mix calls/raises based on hand strength
        if winp > 0.75:
            if rnd < 0.4:
                return "BET_POT"
            elif rnd < 0.7:
                return "BET_THREE_QUARTERS"
            else:
                return "CALL"
        elif winp > 0.6:
            if rnd < 0.3:
                return "BET_THREE_QUARTERS"
            elif rnd < 0.6:
                return "BET_HALF"
            else:
                return "CALL"
        elif winp > 0.45:
            if rnd < 0.3:
                return "BET_HALF"
            elif rnd < 0.5:
                return "BET_QUARTER"
            else:
                return "CALL"
        return "CALL"
    else:
        # No bet faced: check or value bet/bluff
        if winp > 0.8:
            if rnd < 0.5:
                return "BET_POT"
            else:
                return "BET_THREE_QUARTERS"
        elif winp > 0.65:
            if rnd < 0.4:
                return "BET_THREE_QUARTERS"
            elif rnd < 0.7:
                return "BET_HALF"
            else:
                return "CHECK"
        elif winp > 0.5:
            if rnd < 0.4:
                return "BET_HALF"
            elif rnd < 0.6:
                return "BET_QUARTER"
            else:
                return "CHECK"
        elif winp > 0.35:
            if rnd < 0.25:
                return "BET_MIN"
            else:
                return "CHECK"
                
        # Occasionally bluff semi-draws
        if features["flush_draw"] or features["straight_draw"]:
            if rnd < 0.2:
                return "BET_HALF"
            elif rnd < 0.35:
                return "BET_QUARTER" 
            elif rnd < 0.45:
                return "BET_MIN"
            else:
                return "CHECK"
        
        if rnd < 0.08:
            return "BET_MIN"
        
        return "CHECK"

def play_synthetic_hand(mc_samples=80):
    """Simulate a hand and collect (feature, action) pairs."""
    deck_cards = new_deck()
    random.shuffle(deck_cards)
    hero = deck_cards[:2]
    deck_cards = deck_cards[2:]
    opp = deck_cards[:2]
    deck_cards = deck_cards[2:]

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

    # choose a random event for this synthetic hand
    possible_events = ["Normal", "War", "Joker Wilds", "No face cards", "Only face cards",
                       "Cards are only suit", "Cards are only suit and suit", "Rankings are reversed", "Banned Rank"]
    hand_event = random.choice(possible_events)

    def one_round(round_idx, first_is_hero):
        nonlocal pot, hero_stack, opp_stack, to_call_hero, to_call_opp, board, hand_event

        # FIRST player decision
        if first_is_hero:
            faced_bet = to_call_hero > 0
            feats = features_for_state(round_idx, hero, board, pot, hero_stack, opp_stack,
                                       to_call_hero, in_position=0, opp_count=1, mc_samples=mc_samples, event=hand_event)
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
                                           to_call_opp, in_position=0, opp_count=1, mc_samples=mc_samples//2, event=hand_event)
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

        # SECOND player decision if needed
        if first_is_hero:
            faced_bet = to_call_opp > 0
            if faced_bet:
                feats_opp = features_for_state(round_idx, opp, board, pot, opp_stack, hero_stack,
                                            to_call_opp, in_position=1, opp_count=1, mc_samples=mc_samples//2)
                opp_label = sample_action_from_policy(feats_opp, faced_bet)
                pot, opp_stack, hero_stack, to_call_opp, to_call_hero, ended = apply_action(
                    opp_label, pot, opp_stack, hero_stack, to_call_opp, to_call_hero)
                if ended:
                    return True
        
        return False

    # Preflop (hero first)
    ended = one_round(0, first_is_hero=True)
    if ended:
        return seq_feats, seq_labels

    # Flop
    board.extend(deck_cards[:3])
    deck_cards = deck_cards[3:]
    to_call_hero = to_call_opp = 0
    ended = one_round(1, first_is_hero=False)
    if ended:
        return seq_feats, seq_labels

    # Turn
    board.extend(deck_cards[:1])
    deck_cards = deck_cards[1:]
    to_call_hero = to_call_opp = 0
    ended = one_round(2, first_is_hero=False)
    if ended:
        return seq_feats, seq_labels

    # River
    board.extend(deck_cards[:1])
    deck_cards = deck_cards[1:]
    to_call_hero = to_call_opp = 0
    ended = one_round(3, first_is_hero=False)
    return seq_feats, seq_labels


def apply_action(label, pot, actor_stack, other_stack, to_call_actor, to_call_other):
    """
    Resolves a single action label for the acting player.
    Returns updated (pot, actor_stack, other_stack, to_call_actor, to_call_other, ended)
    """
    call_amt, min_bet, quarter, half, three_quarters, pot_bet, allin = discretize_bet_sizes(pot, actor_stack, to_call_actor)

    if label == "FOLD":
        # Hand ends immediately
        return pot, actor_stack, other_stack, to_call_actor, to_call_other, True

    if label == "CHECK":
        # Can only check if no bet to call
        if to_call_actor > 0:
            # Treat as call if there's a bet to call (shouldn't happen with proper policy)
            label = "CALL"
        else:
            # No money changes hands
            return pot, actor_stack, other_stack, 0, to_call_other, False

    if label == "CALL":
        pay = min(call_amt, actor_stack)
        actor_stack -= pay
        pot += pay
        to_call_actor = 0
        to_call_other = 0
        return pot, actor_stack, other_stack, to_call_actor, to_call_other, False

    # Handle various bet sizes
    if label in ("BET_MIN", "BET_QUARTER", "BET_HALF", "BET_THREE_QUARTERS", "BET_POT", "ALLIN"):
        target = {
            "BET_MIN": min_bet,
            "BET_QUARTER": quarter, 
            "BET_HALF": half, 
            "BET_THREE_QUARTERS": three_quarters,
            "BET_POT": pot_bet, 
            "ALLIN": allin
        }[label]
        bet_amt = min(target, actor_stack)
        actor_stack -= bet_amt
        pot += bet_amt
        to_call_other = bet_amt
        to_call_actor = 0
        return pot, actor_stack, other_stack, to_call_actor, to_call_other, False

    # Fallback to check if action unrecognized
    return pot, actor_stack, other_stack, to_call_actor, to_call_other, False

def simulate_sequences(n_hands, mc_samples=60):
    """Generate training data from simulated hands."""
    X, y = [], []
    for i in range(n_hands):
        if i % 100 == 0 and i > 0:
            print(f"Generated {i} hands...")
        feats, labels = play_synthetic_hand(mc_samples=mc_samples)
        if feats and labels:
            X.append(feats)
            y.append(labels)
    return X, y

def train_crf(X, y):
    """Train a CRF model on the provided data."""
    labels = ["FOLD", "CHECK", "CALL", "BET_MIN", "BET_QUARTER", "BET_HALF", "BET_THREE_QUARTERS", "BET_POT", "ALLIN"]

    crf = LinearChainCRF()
    print("Training custom numpy CRF model...")
    crf.fit(X, y, labels=labels, epochs=30, lr=0.0001, l2=0.0001, verbose=1)
    return crf

def main():
    num_hands = 30000
    model = "crf_custom_v2.pkl"
    seq_file = "sequences.pkl"

    if os.path.exists(seq_file):
        with open(seq_file, "rb") as f:
            X, y = pickle.load(f)
        print(f"Loaded {len(X)} sequences from {seq_file}")
    else:
        X, y = simulate_sequences(num_hands, mc_samples=75)
        print(f"Generated {len(X)} sequences with average length ~{sum(len(s) for s in X)/max(1,len(X)):.2f}")
        with open(seq_file, "wb") as f:
            pickle.dump((X, y), f)
        print(f"Saved generated sequences to {seq_file}")

    crf = train_crf(X, y)
    crf.save(model)
    print(f"Saved trained model to {model}")


if __name__ == "__main__":
    main()
