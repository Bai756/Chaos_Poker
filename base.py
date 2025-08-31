import random
from collections import Counter
from itertools import combinations

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

def betting_round(players, min_bet, starting_bet=0, start_idx=0):
    pot = 0
    current_bet = starting_bet

    # Only reset bets if not pre-flop (so blinds are preserved)
    for i, p in enumerate(players):
        if p.chips > 0:
            if not starting_bet:
                p.current_bet = 0
        else:
            p.active = False

    active = [p for p in players if p.active]
    idx = start_idx
    last_raiser = None

    # Special handling for pre-flop: last_raiser is big blind
    preflop = (starting_bet == min_bet and start_idx == 2)
    if preflop:
        last_raiser = players[1]  # big blind

    # Keep track of who has acted this round
    acted = [False] * len(active)
    
    while True:
        if len(active) == 1:
            break

        num_players = len(active)
        
        for offset in range(num_players):
            i = (idx + offset) % num_players
            p = active[i]
            to_call = current_bet - p.current_bet

            if to_call > 0:
                while True:
                    prompt = f"{p.name}: (c)all {to_call}, (r)aise, (f)old? "
                    choice = input(prompt).strip().lower()
                    if choice == 'f':
                        p.active = False
                        active = [x for x in active if x.active]
                        if len(active) == 1:
                            break
                        acted[i] = True
                        break

                    elif choice == 'r':
                        while True:
                            try:
                                amt = int(input(f"Enter raise amount (min {current_bet + min_bet}, you have {p.chips + p.current_bet}): "))
                            except ValueError:
                                continue
                            if amt >= current_bet + min_bet and amt <= p.chips + p.current_bet:
                                break
                        raise_amt = amt - p.current_bet
                        p.chips -= raise_amt
                        pot += raise_amt
                        p.current_bet = amt
                        current_bet = amt
                        last_raiser = p
                        acted = [False] * len(active)
                        acted[i] = True
                        break

                    elif choice == 'c':
                        amt = min(to_call, p.chips)
                        p.chips -= amt
                        p.current_bet += amt
                        pot += amt
                        acted[i] = True
                        break

                    else:
                        print("Invalid input. Try again.")

            else:
                while True:
                    prompt = f"{p.name}: (k)heck, (b)et, (f)old? "
                    choice = input(prompt).strip().lower()
                    if choice == 'f':
                        p.active = False
                        active = [x for x in active if x.active]
                        if len(active) == 1:
                            break
                        acted[i] = True
                        break

                    elif choice == 'b':
                        while True:
                            try:
                                amt = int(input(f"Enter bet amount (min {current_bet + min_bet}, you have {p.chips}): "))
                            except ValueError:
                                continue
                            if current_bet + min_bet <= amt <= p.chips:
                                break
                        p.chips -= amt
                        p.current_bet = amt
                        pot += amt
                        current_bet = amt
                        last_raiser = p
                        acted = [False] * len(active)
                        acted[i] = True
                        break

                    elif choice == 'k':
                        acted[i] = True
                        break
                    
                    else:
                        print("Invalid input. Try again.")

            if all(acted):
                break

        # If everyone has acted and all bets are matched, end the round
        if all(acted) and all(p.current_bet == current_bet for p in active):
            break
            
        # If everyone checked in a round with no bets, end the round
        if current_bet == 0 and all(acted):
            break
            
        # Special case for preflop - big blind gets option if everyone called
        if preflop and all(p.current_bet == current_bet for p in active) and last_raiser == players[1]:
            if acted[active.index(players[1])]:
                break

    # Clean up bets at end of round
    for p in players:
        p.current_bet = 0

    return pot

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

def main():
    n = 0
    while n < 2 or n > 9:
        try:
            n = int(input("Please enter a number of players (2-9): "))
        except ValueError:
            continue
    players = [Player(f"Player {i+1}") for i in range(n)]
    deck = Deck()
    pot = 0

    # post blinds
    small, big = 5, 10
    players[0].chips -= small
    players[0].current_bet = small
    players[1].chips -= big
    players[1].current_bet = big
    pot += small + big

    # deal hole cards
    for p in players:
        p.hand = deck.deal(2)

    community = []
    stages = ['Pre-Flop', 'Flop', 'Turn', 'River']
    for stage in stages:
        print(f"\n-- {stage} --")
        if stage == 'Flop':
            community += deck.deal(3)
        elif stage in ('Turn', 'River'):
            community += deck.deal(1)

        # display table
        print("Community:", ', '.join(str(c) for c in community))
        print("Pot:", pot)
        for p in players:
            status = 'IN' if p.active else 'OUT'
            hand_str = ', '.join(str(c) for c in p.hand)
            print(f"{p.name} [{status}] Chips:{p.chips} Bet:{p.current_bet} Hand:{hand_str}")

        start = 2
        start_bet = big if stage == 'Pre-Flop' else 0
        pot += betting_round(players, big, start_bet, start)

        if sum(1 for p in players if p.active) == 1:
            break

    # showdown
    print("\n-- Showdown --")
    print("Community:", ', '.join(str(c) for c in community))
    best, winner = None, None
    for p in players:
        if p.active:
            score = evaluate_hand(p.hand + community)
            hand_name = hand_name_from_rank(score[0])
            print(f"{p.name}: {hand_name} {score[1:]}")
            if best is None or score > best:
                best, winner = score, p

    if winner:
        winner.chips += pot
        print(f"{winner.name} wins {pot} chips!")


if __name__ == "__main__":
    main()
