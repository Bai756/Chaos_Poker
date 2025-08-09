from main import evaluate_hand, Card

def make_hand(card_strs):
    """Helper to create a list of Card objects from strings like 'AH', '7D', etc."""
    rank_map = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
                'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
    suit_map = {'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs', 'S': 'Spades'}
    cards = []
    for cs in card_strs:
        if len(cs) == 3:
            rank, suit = cs[:2], cs[2]
        else:
            rank, suit = cs[0], cs[1]
        cards.append(Card(rank_map[rank], suit_map[suit]))
    return cards

test_cases = [
    # (Hand Type, [Card strings])
    ("7-card straight-flush", ['7H','8H','9H','10H','JH','QH','KH']),
    ("Quad set (4 + 3)", ['7H','7D','7C','7S','8H','8D','8C']),
    ("6-card straight-flush", ['8H','9H','10H','JH','QH','KH','2D']),
    ("7-card flush", ['2H','4H','6H','8H','10H','QH','KH']),
    ("5-card straight-flush", ['9H','10H','JH','QH','KH','2D','3S']),
    ("Quad House (4 + 2 + 1)", ['7H','7D','7C','7S','8H','8D','9C']),
    ("Super set (3 + 3 + 1)", ['7H','7D','7C','8H','8D','8C','9S']),
    ("7-card straight", ['7H','8D','9C','10S','JH','QH','KH']),
    ("Mega full house (3 + 2 + 2)", ['7H','7D','7C','8H','8D','9C','9S']),
    ("Quads (4 + 1 + 1 + 1)", ['7H','7D','7C','7S','8H','9D','10C']),
    ("6-card flush", ['2H','4H','6H','8H','10H','QH','KD']),
    ("6-card straight", ['6H','7D','8C','9S','10H','JH','2H']),
    ("3 pair (2 + 2 + 2 + 1)", ['7H','7D','8C','8S','9H','9D','10C']),
    ("Full House (3 + 2 + 1 + 1)", ['7H','7D','7C','8H','8D','9C','10S']),
    ("5-card flush", ['2H','4H','6H','8H','10H','QD','KS']),
    ("Trips (3 + 1 + 1 + 1 + 1)", ['7H','7D','7C','8H','9D','10C','QS']),
    ("5-card straight", ['7H','8D','9C','10S','JH','2C','3D']),
    ("2 pair (2 + 2 + 1 + 1 + 1)", ['7H','7D','8C','8S','9H','10D','QC']),
    ("1 pair", ['7H','7D','8C','9S','2H','JD','QC']),
    ("Nothing", ['2H','4D','6C','8S','10H','QD','KS']),
]

for hand_type, card_strs in test_cases:
    cards = make_hand(card_strs)
    result = evaluate_hand(cards)
    print(f"Test: {hand_type:28} | Detected: {result[0]:28} | Details: {result[1:]}")

