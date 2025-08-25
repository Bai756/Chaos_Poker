# Chaos Poker

Heads up Texas Hold'em poker, but using all ***7*** cards instead of 5.
Each round there is a ***game event***.

## Game Events

Chaos Poker includes randomized "events" that change the game.
Possible events:

- Joker Wilds — adds multiple Joker cards that act as wilds.
- Cards are only 1 suit — deck restricted to a single suit.
- Cards are only 2 suits — deck restricted to two suits.
- No face cards — removes J, Q, K, A from the deck.
- Only face cards — keeps only J, Q, K, A in the deck.
- War — compares high-card values differently (high-card showdown).
- Normal — no special event; standard rules.
- Rankings are reversed — hand rankings are inverted.
- Banned Rank — a random rank (e.g., 2..A) is removed from the deck.

## AI

- Opponent driven by a CRF model
- AI action set: FOLD, CHECK, CALL, BET_MIN, BET_QUARTER, BET_HALF, BET_THREE_QUARTERS, BET_POT, ALLIN

## Install

Install dependencies:

```bash
pip install pygame numpy scikit-learn sklearn-crfsuite
```

## Run

Start the game:

```bash
python main.py
```

## Hand probabilities

- **7-card straight-flush**: 0.000021%
- **Quad set (4 + 3)**: 0.000466%
- **6-card straight-flush**: 0.001058%
- **7-card flush**: 0.005131%
- **5-card straight-flush**: 0.026909%
- **Quad House (4 + 2 + 1)**: 0.030784%
- **Super set (3 + 3 + 1)**: 0.041045%
- **7-card straight**: 0.085726%
- **Mega full house (3 + 2 + 2)**: 0.092351%
- **Quads**: 0.136817%
- **6-card flush**: 0.200095%
- **6-card straight**: 0.955231%
- **3 pair (2 + 2 + 2 + 1)**: 1.847029%
- **Full House**: 2.462706%
- **5-card flush**: 2.851351%
- **Trips**: 4.925411%
- **5-card straight**: 5.279031%
- **2 pair**: 22.164351%
- **1 pair**: 47.283950%
- **High card**: 17.655849%
