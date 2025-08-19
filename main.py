import pygame
import sys
import pickle
from base import evaluate_hand, Deck, hand_name_from_rank, Player
from crf import mc_win_prob, new_deck as crf_new_deck, board_texture, draws_flags

BTN_X = 750
BTN_W = 130
BTN_H = 40
BTN_SPACING = 70
BTN_Y_START = 130

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

OK_BTN_RECT = pygame.Rect(700, 480, 70, 32)
CANCEL_BTN_RECT = pygame.Rect(190, 480, 100, 32)

pygame.init()
pygame.font.init()

FONT = pygame.font.SysFont("arial", 24)
BIG_FONT = pygame.font.SysFont("arial", 28, bold=True)
CARD_FONT = pygame.font.SysFont("arial", 32, bold=True)

GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
RED = (220, 0, 0)
GOLD = (255, 215, 0)

WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("7 Card Hold'em")

players = []
community_cards = []
deck = None
pot = 0
round_stage = "preflop"
message = "Welcome to Hold'em!"
winner = None
show_bet_slider = False
bet_slider_value = 0
bet_slider_min = 0
bet_slider_max = 0
bet_slider_preset = []
slider_dragging = False
dealer_position = 0
waiting_for_action = False

try:
    with open("crfv3.pkl", "rb") as f:
        crf_model = pickle.load(f)
    print("CRF AI model loaded successfully")
except Exception as e:
    print(f"Error loading CRF model: {e}")
    crf_model = None

RANK_TO_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                 '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

class Button:
    def __init__(self, text, x, y, w, h, color, action):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.action = action
        self.visible = True

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=6)
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
    pot_size = pot
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
    pygame.draw.rect(screen, (30, 30, 30), (x, y, INFOBOX_W, INFOBOX_H), border_radius=10)
    info = f"Chips: {player.chips}  Bet: {player.current_bet}"
    info_text = FONT.render(info, True, WHITE)
    screen.blit(info_text, (x + 10, y + 15))

def draw_hand(cards, x, y, highlight=False, hide=False):
    for i, card in enumerate(cards):
        display_card = "" if hide else card
        draw_card(screen, display_card, x + i * CARD_SPACING, y, highlight)

def draw_card(surface, card, x, y, highlight=False):
    pygame.draw.rect(surface, (60, 60, 60), (x+4, y+6, CARD_W, CARD_H), border_radius=8)
    border_color = GOLD if highlight else WHITE
    pygame.draw.rect(surface, border_color, (x, y, CARD_W, CARD_H), border_radius=8)
    pygame.draw.rect(surface, (240, 240, 240), (x+3, y+3, CARD_W-6, CARD_H-6), border_radius=6)
    if not card:
        card_str = ""
        color = GRAY
    else:
        card_str = f"{card.rank}{card.suit}"
        color = BLACK if card.suit in ['♠', '♣'] else RED
    text = CARD_FONT.render(card_str, True, color)
    surface.blit(text, (x + 10, y + 30))

def draw_bet_slider():
    pygame.draw.rect(screen, (40, 40, 40), (250, 350, 400, 140), border_radius=12)
    pygame.draw.rect(screen, GOLD, (250, 350, 400, 140), 3, border_radius=12)
    txt = BIG_FONT.render("Choose Bet Amount", True, WHITE)
    screen.blit(txt, (WIDTH//2 - txt.get_width()//2, 360))
    
    pygame.draw.rect(screen, WHITE, (SLIDER_X, SLIDER_Y, SLIDER_W, SLIDER_H), border_radius=4)
    
    # Slider handle
    pos = SLIDER_X if bet_slider_max == bet_slider_min else int(SLIDER_X + (bet_slider_value - bet_slider_min) / (bet_slider_max - bet_slider_min) * SLIDER_W)
    pygame.draw.circle(screen, GOLD, (pos, SLIDER_Y + SLIDER_H//2), 14)
    
    val_txt = FONT.render(f"{bet_slider_value} chips", True, GOLD)
    screen.blit(val_txt, (WIDTH//2 - val_txt.get_width()//2, 440))
    
    preset_labels = ["Min", "Pot", "All In"]
    for i, val in enumerate(bet_slider_preset):
        bx = 300 + i*130
        pygame.draw.rect(screen, GRAY, (bx, PRESET_BY, PRESET_BW, PRESET_BH), border_radius=8)
        label = FONT.render(f"{preset_labels[i]} ({val})", True, BLACK)
        screen.blit(label, (bx + PRESET_BW//2 - label.get_width()//2, PRESET_BY + PRESET_BH//2 - label.get_height()//2))
    
    pygame.draw.rect(screen, GOLD, OK_BTN_RECT, border_radius=8)
    ok_txt = FONT.render("OK", True, BLACK)
    screen.blit(ok_txt, (OK_BTN_RECT.centerx - ok_txt.get_width()//2, OK_BTN_RECT.centery - ok_txt.get_height()//2))
    pygame.draw.rect(screen, RED, CANCEL_BTN_RECT, border_radius=8)
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

def start_new_game():
    global players, community_cards, pot, round_stage, message, winner, dealer_position
    new_deck()
    players = [Player("Player"), Player("Opponent")]
    pot = 0
    round_stage = "preflop"
    winner = None
    message = "New hand!"
    
    dealer_position = (dealer_position + 1) % 2
    
    # Set blinds based on dealer position
    if dealer_position == 0:
        players[0].chips -= 5
        players[0].current_bet = 5
        players[1].chips -= 10
        players[1].current_bet = 10
        message = "You are dealer (small blind)"
    else:
        players[1].chips -= 5
        players[1].current_bet = 5
        players[0].chips -= 10
        players[0].current_bet = 10
        message = "Opponent is dealer (small blind)"
        
    players[0].acted = False
    players[1].acted = False
    
    pot += 15
    
    # Deal
    for p in players:
        p.hand = deck.deal(2)
        p.active = True
    community_cards.clear()
    update_buttons()
    draw()

def deal_community(n):
    global community_cards
    community_cards.extend(deck.deal(n))

def bet_action(amount):
    global pot, message, waiting_for_action
    player = players[0]
    if winner or player.chips < amount:
        return
    player.chips -= amount
    player.current_bet += amount
    pot += amount
    message = f"You bet {amount} chips."
    
    player.acted = True
    
    players[1].acted = False
    
    update_buttons()
    waiting_for_action = False

def check_action():
    global message, pot, waiting_for_action, round_stage
    opponent_bet = players[1].current_bet
    player_bet = players[0].current_bet
    
    if opponent_bet > player_bet:
        to_call = opponent_bet - player_bet
        call_amt = min(to_call, players[0].chips)
        players[0].chips -= call_amt
        players[0].current_bet += call_amt
        pot += call_amt
        
        if call_amt < to_call:
            message = f"You call all-in with {call_amt}"
        else:
            message = f"You called {call_amt}."
    else:
        message = "You checked."
    
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
        message = "Flop dealt."
    elif round_stage == "flop":
        deal_community(1)
        round_stage = "turn"
        message = "Turn dealt."
    elif round_stage == "turn":
        deal_community(1)
        round_stage = "river"
        message = "River dealt."
    elif round_stage == "river":
        round_stage = "showdown"
        determine_winner()
    else:
        start_new_game()
    update_buttons()

def make_buttons():
    return [
        Button("Check", BTN_X, BTN_Y_START, BTN_W, BTN_H, GRAY, check_action),  # Check/Call button now first
        Button("Bet", BTN_X, BTN_Y_START + BTN_SPACING, BTN_W, BTN_H, GRAY, open_bet_slider),  # Bet/Raise button second
        Button("Fold", BTN_X, BTN_Y_START + 2*BTN_SPACING, BTN_W, BTN_H, RED, fold_action),
        Button("New Game", BTN_X, BTN_Y_START + 3*BTN_SPACING, BTN_W, BTN_H, GRAY, start_new_game)
    ]

buttons = make_buttons()

def determine_winner():
    global winner, message
    player_best = evaluate_hand(players[0].hand + community_cards)
    opponent_best = evaluate_hand(players[1].hand + community_cards)
    player_name = hand_name_from_rank(player_best[0])
    opponent_name = hand_name_from_rank(opponent_best[0])
    
    if not players[0].active:
        winner = "Opponent"
        players[1].chips += pot
    elif not players[1].active:
        winner = "Player"
        players[0].chips += pot
    elif player_best > opponent_best:
        winner = "Player"
        players[0].chips += pot
    elif player_best < opponent_best:
        winner = "Opponent"
        players[1].chips += pot
    else:
        winner = "Tie"
        # Split pot
        split = pot // 2
        players[0].chips += split
        players[1].chips += pot - split
        
    message = f"Winner: {winner} ({player_name} vs {opponent_name})"

def close_bet_slider():
    global show_bet_slider
    show_bet_slider = False

def draw():
    screen.fill(GREEN)
    pygame.draw.ellipse(screen, (0, 100, 0), (80, 120, WIDTH-160, HEIGHT-240))

    pot_text = BIG_FONT.render(f"Pot: {pot}", True, GOLD)
    screen.blit(pot_text, (50, 110))

    draw_hand(community_cards, 260, 180)
    draw_info_box(INFOBOX_PLAYER_X, INFOBOX_PLAYER_Y, players[0])
    draw_info_box(INFOBOX_OPP_X, INFOBOX_OPP_Y, players[1])

    highlight = (winner == "Player" and round_stage == "showdown")
    draw_hand(players[0].hand, 400, 500, highlight)
    highlight = (winner == "Opponent" and round_stage == "showdown")
    hide = round_stage != "showdown"
    draw_hand(players[1].hand, 400, 30, highlight, hide)

    if round_stage == "showdown":
        msg_bg = pygame.Surface((WIDTH, 80), pygame.SRCALPHA)
        msg_bg.fill((0, 0, 0, 180))
        screen.blit(msg_bg, (0, HEIGHT // 2 - 40))
        msg_text = BIG_FONT.render(message, True, GOLD)
        screen.blit(msg_text, (WIDTH // 2 - msg_text.get_width() // 2, HEIGHT // 2 - msg_text.get_height() // 2))
    else:
        msg_text = FONT.render(message, True, WHITE)
        screen.blit(msg_text, (50, 140))

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
    
    buttons[3].visible = True
    
    for b in buttons:
        if getattr(b, "visible", True):
            b.draw(screen)
            
    if show_bet_slider:
        draw_bet_slider()
        
    pygame.display.flip()

def opponent_action():
    global waiting_for_action, pot, message, winner, round_stage
    opp = players[1]
    player = players[0]
    to_call = player.current_bet - opp.current_bet
    bet_amount = 0
    
    print(f"\n--- AI Opponent's turn ---")
    print(f"Your bet: {player.current_bet}, AI bet: {opp.current_bet}, To call: {to_call}")
    
    if not crf_model:
        raise Exception("CRF model not loaded. Cannot make AI decision.")
    if opp.chips <= 0:
        opp.acted = True
        waiting_for_action = True
        return
    
    # Extract features for the AI decision
    round_idx = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}[round_stage]
    
    deck_cards = crf_new_deck()
    
    # Calculate win probability using Monte Carlo simulation
    mc_trials = 100
    win_prob = mc_win_prob(opp.hand, community_cards, deck_cards, opp_count=1, trials=mc_trials)
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
        players[0].chips += pot
        round_stage = "showdown"
    
    elif action == "CHECK":
        if to_call > 0:
            action = "CALL"
        else:
            message = "AI opponent checks"
            opp.acted = True
    
    elif action == "CALL":
        call_amt = min(to_call, opp.chips)
        opp.chips -= call_amt
        opp.current_bet += call_amt
        pot += call_amt
        opp.acted = True
        
        if call_amt < to_call:
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
        allin = opp.chips
        
        # Check if AI can't afford the minimum raise/bet
        if to_call > 0 and opp.chips <= to_call:
            # AI can't even call, so go all-in
            action = "CALL"
            call_amt = opp.chips
            opp.chips = 0
            pot += call_amt
            opp.current_bet += call_amt
            opp.acted = True
            message = f"AI opponent calls all-in with {call_amt}"
        elif to_call == 0 and opp.chips < 10:
            # AI can't afford minimum bet, so check
            action = "CHECK"
            message = "AI opponent checks (not enough chips to bet)"
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
            else:  # ALLIN
                bet_amount = opp.chips
                
            if to_call > 0:  # Raising
                min_raise = to_call + 10
                
                if opp.chips < to_call:
                    # Not enough to call, go all-in
                    call_amt = opp.chips
                    opp.chips = 0
                    pot += call_amt
                    opp.current_bet += call_amt
                    message = f"AI opponent calls all-in with {call_amt}"
                elif opp.chips < min_raise:
                    # Enough to call but not raise, just call
                    call_amt = to_call
                    opp.chips -= call_amt
                    pot += call_amt
                    opp.current_bet += call_amt
                    message = f"AI opponent calls {call_amt} (not enough to raise)"
                else:
                    # Can raise
                    actual_raise = min(bet_amount, opp.current_bet + opp.chips)
                    
                    additional_chips = actual_raise - opp.current_bet
                    
                    opp.chips -= additional_chips
                    pot += additional_chips
                    opp.current_bet = actual_raise
                    
                    if additional_chips == opp.chips:
                        message = f"AI opponent raises all-in to {actual_raise}"
                    else:
                        message = f"AI opponent raises to {actual_raise}"
            else: # Betting
                bet_amount = min(bet_amount, opp.chips)
                
                opp.chips -= bet_amount
                pot += bet_amount
                opp.current_bet = bet_amount
                
                if bet_amount == opp.chips:
                    message = f"AI opponent bets all-in ({bet_amount})"
                else:
                    message = f"AI opponent bets {bet_amount}"
                
        opp.acted = True
        
        # Mark player as not acted since they need to respond to a raise/bet
        if bet_amount > player.current_bet and player.chips > 0:
            players[0].acted = False
    
    update_buttons()
    waiting_for_action = True

def get_action_order():
    if round_stage == "preflop":
        return [players[0], players[1]] if dealer_position == 0 else [players[1], players[0]]
    else:
        return [players[1], players[0]] if dealer_position == 0 else [players[0], players[1]]

def game_loop():
    global slider_dragging, bet_slider_value, winner, round_stage
    
    if all(p.acted for p in players if p.active) and any(p.chips == 0 for p in players if p.active):
        # Deal remaining cards
        while round_stage not in ["showdown"]:
            advance_round()
            draw()
    
    while not winner and round_stage != "showdown":
        action_order = get_action_order()
        
        # First player to act
        first_player = action_order[0]
        if not first_player.acted:
            if first_player == players[0]:
                handle_player_action()
            else:
                opponent_action()
            update_buttons()
            draw()
        
        # Check for all-in situation
        if any(p.chips == 0 and p.active for p in players):
            # If someone is all-in and everyone has acted, run out the remaining cards
            if all(p.acted for p in players if p.active):
                while round_stage not in ["showdown"]:
                    advance_round()
                    draw()
                break
        
        if check_round_over():
            advance_round()
            draw()
            continue
            
        if winner or round_stage == "showdown" or not (players[0].active and players[1].active):
            continue
        
        # Second player to act
        second_player = action_order[1]
        if not second_player.acted:
            if second_player == players[0]:
                handle_player_action()
            else:
                opponent_action()
            update_buttons()
            draw()
        
        if check_round_over():
            advance_round()
            draw()
            
        if not players[0].active or not players[1].active:
            determine_winner()
            draw()

def check_round_over():
    active_players = [p for p in players if p.active]
    
    # If there's only one active player, the round is over
    if len(active_players) <= 1:
        return True
    
    all_acted = all(p.acted for p in active_players)
    
    if any(p.chips <= 0 for p in active_players):
        return all_acted
    
    bets_match = len(set(p.current_bet for p in active_players)) <= 1
    
    return all_acted and bets_match

def handle_player_action():
    global slider_dragging, bet_slider_value, waiting_for_action
    waiting_for_action = True
    
    if players[0].acted:
        waiting_for_action = False
        return
        
    while waiting_for_action:
        draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if show_bet_slider:
                    mx, my = event.pos
                    handle_bet_slider_mouse_down(mx, my)
                else:
                    for b in buttons:
                        if getattr(b, "visible", True) and b.rect.collidepoint(event.pos):
                            b.click()
                            if b.text in ["Check", "Fold"] or b.text.startswith("Call ("):
                                waiting_for_action = False
            elif event.type == pygame.MOUSEBUTTONUP:
                slider_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if show_bet_slider and slider_dragging:
                    mx, my = event.pos
                    mx = max(SLIDER_X, min(mx, SLIDER_X + SLIDER_W))
                    rel = (mx - SLIDER_X) / SLIDER_W
                    bet_slider_value = int(bet_slider_min + rel * (bet_slider_max - bet_slider_min))
        pygame.time.wait(10)

def handle_bet_slider_mouse_down(mx, my):
    global slider_dragging, bet_slider_value, waiting_for_action
    # Check if clicking the slider handle
    pos = SLIDER_X if bet_slider_max == bet_slider_min else int(SLIDER_X + (bet_slider_value - bet_slider_min) / (bet_slider_max - bet_slider_min) * SLIDER_W)
    if abs(mx - pos) <= 18 and abs(my - (SLIDER_Y + SLIDER_H//2)) <= 18:
        slider_dragging = True
    else:
        if handle_bet_slider_event(mx, my):
            waiting_for_action = False


if __name__ == "__main__":
    start_new_game()
    while True:
        game_loop()
    
        while round_stage == "showdown":
            draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for b in buttons:
                        if getattr(b, "visible", True) and b.rect.collidepoint(event.pos):
                            b.click()
            pygame.time.wait(10)
