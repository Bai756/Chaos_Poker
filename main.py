import pygame
import sys
from base import evaluate_hand, Deck, hand_name_from_rank, Player

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


class Button:
    def __init__(self, text, x, y, w, h, color, action):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.action = action

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
    update_bet_button()

def update_bet_button():
    opponent_bet = players[1].current_bet
    player_bet = players[0].current_bet
    buttons[0].text = "Raise" if opponent_bet > player_bet else "Bet"

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
    
    # Value
    val_txt = FONT.render(f"{bet_slider_value} chips", True, GOLD)
    screen.blit(val_txt, (WIDTH//2 - val_txt.get_width()//2, 440))
    
    # Preset buttons
    preset_labels = ["Min", "Pot", "All In"]
    for i, val in enumerate(bet_slider_preset):
        bx = 300 + i*130
        pygame.draw.rect(screen, GRAY, (bx, PRESET_BY, PRESET_BW, PRESET_BH), border_radius=8)
        label = FONT.render(f"{preset_labels[i]} ({val})", True, BLACK)
        screen.blit(label, (bx + PRESET_BW//2 - label.get_width()//2, PRESET_BY + PRESET_BH//2 - label.get_height()//2))
    
    # Confirm/cancel
    pygame.draw.rect(screen, GOLD, OK_BTN_RECT, border_radius=8)
    ok_txt = FONT.render("OK", True, BLACK)
    screen.blit(ok_txt, (OK_BTN_RECT.centerx - ok_txt.get_width()//2, OK_BTN_RECT.centery - ok_txt.get_height()//2))
    pygame.draw.rect(screen, RED, CANCEL_BTN_RECT, border_radius=8)
    cancel_txt = FONT.render("Cancel", True, WHITE)
    screen.blit(cancel_txt, (CANCEL_BTN_RECT.centerx - cancel_txt.get_width()//2, CANCEL_BTN_RECT.centery - cancel_txt.get_height()//2))

def handle_bet_slider_event(mx, my):
    global bet_slider_value
    # Slider bar
    if SLIDER_Y <= my <= SLIDER_Y + SLIDER_H and SLIDER_X <= mx <= SLIDER_X + SLIDER_W:
        rel = (mx - SLIDER_X) / SLIDER_W
        bet_slider_value = int(bet_slider_min + rel * (bet_slider_max - bet_slider_min))
    
    # Preset buttons
    for i, val in enumerate(bet_slider_preset):
        bx = 320 + i*110
        if bx <= mx <= bx+PRESET_BW and PRESET_BY <= my <= PRESET_BY+PRESET_BH:
            bet_slider_value = val
    
    # OK button
    if OK_BTN_RECT.collidepoint(mx, my):
        bet_action(bet_slider_value)
        close_bet_slider()
    
    # Cancel button
    if CANCEL_BTN_RECT.collidepoint(mx, my):
        close_bet_slider()

def new_deck():
    global deck
    deck = Deck()

def start_new_game():
    global players, community_cards, pot, round_stage, message, winner
    new_deck()
    players = [Player("Player"), Player("Opponent")]
    pot = 0
    round_stage = "preflop"
    winner = None
    message = "New hand!"
    
    # Blinds
    players[0].chips -= 5
    players[0].current_bet = 5
    players[1].chips -= 10
    players[1].current_bet = 10
    pot += 15
    
    # Deal
    for p in players:
        p.hand = deck.deal(2)
        p.active = True
    community_cards.clear()
    update_bet_button()

def deal_community(n):
    global community_cards
    community_cards.extend(deck.deal(n))

def bet_action(amount):
    global pot, message
    player = players[0]
    if winner or player.chips < amount:
        return
    player.chips -= amount
    player.current_bet += amount
    pot += amount
    message = f"You bet {amount} chips."
    update_bet_button()

def fold_action():
    global message, winner
    players[0].active = False
    winner = "Opponent"
    message = "You folded."

def advance_round():
    global round_stage, message, winner
    if winner:
        return
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
    update_bet_button()

def make_buttons():
    return [
        Button("Bet", BTN_X, BTN_Y_START, BTN_W, BTN_H, GRAY, open_bet_slider),
        Button("Next Round", BTN_X, BTN_Y_START + BTN_SPACING, BTN_W, BTN_H, GRAY, advance_round),
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
    elif not players[1].active:
        winner = "Player"
    elif player_best > opponent_best:
        winner = "Player"
    elif player_best < opponent_best:
        winner = "Opponent"
    else:
        winner = "Tie"
    message = f"Winner: {winner} ({player_name} vs {opponent_name})"

def close_bet_slider():
    global show_bet_slider
    show_bet_slider = False

def draw():
    screen.fill(GREEN)
    pygame.draw.ellipse(screen, (0, 100, 0), (80, 120, WIDTH-160, HEIGHT-240))
    # Pot
    pot_text = BIG_FONT.render(f"Pot: {pot}", True, GOLD)
    screen.blit(pot_text, (50, 110))
    # Community cards
    draw_hand(community_cards, 260, 180)
    
    # Player info boxes
    draw_info_box(INFOBOX_PLAYER_X, INFOBOX_PLAYER_Y, players[0])
    draw_info_box(INFOBOX_OPP_X, INFOBOX_OPP_Y, players[1])
    # Player cards
    highlight = (winner == "Player" and round_stage == "showdown")
    draw_hand(players[0].hand, 400, 500, highlight)
    # Opponent cards
    highlight = (winner == "Opponent" and round_stage == "showdown")
    hide = round_stage != "showdown"
    draw_hand(players[1].hand, 400, 30, highlight, hide)
    
    # Winner message
    if round_stage == "showdown":
        msg_bg = pygame.Surface((WIDTH, 80), pygame.SRCALPHA)
        msg_bg.fill((0, 0, 0, 180))
        screen.blit(msg_bg, (0, HEIGHT // 2 - 40))
        msg_text = BIG_FONT.render(message, True, GOLD)
        screen.blit(msg_text, (WIDTH // 2 - msg_text.get_width() // 2, HEIGHT // 2 - msg_text.get_height() // 2))
    else:
        msg_text = FONT.render(message, True, WHITE)
        screen.blit(msg_text, (50, 140))
    
    # Buttons
    for b in buttons:
        b.draw(screen)
    # Bet slider
    if show_bet_slider:
        draw_bet_slider()
    pygame.display.flip()

start_new_game()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if show_bet_slider:
                mx, my = event.pos
                # Check if clicking the slider handle
                if (300 <= mx <= 600) and (SLIDER_Y-10 <= my <= SLIDER_Y+SLIDER_H+10):
                    slider_dragging = True
                else:
                    handle_bet_slider_event(mx, my)
            else:
                for b in buttons:
                    if b.rect.collidepoint(event.pos):
                        b.click()
        elif event.type == pygame.MOUSEBUTTONUP:
            slider_dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if show_bet_slider and slider_dragging:
                mx, my = event.pos
                # Clamp mouse x to slider bar
                mx = max(SLIDER_X, min(mx, SLIDER_X + SLIDER_W))
                rel = (mx - SLIDER_X) / SLIDER_W
                bet_slider_value = int(bet_slider_min + rel * (bet_slider_max - bet_slider_min))
    draw()
