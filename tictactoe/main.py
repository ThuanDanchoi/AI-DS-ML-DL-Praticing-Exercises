import pygame
import sys
import os
import time

import tictactoe as ttt

# Initialize Pygame
pygame.init()

# Define the window size
size = width, height = 800, 800

# Define colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
yellow = (255, 255, 0)

# Display window
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Tic-Tac-Toe")

# Load fonts
mediumFont = pygame.font.Font("Kanit-LightItalic.ttf", 30)
largeFont = pygame.font.Font("Kanit-LightItalic.ttf", 50)
moveFont = pygame.font.Font("Kanit-LightItalic.ttf", 80)

# Initialize game state
user = None
board = ttt.initial_state()
ai_turn = False


def draw_winning_line(winning_positions, color):
    """
    Draw the winning line with the specified color.
    """
    for pos in winning_positions:
        i, j = pos
        rect = pygame.Rect(
            tile_origin[0] + j * tile_size,
            tile_origin[1] + i * tile_size,
            tile_size, tile_size
        )
        pygame.draw.rect(screen, color, rect, 5)
        move = moveFont.render(board[i][j], True, color)
        moveRect = move.get_rect(center=rect.center)
        screen.blit(move, moveRect)


while True:

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    # Fill the background
    screen.fill(black)

    # Player selection screen
    if user is None:
        # Display the title
        title = largeFont.render("Play Tic-Tac-Toe", True, white)
        titleRect = title.get_rect(center=(width / 2, 50))
        screen.blit(title, titleRect)

        # Create buttons to select X or O
        playXButton = pygame.Rect(width / 8, height / 2, width / 4, 50)
        playX = mediumFont.render("Play as X", True, black)
        playXRect = playX.get_rect(center=playXButton.center)
        pygame.draw.rect(screen, white, playXButton)
        screen.blit(playX, playXRect)

        playOButton = pygame.Rect(5 * (width / 8), height / 2, width / 4, 50)
        playO = mediumFont.render("Play as O", True, black)
        playORect = playO.get_rect(center=playOButton.center)
        pygame.draw.rect(screen, white, playOButton)
        screen.blit(playO, playORect)

        # Handle button clicks
        click, _, _ = pygame.mouse.get_pressed()
        if click == 1:
            mouse = pygame.mouse.get_pos()
            if playXButton.collidepoint(mouse):
                time.sleep(0.2)
                user = ttt.X
            elif playOButton.collidepoint(mouse):
                time.sleep(0.2)
                user = ttt.O

    else:
        # Game board display
        tile_size = 80
        tile_origin = (width / 2 - 1.5 * tile_size, height / 2 - 1.5 * tile_size)
        tiles = []
        for i in range(3):
            row = []
            for j in range(3):
                rect = pygame.Rect(
                    tile_origin[0] + j * tile_size,
                    tile_origin[1] + i * tile_size,
                    tile_size, tile_size
                )
                # Draw the background of each tile based on the game state
                if ttt.terminal(board) and ttt.winner(board) is None:
                    pygame.draw.rect(screen, yellow, rect)
                else:
                    pygame.draw.rect(screen, white, rect, 3)

                if board[i][j] != ttt.EMPTY:
                    move = moveFont.render(board[i][j], True,
                                           black if ttt.terminal(board) and ttt.winner(board) is None else white)
                    moveRect = move.get_rect(center=rect.center)
                    screen.blit(move, moveRect)
                row.append(rect)
            tiles.append(row)

        game_over = ttt.terminal(board)
        player = ttt.player(board)

        # Check if there's a winner
        winning_positions = None
        if game_over:
            winner = ttt.winner(board)
            if winner is not None:
                if winner == user:
                    winning_positions = [(i, j) for i in range(3) for j in range(3) if board[i][j] == winner]
                    draw_winning_line(winning_positions, green)
                else:
                    winning_positions = [(i, j) for i in range(3) for j in range(3) if board[i][j] == winner]
                    draw_winning_line(winning_positions, red)

        # Show title
        if game_over:
            if winner is None:
                title = f"Game Over: Tie."
            else:
                title = f"Game Over: {winner} wins."
        elif user == player:
            title = f"Play as {user}"
        else:
            title = "Computer thinking..."

        title = largeFont.render(title, True, white)
        titleRect = title.get_rect(center=(width / 2, 30))
        screen.blit(title, titleRect)

        # AI move
        if user != player and not game_over:
            if ai_turn:
                time.sleep(0.5)
                move = ttt.minimax(board)
                board = ttt.result(board, move)
                ai_turn = False
            else:
                ai_turn = True

        # User move
        click, _, _ = pygame.mouse.get_pressed()
        if click == 1 and user == player and not game_over:
            mouse = pygame.mouse.get_pos()
            for i in range(3):
                for j in range(3):
                    if board[i][j] == ttt.EMPTY and tiles[i][j].collidepoint(mouse):
                        board = ttt.result(board, (i, j))

        # Restart the game if it's over
        if game_over:
            againButton = pygame.Rect(width / 3, height - 65, width / 3, 50)
            again = mediumFont.render("Play Again", True, black)
            againRect = again.get_rect(center=againButton.center)
            pygame.draw.rect(screen, white, againButton)
            screen.blit(again, againRect)
            click, _, _ = pygame.mouse.get_pressed()
            if click == 1:
                mouse = pygame.mouse.get_pos()
                if againButton.collidepoint(mouse):
                    time.sleep(0.2)
                    user = None
                    board = ttt.initial_state()
                    ai_turn = False

    # Update the display
    pygame.display.flip()