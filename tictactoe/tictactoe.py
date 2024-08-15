import random
import math

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns the starting state of the board (a 3x3 list of lists with None values).
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    """
    Returns the player who has the next turn on a board.
    """
    # Count the number of X's and O's to determine who moves next
    X_count = sum(row.count(X) for row in board)
    O_count = sum(row.count(O) for row in board)
    return X if X_count <= O_count else O

def actions(board):
    """
    Returns the set of all possible actions (i, j) available on the board.
    """
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY}

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] is not EMPTY:
        raise ValueError("Invalid action: cell is not empty!")

    new_board = [row.copy() for row in board]
    new_board[action[0]][action[1]] = player(board)
    return new_board

def winner(board):
    """
    Returns the winner of the game if there is one, otherwise None.
    """
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != EMPTY:
            return row[0]
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != EMPTY:
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    return None

def terminal(board):
    """
    Returns True if the game is over (either someone has won or the board is full).
    """
    return winner(board) is not None or all(cell != EMPTY for row in board for cell in row)

def utility(board):
    """
    Returns 1 if X has won, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    current_player = player(board)

    if current_player == X:
        best_value = -float('inf')
        best_action = None
        for action in actions(board):
            value = minimax_value(result(board, action), False)
            if value > best_value:
                best_value = value
                best_action = action
    else:
        best_value = float('inf')
        best_action = None
        for action in actions(board):
            value = minimax_value(result(board, action), True)
            if value < best_value:
                best_value = value
                best_action = action

    return best_action

def minimax_value(board, is_maximizing):
    """
    Helper function for minimax to return the value.
    """
    if terminal(board):
        return utility(board)

    if is_maximizing:
        value = -float('inf')
        for action in actions(board):
            value = max(value, minimax_value(result(board, action), False))
    else:
        value = float('inf')
        for action in actions(board):
            value = min(value, minimax_value(result(board, action), True))

    return value

