import random
import sys
from copy import deepcopy

# Board size
BOARD_SIZE = 15

# Define the board
class Board:
    def __init__(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    def display(self):
        for row in self.board:
            print(' '.join(row))

    def place_piece(self, x, y, player):
        if self.board[x][y] == ' ':
            self.board[x][y] = player
            return True
        return False

    def check_win(self, player):
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.board[x][y] == player:
                    if self.check_direction(x, y, 1, 0, player) or \
                       self.check_direction(x, y, 0, 1, player) or \
                       self.check_direction(x, y, 1, 1, player) or \
                       self.check_direction(x, y, 1, -1, player):
                        return True
        return False

    def check_direction(self, x, y, dx, dy, player):
        count = 0
        for _ in range(5):
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == player:
                count += 1
            else:
                break
            x += dx
            y += dy
        return count == 5

# Monte Carlo Tree Search
class MCTS:
    def __init__(self, board, player):
        self.board = board
        self.player = player

    def search(self, iterations):
        for _ in range(iterations):
            board_copy = deepcopy(self.board)
            self.simulate(board_copy)
        return self.best_move()

    def simulate(self, board):
        while not board.check_win('X') and not board.check_win('O'):
            x, y = random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1)
            if board.place_piece(x, y, self.player):
                self.player = 'X' if self.player == 'O' else 'O'

    def best_move(self):
        best_score = -sys.maxsize
        best_move = None
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.board.board[x][y] == ' ':
                    score = self.evaluate_move(x, y)
                    if score > best_score:
                        best_score = score
                        best_move = (x, y)
        return best_move

    def evaluate_move(self, x, y):
        score = 0
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 0
            for i in range(-4, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board.board[nx][ny] == self.player:
                    count += 1
            score += count
        return score

# Main game loop
def main():
    board = Board()
    mcts = MCTS(board, 'O')
    current_player = 'X'
    
    print("start")
    while not board.check_win('X') and not board.check_win('O'):
        board.display()
        if current_player == 'X':
            x, y = map(int, input("Enter your move (x, y): ").split())
            if board.place_piece(x, y, current_player):
                current_player = 'O'
        else:
            x, y = mcts.search(iterations=1000)
            if board.place_piece(x, y, current_player):
                current_player = 'X'

    print("end")
    board.display()
    if board.check_win('X'):
        print("Player X wins!")
    elif board.check_win('O'):
        print("Player O wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()
