import tkinter as tk
from tkinter import messagebox
from copy import deepcopy
from random import choice
from math import sqrt, log

# Constants
BOARD_SIZE = 15
AI_PLAYER = 1
HUMAN_PLAYER = 2

# Monte Carlo Tree Search
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def select(self):
        return max(self.children, key=lambda child: child.wins / child.visits + sqrt(2 * log(self.visits) / child.visits))

    def expand(self):
        for move in self.state.get_legal_moves():
            child_state = deepcopy(self.state)
            child_state.make_move(move)
            child = MCTSNode(child_state, self)
            self.children.append(child)

    def rollout(self):
        current_state = deepcopy(self.state)
        while not current_state.is_terminal():
            move = choice(current_state.get_legal_moves())
            current_state.make_move(move)
        return current_state.get_winner()

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result == self.state.current_player
        if self.parent:
            self.parent.backpropagate(result)

def monte_carlo_tree_search(root, iterations=1000):
    for _ in range(iterations):
        node = root
        while node.children:
            node = node.select()
        node.expand()
        result = node.rollout()
        node.backpropagate(result)
    return root.select().state.last_move

# Gomoku game logic
class Gomoku:
    def __init__(self):
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = AI_PLAYER
        self.last_move = None

    def make_move(self, move):
        x, y = move
        self.board[x][y] = self.current_player
        self.last_move = move
        self.current_player = 3 - self.current_player

    def get_legal_moves(self):
        return [(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if self.board[x][y] == 0]

    def is_terminal(self):
        return self.get_winner() or not self.get_legal_moves()

    def get_winner(self):
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.board[x][y] and self.check_five_in_a_row(x, y):
                    return self.board[x][y]
        return None

    def check_five_in_a_row(self, x, y):
        player = self.board[x][y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx][ny] == player:
                    count += 1
                else:
                    break
            if count == 5:
                return True
        return False

# GUI
class GomokuGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gomoku")
        self.geometry("600x600")
        self.resizable(False, False)
        self.canvas = tk.Canvas(self, width=600, height=600, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.game = Gomoku()
        self.draw_board()

    def draw_board(self):
        for i in range(1, BOARD_SIZE):
            self.canvas.create_line(i * 40, 40, i * 40, 600 - 40)
            self.canvas.create_line(40, i * 40, 600 - 40, i * 40)
        if self.game.last_move:
            x, y = self.game.last_move
            color = "black" if self.game.board[x][y] == AI_PLAYER else "white"
            self.canvas.create_oval(x * 40 + 20 - 15, y * 40 + 20 - 15, x * 40 + 20 + 15, y * 40 + 20 + 15, fill=color)

    def on_click(self, event):
        if self.game.current_player == HUMAN_PLAYER and not self.game.is_terminal():
            x, y = event.x // 40, event.y // 40
            if self.game.board[x][y] == 0:
                self.game.make_move((x, y))
                self.draw_board()
                if self.game.is_terminal():
                    winner = self.game.get_winner()
                    if winner:
                        messagebox.showinfo("Game Over", "Player {} wins!".format(winner))
                    else:
                        messagebox.showinfo("Game Over", "It's a draw!")
                else:
                    self.after(100, self.ai_move)

    def ai_move(self):
        if self.game.current_player == AI_PLAYER and not self.game.is_terminal():
            move = monte_carlo_tree_search(MCTSNode(self.game))
            self.game.make_move(move)
            self.draw_board()
            if self.game.is_terminal():
                winner = self.game.get_winner()
                if winner:
                    messagebox.showinfo("Game Over", "Player {} wins!".format(winner))
                else:
                    messagebox.showinfo("Game Over", "It's a draw!")

if __name__ == "__main__":
    app = GomokuGUI()
    app.mainloop()
