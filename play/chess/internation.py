 
# Import necessary libraries
import tkinter as tk
import torch
import numpy as np
import random

# Define the chess board
class ChessBoard:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.game_over = False

    def get_valid_moves(self):
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    valid_moves.append((i, j))
        return valid_moves

    def make_move(self, move):
        if self.board[move[0]][move[1]] != 0:
            return False
        self.board[move[0]][move[1]] = self.current_player
        self.current_player = 3 - self.current_player
        return True

    def check_win(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    continue
                if i + 4 <= self.board_size and len(set(self.board[i:i+5,j])) == 1:
                    return self.board[i][j]
                if j + 4 <= self.board_size and len(set(self.board[i,j:j+5])) == 1:
                    return self.board[i][j]
                if i + 4 <= self.board_size and j + 4 <= self.board_size and len(set([self.board[i+k][j+k] for k in range(5)])) == 1:
                    return self.board[i][j]
                if i + 4 <= self.board_size and j - 4 >= -1 and len(set([self.board[i+k][j-k] for k in range(5)])) == 1:
                    return self.board[i][j]
        if len(self.get_valid_moves()) == 0:
            return 0
        return -1

# Define the GUI
class ChessGUI:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = ChessBoard(board_size)
        self.root = tk.Tk()
        self.root.title("International Chess")
        self.canvas = tk.Canvas(self.root, width=board_size*50, height=board_size*50)
        self.canvas.pack()
        self.draw_board()
        self.canvas.bind("<Button-1>", self.click)

    def draw_board(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i+j) % 2 == 0:
                    self.canvas.create_rectangle(j*50, i*50, (j+1)*50, (i+1)*50, fill="white")
                else:
                    self.canvas.create_rectangle(j*50, i*50, (j+1)*50, (i+1)*50, fill="gray")
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board.board[i][j] == 1:
                    self.canvas.create_oval(j*50+10, i*50+10, (j+1)*50-10, (i+1)*50-10, fill="black")
                elif self.board.board[i][j] == 2:
                    self.canvas.create_oval(j*50+10, i*50+10, (j+1)*50-10, (i+1)*50-10, fill="white")

    def click(self, event):
        if self.board.game_over:
            return
        x, y = event.x // 50, event.y // 50
        if not self.board.make_move((y, x)):
            return
        self.canvas.delete("all")
        self.draw_board()
        winner = self.board.check_win()
        if winner != -1:
            self.board.game_over = True
            if winner == 0:
                tk.messagebox.showinfo("Game Over", "Tie!")
            elif winner == 1:
                tk.messagebox.showinfo("Game Over", "Black wins!")
            else:
                tk.messagebox.showinfo("Game Over", "White wins!")
        else:
            self.ai_move()

    def ai_move(self):
        valid_moves = self.board.get_valid_moves()
        if len(valid_moves) == 0:
            return
        if len(valid_moves) == 1:
            self.board.make_move(valid_moves[0])
            self.canvas.delete("all")
            self.draw_board()
            winner = self.board.check_win()
            if winner != -1:
                self.board.game_over = True
                if winner == 0:
                    tk.messagebox.showinfo("Game Over", "Tie!")
                elif winner == 1:
                    tk.messagebox.showinfo("Game Over", "Black wins!")
                else:
                    tk.messagebox.showinfo("Game Over", "White wins!")
            return
        # Use Monte Carlo tree search to find the best move
        root = Node(self.board, None, None)
        for i in range(1000):
            node = root
            temp_board = self.board
            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                temp_board.make_move(node.move)
            # Expansion
            if node.untried_moves != []:
                m = random.choice(node.untried_moves)
                temp_board.make_move(m)
                node = node.add_child(m, temp_board)
            # Simulation
            while temp_board.get_valid_moves() != []:
                temp_board.make_move(random.choice(temp_board.get_valid_moves()))
            #
 
# Define the Node class for Monte Carlo tree search
class Node:
    def __init__(self, board, move, parent):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = board.get_valid_moves()

    def select_child(self):
        return max(self.children, key=lambda c: c.wins/c.visits + np.sqrt(2*np.log(self.visits)/c.visits))

    def add_child(self, move, board):
        n = Node(board, move, self)
        self.untried_moves.remove(move)
        self.children.append(n)
        return n

    def update(self, result):
        self.visits += 1
        self.wins += result

    def get_value(self):
        if self.visits == 0:
            return 0
        return self.wins/self.visits

    def get_best_move(self):
        return max(self.children, key=lambda c: c.visits).move

    def get_child_visits(self):
        return [c.visits for c in self.children]

    def get_child_values(self):
        return [c.get_value() for c in self.children]

    def get_child_moves(self):
        return [c.move for c in self.children]

    def get_child_win_rates(self):
        return [c.wins/c.visits for c in self.children]

    def get_child_boards(self):
        return [c.board for c in self.children]

    def get_child(self, move):
        for c in self.children:
            if c.move == move:
                return c
        return None

# Define the neural network for the AI player
class ChessNet(torch.nn.Module):
    def __init__(self, board_size=8):
        super(ChessNet, self).__init__()
        self.board_size = board_size
        self.conv1 = torch.nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(128*board_size*board_size//64, 256)
        self.fc2 = torch.nn.Linear(256, board_size*board_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(-1, 128*self.board_size*self.board_size//64)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, board):
        x = torch.tensor(board.board).unsqueeze(0).unsqueeze(0).float()
        y = self.forward(x)
        return y.detach().numpy().reshape(-1)

    def train(self, boards, targets, epochs=10, batch_size=32):
        optimizer = torch.optim.Adam(self.parameters())
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            for i in range(0, len(boards), batch_size):
                x = torch.tensor([b.board for b in boards[i:i+batch_size]]).unsqueeze(1).float()
                y = torch.tensor(targets[i:i+batch_size]).float()
                y_pred = self.forward(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

# Define the AI player using Monte Carlo tree search and neural network
class AIPlayer:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.net = ChessNet(board_size)
        self.net.load_state_dict(torch.load("chess_net.pth"))

    def get_move(self, board):
        root = Node(board, None, None)
        for i in range(1000):
            node = root
            temp_board = ChessBoard(self.board_size)
            temp_board.board = np.copy(board.board)
            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                temp_board.make_move(node.move)
            # Expansion
            if node.untried_moves != []:
                m = random.choice(node.untried_moves)
                temp_board.make_move(m)
                node = node.add_child(m, temp_board)
            # Simulation
            while temp_board.get_valid_moves() != []:
                if len(temp_board.get_valid_moves()) == 1:
                    temp_board.make_move(temp_board.get_valid_moves()[0])
                else:
                    y = self.net.predict(temp_board)
                    m = np.argmax(y)
                    temp_board.make_move((m//self.board_size, m%self.board_size))
            # Backpropagation
            while node != None:
                node.update(temp_board.check_win())
                node = node.parent
        return root.get_best_move()

    def train(self, boards, targets, epochs=10, batch_size=32):
        self.net.train(boards, targets, epochs, batch_size)
        torch.save(self.net.state_dict(), "chess_net.pth")

# Define the main function
def main():
    gui = ChessGUI()
    gui.root.mainloop()

if __name__ == "__main__":
    main()

