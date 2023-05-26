import random

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pixel_x = 30 + 30 * self.x
        self.pixel_y = 30 + 30 * self.y

class Step:
    def __init__(self, count = 0) -> None:
        # 1 -> black
        # 2 -> white
        self.count = count
        self.color = self.count % 2 + 1

class Rule:
    @staticmethod
    def checkRow(records, x, y, player):
        count = 0
        min_ = max(y - 4, 0)
        max_ = min(y + 5, 15)
        for i in range(min_, max_):
            if records[x][i] is None:
                count = 0
            elif records[x][i].color == player:
                count = count + 1
            else:
                count = 0
            if count >= 5:
                return player
        return 0

    @staticmethod
    def checkCol(records, x, y, player):
        count = 0
        min_ = max(x - 4, 0)
        max_ = min(x + 5, 15)
        for i in range(min_, max_):
            if records[i][y] is None:
                count = 0
            elif records[i][y].color == player:
                count = count + 1
            else:
                count = 0
            if count >= 5:
                return player
        return 0

    @staticmethod
    def checkLeftBottomToRightTop(records, x, y, player):
        count = 0
        for i in range(-4, 5):
            x_ = x + i
            y_ = y - i
            if (x_ >= 0) and (y_ >= 0) and (x_ < 15) and (y_ < 15):
                if records[x_][y_] is None:
                    count = 0
                elif records[x_][y_].color == player:
                    count = count + 1
                else:
                    count = 0
                if count >= 5:
                    return player
        return 0
    
    @staticmethod
    def checkLeftTopToRightBottom(records, x, y, player):
        count = 0
        for i in range(-4, 5):
            x_ = x + i
            y_ = y + i
            if (x_ >= 0) and (y_ >= 0) and (x_ < 15) and (y_ < 15):
                if records[x_][y_] is None:
                    count = 0
                elif records[x_][y_].color == player:
                    count = count + 1
                else:
                    count = 0
                if count >= 5:
                    return player
        return 0

class randomAI:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
    
    def play(self, state):
        while True:
            x = random.randint(0, 14)
            y = random.randint(0, 14)
            if not state.records[x][y]:
                break
        return x, y

import os
import sys
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from core.brain.mcts import mcts

class mctsAI(randomAI):
    def __init__(self, width, height) -> None:
        super().__init__(width, height)
    
    def play(self, state):
        class gobangState:
            def __init__(self, state, currentPlayer, step, last_x, last_y) -> None:
                self.board = state
                self.currentPlayer = currentPlayer
                self.step = step
                self.last_x = last_x
                self.last_y = last_y

            def getCurrentPlayer(self): # 1 for maximiser, -1 for minimiser
                return self.currentPlayer

            def getPossibleActions(self):
                possibleActions = []
                search_bound = 1
                min_x = max(self.last_x - search_bound, 0)
                max_x = min(self.last_x + search_bound + 1, len(self.board))
                min_y = max(self.last_y - search_bound, 0)
                max_y = min(self.last_y + search_bound + 1, len(self.board))
                for i in range(min_x, max_x):
                    for j in range(min_y, max_y):
                        if self.board[i][j] is None:
                            possibleActions.append(gobangAction(self.currentPlayer, i, j))
                # print(possibleActions)
                return possibleActions

            def takeAction(self, action):
                newState = deepcopy(self)
                newState.board[action.x][action.y] = Step(self.step)
                newState.currentPlayer = self.currentPlayer * -1
                newState.step = self.step + 1
                newState.last_x = action.x
                newState.last_y = action.y
                return newState

            def isTerminal(self):
                records = self.board
                player = (self.step - 1) % 2 + 1
                x = self.last_x
                y = self.last_y
                if Rule.checkRow(records, x, y, player) or Rule.checkCol(records, x, y, player) or Rule.checkLeftBottomToRightTop(records, x, y, player) or Rule.checkLeftTopToRightBottom(records, x, y, player):
                    return 5
                return 0

            def getReward(self): # only needed for terminal states
                records = self.board
                player = (self.step - 1) % 2 + 1
                x = self.last_x
                y = self.last_y
                if Rule.checkRow(records, x, y, player) or Rule.checkCol(records, x, y, player) or Rule.checkLeftBottomToRightTop(records, x, y, player) or Rule.checkLeftTopToRightBottom(records, x, y, player):
                    return True
                return False

        class gobangAction:
            def __init__(self, player, x, y):
                self.player = player
                self.x = x
                self.y = y
            
            def __str__(self):
                return str((self.x, self.y))

            def __repr__(self):
                return str(self)

            def __eq__(self, other):
                return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

            def __hash__(self):
                return hash((self.x, self.y, self.player))
        
        state_ = gobangState(state.records, state.who_to_play(), state.count, 
                             state.last_play_x, state.last_play_y)
        searcher = mcts(timeLimit=3000)
        action = searcher.search(initialState=state_)

        return action.x, action.y

class Board:
    def __init__(self, ai_first, has_ai):
        self.count = 0
        self.ai = None
        self.last_play_x = None
        self.last_play_y = None
        self.records = [[None for i in range(15)] for j in range(15)]
        if has_ai:
            # self.ai = randomAI(15, 15)
            self.ai = mctsAI(15, 15)
        if ai_first:
            self.insert_record(7, 7)
            self.last_play_x = 7
            self.last_play_y = 7
            self.count += 1

    def has_record(self, x, y):
        return self.records[x][y] is not None

    def insert_record(self, x, y):
        self.last_play_x = x
        self.last_play_y = y
        self.records[x][y] = Step(self.count)
        mark = 'white' if self.records[x][y].color == 2 else 'black'
        print("%s, %d, %d" % (mark, x, y))

    def chess(self, x, y):
        x_ai, y_ai = None, None
        self.insert_record(x, y)
        # check with the result status
        result = self.check(x, y)
        if result != 0:
            return result, x_ai, y_ai
        self.count += 1
        if self.ai:
            x_ai, y_ai = self.ai.play(self)
            self.insert_record(x_ai, y_ai)
            result = self.check(x_ai, y_ai)
            self.count += 1
        return result, x_ai, y_ai

    def who_to_play(self):
        return self.count % 2 + 1

    def check(self, x, y):
        records = self.records
        player = self.who_to_play()
        if Rule.checkRow(records, x, y, player) or Rule.checkCol(records, x, y, player) or Rule.checkLeftBottomToRightTop(records, x, y, player) or Rule.checkLeftTopToRightBottom(records, x, y, player):
            return player
        return 0
