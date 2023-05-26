#!/usr/bin/env python
#-*- coding: utf-8 -*-

import GUI
import tkinter as Tkinter
import sys

from board import Board, mctsAI

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == "ai":
        # ai play and train
        record_chess_board = Board(ai_first=True, has_ai=True)
        my_ai = mctsAI(len(record_chess_board.records), len(record_chess_board.records[0]))
        result = 0
        while not result:
            x_ai, y_ai = my_ai.play(record_chess_board)
            result, x_ai, y_ai = record_chess_board.chess(x_ai, y_ai)
            print(result)
        exit(0)

    window = Tkinter.Tk()
    gui_chess_board = GUI.Chess_Board_Frame(window)
    gui_chess_board.pack()
    window.mainloop()
