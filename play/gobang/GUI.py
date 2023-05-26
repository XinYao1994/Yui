#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tkinter as Tkinter
import threading
import math

from sympy import true
from board import Point
from board import Board

class Chess_Board_Canvas(Tkinter.Canvas):
    #棋盘绘图板,继承自Tkinter.Canvas类
    def __init__(self, master=None, height=0, width=0):
        Tkinter.Canvas.__init__(self, master, height=height, width=width)
        self.step_record_chess_board = Board(ai_first=False, has_ai=False)
        #初始化计步器对象
        self.init_chess_board_points()    #画点
        self.init_chess_board_canvas()    #绘制棋盘

    def init_chess_board_points(self):
        self.chess_board_points = [[None for i in range(15)] for j in range(15)]
        for i in range(15):
            for j in range(15):
                self.chess_board_points[i][j] = Point(i, j); #棋盘坐标向像素坐标转化
                if self.step_record_chess_board.records[i][j] is not None:
                    self.create_oval(self.chess_board_points[i][j].pixel_x-10, 
                                     self.chess_board_points[i][j].pixel_y-10, 
                                     self.chess_board_points[i][j].pixel_x+10, 
                                     self.chess_board_points[i][j].pixel_y+10, 
                                     fill='black')

    def init_chess_board_canvas(self):
        for i in range(15):  
            self.create_text(self.chess_board_points[i][0].pixel_x, self.chess_board_points[i][0].pixel_y - 15, text=str(i))
            self.create_line(self.chess_board_points[i][0].pixel_x, self.chess_board_points[i][0].pixel_y, self.chess_board_points[i][14].pixel_x, self.chess_board_points[i][14].pixel_y)
        for j in range(15):  
            self.create_text(self.chess_board_points[0][j].pixel_x - 15, self.chess_board_points[0][j].pixel_y, text=str(j))
            self.create_line(self.chess_board_points[0][j].pixel_x, self.chess_board_points[0][j].pixel_y, self.chess_board_points[14][j].pixel_x, self.chess_board_points[14][j].pixel_y)
        for i in range(15):  
            for j in range(15):
                r = 1
                self.create_oval(self.chess_board_points[i][j].pixel_x-r, self.chess_board_points[i][j].pixel_y-r, self.chess_board_points[i][j].pixel_x+r, self.chess_board_points[i][j].pixel_y+r);

    def chess_result(self, x, y):
        result, x_ai, y_ai = self.step_record_chess_board.chess(x, y)
        if x_ai is not None:
            ai_color = 'black'
            if self.step_record_chess_board.who_to_play() == 1:
                ai_color = 'white'
            self.create_oval(self.chess_board_points[x_ai][y_ai].pixel_x-10, 
                             self.chess_board_points[x_ai][y_ai].pixel_y-10, 
                             self.chess_board_points[x_ai][y_ai].pixel_x+10, 
                             self.chess_board_points[x_ai][y_ai].pixel_y+10, 
                             fill=ai_color)

        # done
        if result == 1:
            self.create_text(240, 550, text='the black wins')
            self.unbind('<Button-1>')
        elif result == 2:
            self.create_text(240, 550, text='the white wins')
            self.unbind('<Button-1>')

    def click_chess(self, event): 
        for i in range(15):
            for j in range(15):
                square_distance = math.pow((event.x - self.chess_board_points[i][j].pixel_x), 2) + math.pow((event.y - self.chess_board_points[i][j].pixel_y), 2)
                if (square_distance <= 200) and (not self.step_record_chess_board.has_record(i, j)): #距离小于14并且没有落子
                    if self.step_record_chess_board.who_to_play() == 1:
                        self.create_oval(self.chess_board_points[i][j].pixel_x-10, 
                                         self.chess_board_points[i][j].pixel_y-10, 
                                         self.chess_board_points[i][j].pixel_x+10, 
                                         self.chess_board_points[i][j].pixel_y+10, 
                                         fill='black')
                    elif self.step_record_chess_board.who_to_play() == 2:
                        self.create_oval(self.chess_board_points[i][j].pixel_x-10, 
                                         self.chess_board_points[i][j].pixel_y-10, 
                                         self.chess_board_points[i][j].pixel_x+10, 
                                         self.chess_board_points[i][j].pixel_y+10, 
                                         fill='white')
                    
                    #chess and 判断是否有五子连珠
                    # self.chess_result(i, j)
                    t = threading.Thread(target=self.chess_result, args=(i, j))
                    t.setDaemon(True)
                    t.start()
                    # t.join()


class Chess_Board_Frame(Tkinter.Frame):
    def __init__(self, master=None):
        Tkinter.Frame.__init__(self, master)
        self.create_widgets()

    def create_widgets(self):
        self.chess_board_label_frame = Tkinter.LabelFrame(self, text="Chess Board", padx=5, pady=5)
        self.chess_board_canvas = Chess_Board_Canvas(self.chess_board_label_frame, height=600, width=480)
        self.chess_board_canvas.bind('<Button-1>', self.chess_board_canvas.click_chess)
        self.chess_board_label_frame.pack()
        self.chess_board_canvas.pack()
