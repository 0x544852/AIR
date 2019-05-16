import numpy as np


class SlotGame:
    board = None
    board_size = 0

    def __init__(self, board_size=4):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_size)

    def play(self, cell):
        # returns a tuple: (reward, game_over?)
        if self.board[cell] == 0:
            self.board[cell] = 1
            game_over = len(np.where(self.board == 0)[0]) == 0
            return 1, game_over
        else:
            return -1, False
