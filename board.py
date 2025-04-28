import copy
import numpy as np


class Board:

    def __init__(self, board_size=5):
        
        self.BLACK_KING = 7
        self.BLACK_QUEEN = 8
        self.BLACK_ROOK = 9
        self.BLACK_BISHOP = 10
        self.BLACK_KNIGHT = 11
        self.BLACK_PAWN = 12
        self.BLACK_PIECES = range(7, 13)
        
        self.EMPTY_FIELD = 0
        
        self.WHITE_PAWN = 1
        self.WHITE_KNIGHT = 2
        self.WHITE_BISHOP = 3
        self.WHITE_ROOK = 4
        self.WHITE_QUEEN = 5
        self.WHITE_KING = 6
        self.WHITE_PIECES = range(1, 7)
        

        self.board_size = board_size
        self.field = self.init_fields()

    def init_fields(self):
        """
        returns the new chess fields including the pieces (whites view)
        """
        field = np.zeros((self.board_size, self.board_size))
        for row in range(self.board_size):
            for col in range(self.board_size):
                if row == 1:
                    field[row][col] = self.BLACK_PAWN
                elif row == 3:
                    field[row][col] = self.WHITE_PAWN
                elif row == 0:
                    if col == 0:
                        field[row][col] = self.BLACK_ROOK
                    elif col == 1:
                        field[row][col] = self.BLACK_KNIGHT
                    elif col == 2:
                        field[row][col] = self.BLACK_BISHOP
                    elif col == 3:
                        field[row][col] = self.BLACK_QUEEN
                    elif col == 4:
                        field[row][col] = self.BLACK_KING
                elif row == 4:
                    if col == 0:
                        field[row][col] = self.WHITE_ROOK
                    elif col == 1:
                        field[row][col] = self.WHITE_KNIGHT
                    elif col == 2:
                        field[row][col] = self.WHITE_BISHOP
                    elif col == 3:
                        field[row][col] = self.WHITE_QUEEN
                    elif col == 4:
                        field[row][col] = self.WHITE_KING
        return field

    def board_reset(self):
        """
        resets the board
        """
        self.field = self.init_fields()

    def get_board_value(self, board):
        white_material_value, black_material_value = -100, -100
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece_type = board[row][col]
                if piece_type == self.WHITE_PAWN:
                    white_material_value += 1
                elif piece_type == self.BLACK_PAWN:
                    black_material_value += 1
                elif piece_type == self.WHITE_KNIGHT or piece_type == self.WHITE_BISHOP:
                    white_material_value += 3
                elif piece_type == self.BLACK_KNIGHT or piece_type == self.BLACK_BISHOP:
                    black_material_value += 3
                elif piece_type == self.WHITE_ROOK:
                    white_material_value += 5
                elif piece_type == self.BLACK_ROOK:
                    black_material_value += 5
                elif piece_type == self.WHITE_QUEEN:
                    white_material_value += 9
                elif piece_type == self.BLACK_QUEEN:
                    black_material_value += 9
                elif piece_type == self.WHITE_KING:
                    white_material_value += 100
                elif piece_type == self.BLACK_KING:
                    black_material_value += 100
        return white_material_value, black_material_value

    def print(self):
        """
        prints board in console
        """
        print(self.field)

    def switch_piece_color(self, piece_type):
        """
        returns the piece type with the opposite color (switches white <---> black)
        """
        return int((6.5 - piece_type) + 6.5) if piece_type != 0 else 0

    def reverse_board_view(self, field=None):
        """
        returns reversed board view from whites to blacks view (or otherwise)
        """
        if field is None:
            field = self.field
        reversed_board = copy.deepcopy((np.flipud(field)))  # kontrollieren, ob deepcopy nötig ist
        for i in range(self.board_size):
            for j in range(self.board_size):
                # change piece colors
                if reversed_board[i][j] != 0:
                    reversed_board[i][j] = int((6.5 - reversed_board[i][j]) + 6.5)
        return reversed_board

    def reverse_move(self, move):
        """
        reverses a move you would make on a reversed board to do the action on a normal board
        """
        from_j, from_i, to_j, to_i = move[0], move[1], move[2], move[3]
        return from_j, 4 - from_i, to_j, 4 - to_i
