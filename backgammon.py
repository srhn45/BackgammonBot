import numpy as np
import random
import torch

class BackgammonBoard:
    def __init__(self):
        self.board = np.zeros(24, dtype=int)
        self.board[0] = 2    # Player 1's pieces
        self.board[11] = 5   
        self.board[16] = 3   
        self.board[18] = 5  
 
        self.board[23] = -2  # Player 2's pieces
        self.board[12] = -5  
        self.board[7] = -3   
        self.board[5] = -5   

    def display(self):
        top_row = " ".join(f"{self.board[i]:2d}" for i in range(12, 24))
        bottom_row = " ".join(f"{self.board[i]:2d}" for i in reversed(range(12)))

        print("          Top (Points 13-24)")
        print(top_row)
        print("-" * 35)
        print(bottom_row)
        print("        Bottom (Points 12-1)")
        
    def clone(self):
        new_board = BackgammonBoard()
        new_board.board = self.board.copy()
        return new_board

class Game:
    def __init__(self, starting_board: BackgammonBoard = None):
        if starting_board:
            self.board = starting_board
        else:
            self.board = BackgammonBoard()

        self.current_player = 1
        self.game_over = False

        self.dice = [0, 0]
        self.broken_pieces = {1: 0, -1: 0}
        self.collected_pieces = {1: 0, -1: 0}
        self.legal_moves = []

    def roll_dice(self):
        die1 = np.random.randint(1, 7)
        die2 = np.random.randint(1, 7)
        self.dice = [die1, die2] if die1 != die2 else [die1, die1, die1, die1]
        return self.dice

    def play_one_move(self, start, end, die):
        if (start, end, die) not in self.legal_moves:
            raise ValueError("Illegal move attempted.")
        
        if self.broken_pieces[self.current_player] > 0:
            if start != -1:
                raise ValueError("Must place broken pieces first.")
            
            if self.current_player == 1:
                end = die-1

            self.broken_pieces[self.current_player] -= 1
            if self.board.board[end] * -self.current_player == 1:
                self.broken_pieces[-self.current_player] += 1
                self.board.board[end] = self.current_player
            else:
                self.board.board[end] += self.current_player
        else:
            if end >= 24 or end < 0:
                self.collected_pieces[self.current_player] += 1
                self.board.board[start] -= self.current_player
            else:
                self.board.board[start] -= self.current_player
                if self.board.board[end] * -self.current_player == 1:
                    self.broken_pieces[-self.current_player] += 1
                    self.board.board[end] = self.current_player
                else:
                    self.board.board[end] += self.current_player
        
        self.dice.remove(die)
    
    def switch_player(self):
        self.current_player *= -1
        self.roll_dice()
        self.get_legal_moves()

    def get_legal_moves(self):
        self.legal_moves = []
        moves = self.dice.copy()
        if self.broken_pieces[self.current_player] > 0:
            for die in set(moves):
                if self.board.board[die-1 if self.current_player == 1 else 24 - die] * -self.current_player < 2:
                    self.legal_moves.append((-1, die if self.current_player == 1 else 24 - die, die))
            return self.legal_moves

        if self.current_player == 1:
            all_home = all(v <= 0 for v in self.board.board[0:18])
        else:
            all_home = all(v >= 0 for v in self.board.board[6:24])
        if all_home:
            for i in range(18, 24) if self.current_player == 1 else range(0, 6):
                if self.board.board[i] * self.current_player > 0:
                    for die in set(moves):
                        end = i + die * self.current_player
                        if end >= 24 or end < 0:
                            self.legal_moves.append((i, end, die))
                        elif self.board.board[end] * -self.current_player < 2:
                            self.legal_moves.append((i, end, die))
        else:
            for i in range(24):
                if self.board.board[i] * self.current_player > 0:
                    for die in set(moves):
                        end = i + die * self.current_player
                        if self.board.board[i] * self.current_player > 0 and (0 <= end < 24) and self.board.board[end] * -self.current_player < 2:
                            self.legal_moves.append((i, end, die))
        
        return self.legal_moves
    
    def check_game_over(self):
        if self.collected_pieces[1] == 15:
            self.game_over = True
            return 1
        elif self.collected_pieces[-1] == 15:
            self.game_over = True
            return -1
        return 0
    
    def move_translator(self, move):
        if self.broken_pieces[self.current_player] == 0:
            start, end, die = move//6, move//6 + self.current_player*((move%6)+1), (move%6)+1
            return start, end, die
        else:
            start = -1
            end, die = move+1 if self.current_player == 1 else 24-(move+1), move
    
    def get_input_matrix(self):
        input_matrix = np.zeros(24+1+2+2+4) # 24 points + 1 for current player + 2 for broken pieces + 2 for collected pieces + 4 for dice
        for i in range(24):
            input_matrix[i] = self.board.board[i]
        
        input_matrix[24] = self.current_player
        input_matrix[25] = self.broken_pieces[1]
        input_matrix[26] = self.broken_pieces[-1]
        input_matrix[27] = self.collected_pieces[1]
        input_matrix[28] = self.collected_pieces[-1]
        for i in range(4):
            input_matrix[29+i] = self.dice[i] if i < len(self.dice) else 0

        return torch.tensor(input_matrix, dtype=torch.float32)
    
    def clone(self):
        new_game = Game(starting_board=self.board.clone())
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.dice = self.dice.copy()
        new_game.broken_pieces = self.broken_pieces.copy()
        new_game.collected_pieces = self.collected_pieces.copy()
        return new_game