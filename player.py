from board import HexBoard
import math
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from itertools import product
import networkx as nx
from copy import deepcopy
import random

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("¡Implementa este método!")

class RamdomPlayer(Player):
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    def play(self, board: HexBoard) -> tuple:
        moves = board.get_possible_moves()
        return moves.pop(np.random.randint(0, high=len(moves))) 

class NormalPlayer(Player):
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    
    def play(self, board: HexBoard) -> tuple:                
        _ , move = self.minmax_alfabeta(board, self.player_id, True)
        return move     
    
    def valid(self, board, x, y):
            return x >= 0 and x < board.size and y >= 0 and y < board.size
    
    def minDist(self, A, B):
            m = float("inf")
            for coor_a in A:
                for coor_b in B:
                    m = min(m, math.sqrt( (coor_a[0] - coor_b[0])**2 + (coor_a[1] - coor_b[1])**2))
            return m
    
    def evaluate_board(self, board: HexBoard, player_id: int) -> int:
        ds = DisjointSet()
        
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == player_id:
                    ds.add((i,j))
                    
                    right = (i, j+1)
                    down_right = (i+1, j)
                    
                    if self.valid(board,right[0], right[1]) and board.board[right[0]][right[1]] == player_id:
                        ds.add(right)
                        ds.merge((i,j), right)
                    
                    if self.valid(board,down_right[0], down_right[1]) and board.board[down_right[0]][down_right[1]] == player_id:
                        ds.add(down_right)
                        ds.merge((i,j), down_right)
        
        G = nx.Graph()
        G.add_node(-1)
        G.add_node(board.size)
        
        length = len(ds.subsets())
        for i in range(length):
            for j in range(i + 1, length):
                m = self.minDist(ds.subsets()[i], ds.subsets()[j])
                G.add_edge(i, j, weight=m)
            
            index = 1 if player_id == 1 else 0
            S = [t[index] for t in ds.subsets()[i]]
            G.add_edge(-1, i, weight=min(S))
            G.add_edge(board.size, i, weight=(board.size - max(S) - 1))
                    
        return -nx.shortest_path_length(G, -1, board.size, "weight")
         
    def minmax_alfabeta(self, board: HexBoard, player_id: int, is_max: bool, depth: int = 5, alfa: int = -math.inf, beta: int = math.inf):
        
        if depth == 0:
            return (self.evaluate_board(board, player_id), None)                

        for i in board.get_possible_moves():
            copy = board.clone()
            copy.place_piece(i[0], i[1], player_id)
            
            if copy.check_connection(player_id) :
                return ( math.inf if self.player_id == player_id else -math.inf , i)
            
            old_value = (alfa, beta)
            
            if is_max:
                value, move = self.minmax_alfabeta(copy, 3-player_id, False, depth - 1, alfa, beta) 
                alfa = max(alfa, value) 
            else:
                value, move = self.minmax_alfabeta(copy, player_id, True, depth - 1, alfa, beta)
                beta = min(beta, value)
                                
            if beta <= alfa:
                break
            
            if (alfa, beta) != old_value or not move:
                move = i 
            
        return (alfa if is_max else beta, move)

class CrazyPlayer(Player):
    def __init__(self, player_id: int, depth=5, divisor=1):
        self.player_id = player_id  # Tu identificador (1 o 2)
        self.depth = depth
        self.divisor = divisor
    
    def play(self, board: HexBoard) -> tuple:                
        _ , move = self.minmax_alfabeta(board, self.player_id, True, depth=self.depth)
        return move     
    
    def valid(self,board, x, y):
            return x >= 0 and x < board.size and y >= 0 and y < board.size
    
    def minDist(self, A, B):
            m = float("inf")
            for coor_a in A:
                for coor_b in B:
                    m = min(m, math.sqrt( (coor_a[0] - coor_b[0])**2 + (coor_a[1] - coor_b[1])**2))
            return m
    
    def evaluate_board(self, board: HexBoard, player_id: int) -> int:
        ds = DisjointSet()
        
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == player_id:
                    ds.add((i,j))
                    
                    right = (i, j+1)
                    down_right = (i+1, j)
                    
                    if self.valid(board, right[0], right[1]) and board.board[right[0]][right[1]] == player_id:
                        ds.add(right)
                        ds.merge((i,j), right)
                    
                    if self.valid(board, down_right[0], down_right[1]) and board.board[down_right[0]][down_right[1]] == player_id:
                        ds.add(down_right)
                        ds.merge((i,j), down_right)
        
        G = nx.Graph()
        G.add_node(-1)
        G.add_node(board.size)
        
        length = len(ds.subsets())
        for i in range(length):
            for j in range(i + 1, length):
                m = self.minDist(ds.subsets()[i], ds.subsets()[j])
                G.add_edge(i, j, weight=m)
            
            index = 1 if player_id == 1 else 0
            S = [t[index] for t in ds.subsets()[i]]
            G.add_edge(-1, i, weight=min(S))
            G.add_edge(board.size, i, weight=(board.size - max(S) - 1))
                    
        return -nx.shortest_path_length(G, -1, board.size, "weight")
         
    def minmax_alfabeta(self, board: HexBoard, player_id: int, is_max: bool, depth: int = 7, alfa: int = -math.inf, beta: int = math.inf):
        
        if depth == 0:
            return (self.evaluate_board(board, player_id), None)                

        #for i in board.get_possible_moves():
        for i in self.random_pos(board):    
            copy = board.clone()
            copy.place_piece(i[0], i[1], player_id)
            
            if copy.check_connection(player_id) :
                return ( math.inf if self.player_id == player_id else -math.inf , i)
            
            old_value = (alfa, beta)
            
            if is_max:
                value, move = self.minmax_alfabeta(copy, 3-player_id, False, depth - 1, alfa, beta) 
                alfa = max(alfa, value) 
            else:
                value, move = self.minmax_alfabeta(copy, player_id, True, depth - 1, alfa, beta)
                beta = min(beta, value)
                                
            if beta <= alfa:
                break
            
            if (alfa, beta) != old_value or not move:
                move = i 
            
        return (alfa if is_max else beta, move)
    
    def random_pos(self, board: HexBoard):
        possible_mov_length = board.size**2 - len(board.player_positions[1]) - len(board.player_positions[2])
        possible_mov = board.get_possible_moves()
        
        if possible_mov_length > 5:
        
            count = np.exp(self.divisor/possible_mov_length)
            while count > 0 and len(possible_mov) > 0:
                count -= 1
                yield possible_mov.pop(np.random.randint(0, high=len(possible_mov)))
            
        else:
            for i in board.get_possible_moves():
                yield i
        
class MonteCarloPlayer(Player):
    def __init__(self, player_id, num_simulation=100):
        super().__init__(player_id)
        self.num_simulation = num_simulation
        
    def play(self, board):
        
        best_move = (-math.inf, None)
        possible_moves = board.get_possible_moves()
        
        for i in range(len(possible_moves)):
            move = possible_moves[i]
            
            copy = board.clone()
            copy.place_piece(move[0], move[1], self.player_id)
            
            if copy.check_connection(self.player_id):
                return move
            
            n = 0
            possible_moves_copy = deepcopy(possible_moves)
            possible_moves_copy.pop(i)
            
            for _ in range(int(math.sqrt(len(possible_moves) - 1))):                
                n += self.monte_carlo_search(deepcopy(copy), 3 - self.player_id, deepcopy(possible_moves_copy))

            if n > best_move[0]:
                best_move = (n, move)
                
        return best_move[1]    
    
    def monte_carlo_search(self, board: HexBoard, player_id: int, possible_moves: list):
        
        move = possible_moves.pop(np.random.randint(0, len(possible_moves)))
        board.place_piece(move[0], move[1], player_id)
        
        if len(possible_moves) == 0 or board.check_connection(player_id):
            return 1 if player_id == self.player_id else 0
    
        return self.monte_carlo_search(board, 3 - player_id, possible_moves)
        