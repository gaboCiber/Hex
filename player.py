from board import HexBoard
import math
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from itertools import product
import networkx as nx

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    def valid(self, x, y):
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
                    
                    if self.valid(right[0], right[1]) and board.board[right[0]][right[1]] == player_id:
                        ds.add(right)
                        ds.merge((i,j), right)
                    
                    if self.valid(down_right[0], down_right[1]) and board.board[down_right[0]][down_right[1]] == player_id:
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
      
    
    
    def play(self, board: HexBoard) -> tuple:
                    
        _ , move = self.minmax_alfabeta(board, self.player_id, True)
        return move     
        
        
P1 = Player(1)
P2 = Player(2)
board = HexBoard(6)
board.print()

i = 0
while True:
    
    P = P1 if i % 2 == 0 else P2
    
    print(f"\nJugador {P.player_id}")
    move = input("> ").split(' ') if i % 2 == 0 else P.play(board)
    board.place_piece(int(move[0]), int(move[1]), P.player_id)
    board.print()
    
    if board.check_connection(P.player_id):
        print(f"GANO {P.player_id}")
        break

    i+=1


        
    
    
    