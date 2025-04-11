from collections import deque
import scipy
from board import HexBoard
import math
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial import distance
import networkx as nx
from copy import deepcopy
import random
import concurrent.futures

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("¡Implementa este método!")

class MonteCarloPlayer(Player):
    def __init__(self, player_id, n_simulations=30):
        super().__init__(player_id)
        self.n_simulations = n_simulations

    def play(self, board):
        best_score = -math.inf
        best_move = None
        possible_moves = board.get_possible_moves()

        for i, move in enumerate(possible_moves):
            copy = board.clone()
            copy.place_piece(move[0], move[1], self.player_id)

            if copy.check_connection(self.player_id):
                return move

            wins = 0
            remaining_moves = deepcopy(possible_moves)
            remaining_moves.pop(i)

            for _ in  range(int(math.sqrt(len(possible_moves) - 1))):#range(self.n_simulations):
                sim_board = copy.clone()
                sim_moves = deepcopy(remaining_moves)
                random.shuffle(sim_moves)
                winner = self.simulate_random_game(sim_board, sim_moves, 3 - self.player_id)
                if winner == self.player_id:
                    wins += 1

            if wins > best_score:
                best_score = wins
                best_move = move

        return best_move

    def simulate_random_game(self, board: HexBoard, moves:list, current_player: int):
        for move in moves:
            board.place_piece(move[0], move[1], current_player)
            if board.check_connection(current_player):
                return current_player
            current_player = 3 - current_player
            
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
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)
    
    def play(self, board: HexBoard) -> tuple:
                        
        _ , move = self.minmax_alfabeta(board, self.player_id, True, depth=self.calculate_depth(board))
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

        for i in self.next_moves(board):    
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
    
    def next_moves(self, board: HexBoard):
        possible_mov = board.get_possible_moves()
        
        if len(possible_mov) > 5:
        
            count = math.sqrt(len(possible_mov))
            while count > 0 and len(possible_mov) > 0:
                count -= 1
                yield possible_mov.pop(np.random.randint(0, high=len(possible_mov)))
            
        else:
            for i in board.get_possible_moves():
                yield i

    def calculate_depth(self, board: HexBoard):
        possible_moves_lenght = len(board.get_possible_moves())
        if possible_moves_lenght > 50:
            return 1
        if possible_moves_lenght > 20:
            return 2
        if possible_moves_lenght > 10:
            return 3
        if possible_moves_lenght > 5:
            return 4
        return 7

class InTerapyPlayer(Player):
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)
    
    def play(self, board: HexBoard) -> tuple:
                        
        _ , move = self.minmax_alfabeta(board, self.player_id, True, depth=self.calculate_depth(board))
        return move     
    
    def valid(self,board, x, y):
            return x >= 0 and x < board.size and y >= 0 and y < board.size
    
    def minDist(self, A, B):
            m = float("inf")
            for coor_a in A:
                for coor_b in B:
                    m = min(m, math.sqrt( (coor_a[0] - coor_b[0])**2 + (coor_a[1] - coor_b[1])**2))
            return m
    
    def break_rival_bridges(self, board: HexBoard, player_id: int) -> int:
        x = [0 , 0, -1, -1, 1, 1]
        y = [-1, 1, 0, 1, -1, 0]
        
        broken_bridges = 0
        for i in range(1, board.size - 1):
            for j in range(1, board.size - 1):
                if board.board[i][j] == player_id:
                    c = 0
                    for k in range(6):
                        new_x = i + x[k]
                        new_y = j + y[k]
                        if self.valid(board, new_x, new_y) and board.board[new_x][new_y] == 3-player_id:
                            c+=1
                    broken_bridges += math.comb(c,2)
        
        return broken_bridges
    
    def evaluate_board(self, board: HexBoard, player_id: int) -> int:
        return self.shortest_path(board, player_id) + self.center_plays(board, player_id) + self.break_rival_bridges(board, player_id)
    
    def center_plays(self, board: HexBoard, player_id: int) -> int:
        c = 0
        middle = int(board.size / 2) - 1
        for i in range(board.size):
            if board.size > 2 and board.board[i][middle] == player_id:
                c+=1
            if board.size > 2 and board.board[middle][i] == player_id:
                c+=1
            if board.size > 3 and board.board[i][middle+1] == player_id:
                c+=1
            if board.size > 3 and board.board[middle+1][i] == player_id:
                c+=1
        
        return c
    
    def shortest_path(self, board: HexBoard, player_id: int):
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
        
        if not ds:
            return 0
        
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

        for i in self.next_moves(board):    
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
    
    def next_moves(self, board: HexBoard):
        possible_mov = board.get_possible_moves()
        
        if len(possible_mov) > 7:
        
            count = math.sqrt(len(possible_mov))
            while count > 0 and len(possible_mov) > 0:
                count -= 1
                yield possible_mov.pop(np.random.randint(0, high=len(possible_mov)))
            
        else:
            for i in board.get_possible_moves():
                yield i

    def calculate_depth(self, board: HexBoard):
        possible_moves_lenght = len(board.get_possible_moves())
        if possible_moves_lenght > 50:
            return 1
        if possible_moves_lenght > 20:
            return 2
        if possible_moves_lenght > 10:
            return 3
        if possible_moves_lenght > 5:
            return 4
        return 7
            
class RehabPlayer(Player):
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)
    
    def play(self, board: HexBoard, time_limit: float = None) -> tuple:
        if time_limit is None:
            # No hay límite
            _, move = self.max_alfabeta(board, self.player_id, self.calculate_depth(board))
            return move
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.max_alfabeta, board, self.player_id, self.calculate_depth(board))
            try:
                _, move = future.result(timeout=time_limit)
                return move
            except concurrent.futures.TimeoutError:
                return board.get_possible_moves()[0]      
    
    def max_alfabeta(self, board: HexBoard, player_id: int, depth: int, alfa: int = -math.inf, beta: int = math.inf):
        if depth == 0:
            return (self.evaluate_board(board, player_id), None)
        
        v = -math.inf
        move = None
        for a in self.next_moves(board):   
            
            copy = board.clone()
            copy.place_piece(a[0], a[1], player_id)
            
            if copy.check_connection(player_id) :
                return ( math.inf if self.player_id == player_id else -math.inf , a)
            
            v2, a2 = self.min_alfabeta(copy, 3-player_id, depth-1, alfa, beta)
            
            if v2 > v:
                v, move = v2, a
                alfa = max(alfa, v)
            
            if not move:
                move = a
            
            if v >= beta:
                return (v, move)
        
        return v, move
            
    def min_alfabeta(self, board: HexBoard, player_id: int, depth: int, alfa: int = -math.inf, beta: int = math.inf):
        
        if depth == 0:
            return (self.evaluate_board(board, player_id), None)
        
        v = math.inf
        move = None
        
        for a in self.next_moves(board):   
            
            copy = board.clone()
            copy.place_piece(a[0], a[1], player_id)
            
            if copy.check_connection(player_id) :
                return ( math.inf if self.player_id == player_id else -math.inf , a)
            
            v2, a2 = self.max_alfabeta(copy, 3-player_id, depth-1, alfa, beta)
            
            if v2 < v:
                v, move = v2, a
                beta = max(beta, v)
            
            if not move:
                move = a
            
            if v <= alfa:
                return (v, move)
        
        return v, move
   
    
    def evaluate_board(self, board: HexBoard, player_id: int) -> int:      
        return (3 * self.shortest_path(board, player_id) + 
                10 * self.center_plays(board, player_id) + 
                3 * self.break_rival_bridges(board, player_id) +
                6 * self.rival_influence(board, player_id) 
                )
    
    def valid(self,board, x, y):
            return x >= 0 and x < board.size and y >= 0 and y < board.size
    
    def minDist(self, A, B):
        m = float("inf")
        for coor_a in A:
            for coor_b in B:
                m = min(m, math.sqrt( (coor_a[0] - coor_b[0])**2 + (coor_a[1] - coor_b[1])**2))
        return m
    
    def rival_influence(self, board: HexBoard, player_id: int) -> int:
        c = 0
        
        x = [0 , 0, -1, -1, 1, 1]
        y = [-1, 1, 0, 1, -1, 0]
        
        for i in range(1, board.size - 1):
            for j in range(1, board.size - 1):
                if board.board[i][j] == 0:
                    for k in range(6):
                        new_x = i + x[k]
                        new_y = j + y[k]
                        if self.valid(board, new_x, new_y) and board.board[new_x][new_y] == 3-player_id:
                            c+=1  

        return -c                
    
    def break_rival_bridges(self, board: HexBoard, player_id: int) -> int:
        x = [0 , 0, -1, -1, 1, 1]
        y = [-1, 1, 0, 1, -1, 0]
        
        broken_bridges = 0
        for i in range(1, board.size - 1):
            for j in range(1, board.size - 1):
                if board.board[i][j] == player_id:
                    c = 0
                    for k in range(6):
                        new_x = i + x[k]
                        new_y = j + y[k]
                        if self.valid(board, new_x, new_y) and board.board[new_x][new_y] == 3-player_id:
                            c+=1
                    broken_bridges += math.comb(c,2)
        
        return broken_bridges
       
    def center_plays(self, board: HexBoard, player_id: int) -> int:
        c = 0
        middle = int(board.size / 2) - 1
        for i in range(board.size):
            if board.size > 2 and board.board[i][middle] == player_id:
                c+=1
            if board.size > 2 and board.board[middle][i] == player_id:
                c+=1
            if board.size > 3 and board.board[i][middle+1] == player_id:
                c+=1
            if board.size > 3 and board.board[middle+1][i] == player_id:
                c+=1
        
        return c

    def shortest_path(self, board: HexBoard, player_id: int):
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
        
        if not ds:
            return 0
        
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

    def next_moves(self, board: HexBoard):
        possible_mov = board.get_possible_moves()
        center = board.size // 2

        # Ordenamos por distancia al centro
        possible_mov.sort(key=lambda m: (m[0] - center)**2 + (m[1] - center)**2)
            
        if len(possible_mov) > 7:        
            for _ in range(int(math.sqrt(len(possible_mov)) + board.size)):            
                yield possible_mov.pop(int(np.random.exponential(2)) % len(possible_mov))
            
        else:
            for i in possible_mov:
                yield i

    def calculate_depth(self, board: HexBoard):
        possible_moves_lenght = len(board.get_possible_moves())
        if possible_moves_lenght > 50:
            return 1
        if possible_moves_lenght > 20:
            return 2
        if possible_moves_lenght > 10:
            return 3
        if possible_moves_lenght > 5:
            return 4
        return 7
