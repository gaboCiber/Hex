from collections import deque
from board import HexBoard
import math
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
import networkx as nx
from copy import deepcopy
import concurrent.futures
import time
from A_Star import A_star

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)
        self.time_limit = None
        
    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("¡Implementa este método!")

class ThePlayer(Player):
    def __init__(self, player_id: int):
        self.player_id = player_id  # Identificador del jugador (1 o 2)
    
    def chech_time(self):
        if self.time_limit and time.time() - self.start_time >= self.time_limit:
            raise concurrent.futures.TimeoutError()
    
    def play(self, board: HexBoard, time_limit: float = None) -> tuple:
        # Decide la jugada a realizar según el tiempo disponible
        if time_limit is None:
            # Si no hay límite de tiempo, simplemente aplica búsqueda alfa-beta
            _, move = self.max_alfabeta(board, self.player_id, self.calculate_depth(board))
            return move
        
        # Si hay un límite de tiempo, ejecuta la búsqueda en otro hilo y controla el tiempo
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.start_time = time.time()
            self.time_limit = time_limit
            future = executor.submit(self.max_alfabeta, board, self.player_id, self.calculate_depth(board))
            try:
                _, move = future.result(timeout=time_limit)
                return move
            except concurrent.futures.TimeoutError:
                print("out of time")
                return self.random_move  # Devuelve una jugada random si se acaba el tiempo

    def max_alfabeta(self, board, player_id, depth, alfa=-math.inf, beta=math.inf):
        self.chech_time()
        
        # Función de maximización con poda alfa-beta
        if depth == 0:
            return (self.evaluate_board(board, player_id), None)

        v = -math.inf
        move = None
        for a in self.next_moves(board):  # Genera las próximas jugadas ordenadas
            copy = board.clone()
            copy.place_piece(a[0], a[1], player_id)

            if copy.check_connection(player_id):
                # Si conecta el lado, gana automáticamente
                return (math.inf if self.player_id == player_id else -math.inf, a)

            v2, _ = self.min_alfabeta(copy, 3 - player_id, depth - 1, alfa, beta)

            if v2 > v:
                v, move = v2, a
                alfa = max(alfa, v)

            if not move:
                move = a  # Por si acaso no se encuentra mejor jugada

            if v >= beta:
                return (v, move)  # Poda

        return v, move

    def min_alfabeta(self, board, player_id, depth, alfa=-math.inf, beta=math.inf):
        
        self.chech_time()
        
        # Función de minimización con poda alfa-beta
        if depth == 0:
            return (self.evaluate_board(board, player_id), None)

        v = math.inf
        move = None
        for a in self.next_moves(board):
            copy = board.clone()
            copy.place_piece(a[0], a[1], player_id)

            if copy.check_connection(player_id):
                return (math.inf if self.player_id == player_id else -math.inf, a)

            v2, _ = self.max_alfabeta(copy, 3 - player_id, depth - 1, alfa, beta)

            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)

            if not move:
                move = a

            if v <= alfa:
                return (v, move)  # Poda

        return v, move

    def evaluate_board(self, board: HexBoard, player_id: int) -> int:
        
        self.chech_time()
        
        # Función de evaluación del tablero combinando varios factores heurísticos
        return (
            3 * self.shortest_path(board, player_id) +
            10 * self.center_plays(board, player_id) +
            3 * self.break_rival_bridges(board, player_id) +
            6 * self.rival_influence(board, player_id)
        )

    def valid(self, board, x, y):
        self.chech_time()
        
        # Verifica si una coordenada está dentro de los límites del tablero
        return 0 <= x < board.size and 0 <= y < board.size

    def minDist(self, A, B):
        # Calcula la distancia euclidiana mínima entre dos conjuntos de coordenadas
        m = float("inf")
        for coor_a in A:
            for coor_b in B:
                m = min(m, math.dist(coor_a, coor_b))
        return m

    def rival_influence(self, board, player_id):
        # Penaliza casillas vacías muy influenciadas por el rival
        c = 0
        x = [0, 0, -1, -1, 1, 1]
        y = [-1, 1, 0, 1, -1, 0]

        for i in range(1, board.size - 1):
            for j in range(1, board.size - 1):
                if board.board[i][j] == 0:
                    for k in range(6):
                        new_x = i + x[k]
                        new_y = j + y[k]
                        if self.valid(board, new_x, new_y) and board.board[new_x][new_y] == 3 - player_id:
                            c += 1
        return -c  # Negativo porque es malo que el rival tenga influencia

    def break_rival_bridges(self, board, player_id):
        # Cuenta cuántos "puentes" del rival podrían ser interrumpidos por la jugada
        x = [0, 0, -1, -1, 1, 1]
        y = [-1, 1, 0, 1, -1, 0]
        broken_bridges = 0

        for i in range(1, board.size - 1):
            for j in range(1, board.size - 1):
                if board.board[i][j] == player_id:
                    c = 0
                    for k in range(6):
                        new_x = i + x[k]
                        new_y = j + y[k]
                        if self.valid(board, new_x, new_y) and board.board[new_x][new_y] == 3 - player_id:
                            c += 1
                    broken_bridges += math.comb(c, 2)
        return broken_bridges

    def center_plays(self, board, player_id):
        # Da valor a jugar cerca del centro
        c = 0
        middle = int(board.size / 2) - 1
        for i in range(board.size):
            if board.size > 2 and board.board[i][middle] == player_id:
                c += 1
            if board.size > 2 and board.board[middle][i] == player_id:
                c += 1
            if board.size > 3 and board.board[i][middle + 1] == player_id:
                c += 1
            if board.size > 3 and board.board[middle + 1][i] == player_id:
                c += 1
        return c

    def shortest_path(self, board, player_id):
        # Calcula una estimación del camino más corto del jugador
        ds = DisjointSet()

        # Conectamos piezas del jugador entre sí (adyacentes)
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == player_id:
                    ds.add((i, j))
                    right = (i, j + 1)
                    down_right = (i + 1, j)

                    if self.valid(board, right[0], right[1]) and board.board[right[0]][right[1]] == player_id:
                        ds.add(right)
                        ds.merge((i, j), right)
                    if self.valid(board, down_right[0], down_right[1]) and board.board[down_right[0]][down_right[1]] == player_id:
                        ds.add(down_right)
                        ds.merge((i, j), down_right)

        if not ds:
            return 0

        # Creamos grafo con componentes conectadas para estimar el camino
        G = nx.Graph()
        G.add_node(-1)  # Inicio virtual
        G.add_node(board.size)  # Fin virtual

        length = len(ds.subsets())
        for i in range(length):
            for j in range(i + 1, length):
                m = self.minDist(ds.subsets()[i], ds.subsets()[j])
                G.add_edge(i, j, weight=m)

            # Conecta componente con borde inicial/final
            index = 1 if player_id == 1 else 0
            S = [t[index] for t in ds.subsets()[i]]
            G.add_edge(-1, i, weight=min(S))
            G.add_edge(board.size, i, weight=(board.size - max(S) - 1))

        
        
        return -nx.shortest_path_length(G, -1, board.size, "weight")

    def next_moves(self, board):
        self.chech_time()
        
        # Genera los movimientos posibles ordenados por cercanía al centro
        possible_mov = board.get_possible_moves()
        center = board.size // 2
        possible_mov.sort(key=lambda m: (m[0] - center)**2 + (m[1] - center)**2)

        # Prioriza jugadas más cercanas al centro de manera estocástica
        if len(possible_mov) > 7:
            for _ in range(int(math.sqrt(len(possible_mov)) + board.size)):
                yield possible_mov.pop(int(np.random.exponential(2)) % len(possible_mov))
        else:
            for i in possible_mov:
                yield i

    def calculate_depth(self, board):
        # Calcula profundidad de búsqueda en función de cuántas jugadas posibles hay
        possible_moves = board.get_possible_moves()
        self.random_move = possible_moves[np.random.randint(0, high=len(possible_moves))]

        possible_moves_lenght = len(possible_moves)
        if possible_moves_lenght > 50:
            return 1
        if possible_moves_lenght > 20:
            return 2
        if possible_moves_lenght > 10:
            return 3
        if possible_moves_lenght > 5:
            return 4
        return 7
