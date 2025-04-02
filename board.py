import numpy as np
import copy

class HexBoard:
    def __init__(self, size: int):
        self.size = size  # Tamaño N del tablero (NxN)
        self.board = [[0] * size for _ in range(size)]  # Matriz NxN (0=vacío, 1=Jugador1, 2=Jugador2)
        self.player_positions = {1: set(), 2: set()}  # Registro de fichas por jugador
        
        
    def clone(self):# -> HexBoard:
        """Devuelve una copia del tablero actual"""
        return copy.deepcopy(self)

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """Coloca una ficha si la casilla está vacía."""
        
        if self.board[row][col] == 0:
            self.board[row][col] = player_id
            self.player_positions[player_id].add((row, col))

    def get_possible_moves(self) -> list:
        """Devuelve todas las casillas vacías como tuplas (fila, columna)."""
        return [ (i,j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
    
    def check_connection(self, player_id: int) -> bool:
        """Verifica si el jugador ha conectado sus dos lados"""
        
        visited = np.zeros((self.size, self.size), dtype=bool)
        x = [0 , 0, -1, -1, 1, 1]
        y = [-1, 1, 0, 1, -1, 0]
        
        def valid(x,y):
            return x >= 0 and x < self.size and y >= 0 and y < self.size 
        
        def search(row, col):
            visited[row][col] = True
            
            if (player_id == 1 and row == self.size - 1) or (player_id == 2 and col == self.size - 1):
                return True
            
            for i in range(6):
                new_row = row + x[i]
                new_col = col + y[i]
                
                if valid(new_row, new_col) and not visited[new_row][new_col] and self.board[new_row][new_col] == player_id: 
                    if search(new_row, new_col):
                        return True
                    
            visited[row][col] = False
            return False
        
        for i in range(self.size):
            row = 0 if player_id==1 else i
            col = i if player_id==1 else 0
            if self.board[row][col] == player_id and search(row,col):
                return True
            
        return False
    
    def print(self):
        
        print()
        
        for i in self.board:
            print(i)

                       
# table = HexBoard(3)
# pid = 1
# table.place_piece(1,0,pid)
# table.place_piece(1,1,pid)
# table.place_piece(1,2,pid)
# table.place_piece(0,2,pid)
# table.place_piece(2,1,pid)

# for i in table.board:
#     print(i)

# print(table.check_connection(pid))