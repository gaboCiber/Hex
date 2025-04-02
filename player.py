from board import HexBoard
import math

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    def play(self, board: HexBoard) -> tuple:
        
        rival = Player(player_id=3-self.player_id)
        
        def evaluate_board(board: HexBoard, P: Player) -> int:
            return sum(sum(j == P.player_id for j in i) for i in board.board)
        
        def minmax(board: HexBoard, P: Player, is_max: bool, depth: int = 5):
            
            value = -1 if is_max else board.size + 1
            move = None

            for i in board.get_possible_moves():
                
                copy = board.clone()
                copy.place_piece(i[0], i[1], P.player_id)
                
                if copy.check_connection(P.player_id) or depth == 0:
                    return (evaluate_board(copy, P), i)
                
                comp, move = minmax(copy, rival, False, depth + 1) if is_max else minmax(copy, self, True, depth + 1)
                
                fun = max if is_max else min                
                new = fun(value, comp)
                
                if value != new or not move:
                    move = i 

                value = new
                
            return (value, move)
                    
        def minmax_alfabeta(board: HexBoard, P: Player, is_max: bool, alfa: int = -math.inf, beta: int = math.inf, depth: int = 5):
            
            move = None                

            for i in board.get_possible_moves():
                
                copy = board.clone()
                copy.place_piece(i[0], i[1], P.player_id)
                
                if copy.check_connection(P.player_id) or depth == 0:
                    return (evaluate_board(copy, P), i)
                
                old_value = (alfa, beta)
                
                if is_max:
                    value, move = minmax_alfabeta(copy, rival, False, depth - 1, alfa, beta) 
                    alfa = max(alfa, value) 
                else:
                    value, move = minmax_alfabeta(copy, self, True, depth - 1, alfa, beta)
                    beta = min(beta, value)
                                 
                if beta <= alfa:
                    break
                
                if (alfa, beta) != old_value or not move:
                    move = i 
                
            return (alfa if is_max else beta, move)
          
        _ , move = minmax_alfabeta(board, self, True)
        return move     
        
        
P1 = Player(1)
P2 = Player(2)
board = HexBoard(2)
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


        
    
    
    