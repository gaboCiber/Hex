from time import sleep
from other_players import NormalPlayer, RamdomPlayer, CrazyPlayer, MonteCarloPlayer, InTerapyPlayer, RehabPlayer
from player import ThePlayer
from board import HexBoard
# from HexPlayer import MyPlayer
# from player_carlos import AIPlayer

wins = {1: 0, 2: 0}
printable = False

for _ in range(100):
    
    P1 = RehabPlayer(1)
    P2 = ThePlayer(2)
    board = HexBoard(5)
    
    if printable:
        board.print()
    
    i = 0
    while True:    

        P = P1 if i % 2 == 0 else P2
        #sleep(5)
        if printable:
            print(f"\nJugador {P.player_id}")
        
        move = P.play(board, 1)
        board.place_piece(int(move[0]), int(move[1]), P.player_id)
        
        if printable:
            print(move)    
            board.print()
        
        if board.check_connection(P.player_id):
            #if printable:
            print(f"GANO {P.player_id}")
            wins[P.player_id] += 1
            break

        i+=1

print(wins)