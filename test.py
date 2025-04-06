from player import NormalPlayer, RamdomPlayer, CrazyPlayer
from board import HexBoard

wins = {1: 0, 2: 0}

for _ in range(10):
    
    P1 = CrazyPlayer(1, divisor=1)
    P2 = CrazyPlayer(2, divisor=10)
    board = HexBoard(6)
    board.print()
    
    i = 0
    while True:
        
        P = P1 if i % 2 == 0 else P2
        
        print(f"\nJugador {P.player_id}")
        move = P.play(board)
        print(move)
        board.place_piece(int(move[0]), int(move[1]), P.player_id)
        board.print()
        
        if board.check_connection(P.player_id):
            print(f"GANO {P.player_id}")
            wins[P.player_id] += 1
            break

        i+=1

print(wins)