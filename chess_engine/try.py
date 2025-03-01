import chess.pgn

pgn = open("puzzle_mini.pgn")

first_game = chess.pgn.read_game(pgn)
second_game = chess.pgn.read_game(pgn)

print(first_game)
print(second_game)

board = first_game.board()

for move in first_game.mainline_moves():
    board.push(move)

print(board)