from board import ChessBoard


def main():
    board = ChessBoard('./images/tests/2.png')
    for column in board.get_board():
        print(column)

if __name__ == "__main__":
    main()
