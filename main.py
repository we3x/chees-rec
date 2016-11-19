from board import ChessBoard


def main():
    board = ChessBoard('./images/initial.png')

    for column in board.get_board():
        print(column)

if __name__ == "__main__":
    main()
