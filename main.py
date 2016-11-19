import cv2
from model import PCAModel
import numpy as np
import math


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 130, 255, cv2.THRESH_BINARY)
    return image_bin


def get_piece_color(image, label):
    B = 0
    W = 0
    color = 0
    for byte in image:
        if byte > 128:
            W = W + 1
        else:
            B = B + 1
    P = math.ceil(B/(B+W)*100)
    if P >= 25:
        color = -1
    elif P >= 6:
        color = 1
    else:
        color = 0
    if label == "queen" and P < 30:
        color = 1
    return color

pieces = ["rook", "knight", "bishop", "king", "queen", "pawn"]
colors = ['black', 'white']
data = []
board = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]


def main():
    img_core = load_image("./images/initial.png")
    img_gray = image_gray(img_core)
    img_bin = image_bin(img_gray)
    img_bin = cv2.resize(img_bin, (240, 240))
    cv2.imwrite('tmp.png', img_bin)
    for color in colors:
        for piece in pieces:
            path = './images/'+color+'/'+piece+'.png'
            label = piece
            component_core = cv2.imread(path)
            component_gs = image_gray(component_core)
            component_bin = image_bin(component_gs)
            component = cv2.resize(component_bin, (30, 30))
            vector = np.array(component).ravel()
            data.append({'label': label, 'sample': vector})

    EigenPieceChess = PCAModel()
    EigenPieceChess.train(data)
    for i in range(8):
        for j in range(8):
            vector = np.array(img_bin[i*30:(i+1)*30, j*30:(j+1)*30]).ravel()
            label, res = EigenPieceChess.classify(vector)
            board[i][j] = (pieces.index(label)+1)*get_piece_color(vector, label)
    for column in board:
        print(column)

if __name__ == "__main__":
    main()
