import cv2
from model import PCAModel
import numpy as np


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 130, 255, cv2.THRESH_BINARY)
    return image_bin

pieces = ["king", "pawn", "knight", "bishop", "queen", "rook"]
colors = ['black', 'white']
data = []
board = [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]
]


def main():
    img_core = load_image("./images/initial.png")
    img_gray = image_gray(img_core)
    img_bin = image_bin(img_gray)
    img_bin = cv2.resize(img_bin, (240, 240))
    for color in colors:
        for piece in pieces:
            path = './images/'+color+'/'+piece+'.png'
            label = piece
            component_core = cv2.imread(path)
            component_gs = image_gray(component_core)
            component_bin = image_bin(component_gs)
            component = cv2.resize(component_bin, (30, 30))
            vector = np.array(component).ravel()
            data.append({'label':label, 'sample':vector})

    EigenPieceChess = PCAModel()
    EigenPieceChess.train(data)
    for i in range(8):
        for j in range(8):
            vector = np.array(img_bin[i*30:(i+1)*30, j*30:(j+1)*30]).ravel()
            label, res = EigenPieceChess.classify(vector)
            board[i][j] = pieces.index(label)
            if label == "pawn":
                if res > 1000:
                    board[i][j] = 0

    for i in board:
        print(i)

if __name__ == "__main__":
    main()
