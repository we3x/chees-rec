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

pieces = ['king', 'pawn', 'knight', 'bishop', 'queen', 'rook']
colors = ['black', 'white']
data = []

def main():
    img_core = load_image("./images/initial.png")
    img_gray = image_gray(img_core)
    img_bin = image_bin(img_gray)
    img_bin = cv2.resize(img_bin, (240, 240))
    for color in colors:
        for piece in pieces:
            path = './images/'+color+'/'+piece+'.png'
            label = color[0] + piece
            component = cv2.imread(path)
            component = cv2.resize(component, (30, 30))
            vector = np.array(component).ravel()
            data.append({'label':label, 'sample':vector})

    EigenPieceChess = PCAModel()
    EigenPieceChess.train(data)

if __name__ == "__main__":
    main()
