import cv2
import numpy as np


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 130, 255, cv2.THRESH_BINARY)
    return image_bin


def main():
    img_core = load_image("./images/initial.png")
    img_gray = image_gray(img_core)
    img_bin = image_bin(img_gray)
    template = cv2.imread('./images/white/wp.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_bin, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.25
    loc = np.where(res >= threshold)
    print(loc)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_bin, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite('nn.png', img_bin)


if __name__ == "__main__":
    main()
