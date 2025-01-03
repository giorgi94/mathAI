import os
import re
import sys
import pickle
import numpy as np

from PIL import Image, ImageOps, ImageDraw, ImageFont


def assure_path_exists(path, isfile=True):
    if isfile:
        dir_path = os.path.dirname(path)
    else:
        dir_path = path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def draw_text(text, size, font, savepath):
    if type(text) != str:
        text = str(text)
    img = Image.new("RGB", size, "white")
    fnt = ImageFont.truetype(font, 33)
    draw = ImageDraw.Draw(img)
    W, H = size
    w, h = draw.textsize(text, font=fnt)
    coords = ((W - w) / 2, (H - h) / 2 - 3)
    draw.text(coords, text, font=fnt, fill="black")
    img.save(savepath)


def draw_numbers(col, font):
    to = f"data/imgs/{col}"
    to = os.path.abspath(to)
    size = (28, 28)
    assure_path_exists(to, isfile=False)

    for i in range(10):
        draw_text(i, size, font, savepath=f"{to}/{i}.jpg")


def draw_training_numbers():

    command = 'find /usr/share/fonts -type f -name "*.ttf"'
    fontfiles = [i.strip() for i in os.popen(command).read().split("\n")]

    index = 0

    for font in fontfiles[:10]:
        try:
            draw_numbers(index, font)
            index += 1
        except:
            pass


def name2vec(num, size=10):
    y = np.zeros((size, 1))
    y[num][0] = 1
    return y


def img2vec(path):
    with Image.open(path) as img:
        img = ImageOps.grayscale(img)
        size = np.prod(img.size)
        return np.array(img.getdata(), np.uint8).reshape(size, 1) / 255


def write_training_numbers():
    read_from = os.path.abspath("data/imgs")
    write_to = os.path.abspath("data/imgvec.pkl")

    assure_path_exists(read_from, False)

    matrices = []

    names = os.listdir(read_from)

    names.sort(key=lambda n: (n, int(n)))

    for name in names:
        for i in range(10):
            A = img2vec(os.path.join(read_from, name, f"{i}.jpg"))
            number = int(re.search(r"\d+", name).group())
            matrices.append((A.reshape((A.size, 1)), name2vec(number)))

    with open(write_to, "wb") as f:
        pickle.dump(matrices, f)


if __name__ == "__main__":

    # draw_training_numbers()
    write_training_numbers()
