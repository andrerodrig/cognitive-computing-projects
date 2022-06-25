import cv2
import os
import numpy as np
from computer_vision.src.image_proc import area_folha,\
    comprimento_folha,largura_folha,subtract_sobel,open_otsu,\
    watershed_proc,dataframe_csv, IoU_image_seg, plot_and_save


def pipeline(path_imgs):
    """
        Pipeline do projeto
        Arguments:
          path_imgs: str -- caminho para a pasta das imagens

        Return:
    """

    files_ = os.listdir(path_imgs)
    #width = 960
    #height = 1280
    scale_percent = 30 # percent of original size
    width = int(960 * scale_percent / 100)
    height = int(1280 * scale_percent / 100)

    dim = (width, height)

    area = []
    lar_max = []
    comp_max = []
    largura = []
    comprimento = []
    name = []

    for f in files_:
        name.append(f)
        print(f'[INFO] name: {f}')
        img = cv2.imread(path_imgs + f)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

        sob = subtract_sobel(gray)
        ot = open_otsu(gray)
        wat = watershed_proc(resized)

        plot_and_save(resized, ot, f.split('.')[0])

        l = largura_folha(ot)
        c = comprimento_folha(ot)
        area.append(area_folha(ot))
        largura.append(l)
        comprimento.append(c)

        lar_max.append(np.argmax(l))
        comp_max.append(np.argmax(l))

    dataframe_csv(name, area, lar_max, comp_max, 'data_valueMax.csv')
    dataframe_csv(name, area, largura, comprimento, 'dataframe_final.csv')


def main():
    path = 'images/'
    pipeline(path)


if __name__ == "__main__":
    main()
