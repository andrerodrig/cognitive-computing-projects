import cv2
import os
import numpy as np
import computer_vision.src.image_proc as func


def pipeline(path_imgs, path_save, segmentation=1):
    """
        Pipeline do projeto
        Arguments:
          path_imgs: str -- caminho para a pasta das imagens
          path_save: str -- caminho para salvar os resultados
          segmentation: int -- 1: Limiar de Otsu
                               2: Sobel
                               3: Canny

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

        if segmentation == 1:
            seg = func.open_otsu(gray)

        elif segmentation == 2:
            seg = func.subtract_sobel(gray)

        elif segmentation == 3:
            seg = func.subtract_canny(gray)

        func.plot_and_save(resized, seg, path_save, f.split('.')[0])

        l = func.largura_folha(seg)
        c = func.comprimento_folha(seg)
        area.append(func.area_folha(seg))
        largura.append(l)
        comprimento.append(c)

        lar_max.append(np.argmax(l))
        comp_max.append(np.argmax(l))

    func.dataframe_csv(name, area, lar_max, comp_max, 'data_valueMax.csv')
    func.dataframe_csv(name, area, largura, comprimento, 'dataframe_final.csv')


def main():
    path = 'images/'
    path_save = 'data/'
    pipeline(path, path_save)


if __name__ == "__main__":
    main()
