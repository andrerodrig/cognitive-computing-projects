import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def subtract_sobel(image_gray):
    """
        Funcao responsavel por realçar uma imagem utilizando a operação de subtração com o filtro de sobel
       Arguments:
          image_gray: np.array -- Uma matriz NumPy que representa a imagem na escala de cinza
                com a forma (num_rows, num_cols, num_channels)
        Return:
          image_enhancement: np.array -- Imagem realçada
    """
    sobelx = cv2.Sobel(image_gray, cv2.CV_8U, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_gray, cv2.CV_8U, 0, 1, ksize=3)
    sobel = cv2.bitwise_or(sobelx, sobely)
    image_enhancement = cv2.subtract(image_gray, sobel)
    return image_enhancement


def open_otsu(image):
    """
        Funcao responsavel por binarizar uma imagem utilizando a limiarização de otsu.
        Arguments:
          image: np.array -- Uma matriz NumPy que representa a imagem na escala de cinza
                pre-processada com a forma (num_rows, num_cols, num_channels)
        Return:
          image_binary: np.array -- imagem binarizada
    """
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tipo = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    limiar, image_binary = cv2.threshold(image, 0, 255, tipo)

    return image_binary


def subtract_canny(image_gray):
    """
        Funcao responsavel por realçar uma imagem utilizando a operação de subtração com o filtro canny
        Arguments:
          image_gray: np.array -- Uma matriz NumPy que representa a imagem na escala de cinza
                com a forma (num_rows, num_cols, num_channels)
        Return:
          image_enhancement: np.array -- Imagem realçada
    """
    canny = cv2.Canny(image_gray, 100, 200)
    image_enhancement = cv2.subtract(image_gray, canny)
    return image_enhancement


def IoU_image_seg(img_gray, ground_truth, img_seg):
    """
              Funcao responsavel pela métrica IoU
              Arguments:
                img_gray: np.array -- Uma matriz NumPy que representa a imagem na escala de cinza
                  com a forma (num_rows, num_cols, num_channels)
                ground_truth: np.array -- Uma matriz NumPy que representa a imagem na escala de cinza
                  com a forma (num_rows, num_cols, num_channels)
                img_seg:  np.array -- Uma matriz NumPy que representa a imagem binarizada
              Return:
                  iou_score: float

    """
    result1 = cv2.bitwise_and(img_gray, ground_truth)
    result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)

    result2 = cv2.bitwise_and(img_gray, img_seg)
    result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)

    intersection = np.logical_and(result1, result2)
    union = np.logical_or(result1, result2)
    iou_score = np.sum(intersection) / np.sum(union)
    print('IoU is %s' % iou_score)

    return iou_score


def area_folha(image_binary):
    # print(image_binary.shape)
    lin, col = image_binary.shape
    area = 0
    for linha in range(lin):
        for coluna in range(col):
            if image_binary[linha, coluna] != 0:
                area += 1

    return area


def largura_folha(image_binary):
    # print(image_binary.shape)
    lin, col = image_binary.shape
    larguras = []
    for linha in range(lin):
        largura = 0
        for coluna in range(col):
            if image_binary[linha, coluna] != 0: largura += 1
        larguras.append(largura)

    #print("A folha possui as larguras:")
    #print(larguras)

    return larguras


def comprimento_folha(image_binary):
    # print(image_binary.shape)
    lin, col = image_binary.shape
    comprimentos = []
    for coluna in range(col):
        comprimento = 0
        for linha in range(lin):
            if image_binary[linha, coluna] != 0: comprimento += 1
        comprimentos.append(comprimento)

    #print("A folha possui as comprimentos:")
    #print(comprimentos)

    return comprimentos


def dataframe_csv(name, area, larguras, comprimentos, path):
    #print(len(name))
    #print(len(area))
    #print(len(larguras))
    dados_folha = {
        "name": name,
        "area": area,
        "largura": larguras,
        "comprimento": comprimentos
    }

    df = pd.DataFrame(dados_folha)
    df.to_csv(path)


def plot_and_save(img, maskwater, path_save,name_img):
    fig, ax = plt.subplots(1, 2, figsize=(10, 15))
    fig.suptitle('Imagem original x Segmentation ')
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Imagem original')
    ax[1].imshow(maskwater, cmap='gray')
    ax[1].set_title('Segmentation')
    #plt.imshow()

    plt.savefig(path_save + name_img + '.jpg',
                bbox_inches='tight', pad_inches=0)
