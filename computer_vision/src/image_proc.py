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
          blockSize: list -- tamanho da mascara do elemento estruturante
        Return:
          image_binary: np.array -- imagem binarizada
    """
    #scale_percent = 80 # percent of original size
    #width = int(image.shape[1] * scale_percent / 100)
    #height = int(image.shape[0] * scale_percent / 100)

    #blockSize=(width, height)
    blockSize=(80,80)

    structuring_element= cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, blockSize
    )
    image_treated = cv2.morphologyEx(
        image, cv2.MORPH_OPEN, structuring_element
    )
    image_treated = cv2.subtract(image, image_treated)

    image_treated = cv2.add(image_treated, image_treated)

    tipo = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    limiar, image_binary = cv2.threshold(image_treated, 0, 255, tipo)

    return image_binary


def watershed_proc(img2):
    img = img2
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)  # escala de cinza
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # pegando threshold
    kernel = np.ones((3, 3), np.uint8)  # criando o kernel 3 x 3
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                              iterations=2)  # aplicando tecnicas de erosão e posteriomente dilatação
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  #
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    return markers


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


def plot_and_save(img, maskwater, name_img):
    fig, ax = plt.subplots(1, 2, figsize=(10, 15))
    fig.suptitle('Imagem original x mask ')
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Imagem original')
    ax[1].imshow(maskwater, cmap='gray')
    ax[1].set_title('mask')
    #plt.imshow()

    plt.savefig('data/' + name_img + '.jpg',
                bbox_inches='tight', pad_inches=0)
