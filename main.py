import cv2
import numpy as np

def apply_threshold_filter(image, threshold_value=64):
    # Lê a imagem em escala de cinza
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica limiarização binária invertida
    _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    return thresh

def apply_greyscale_filter(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_high_pass_filter(img):
    # Converte a imagem para escala de cinza (se não estiver em escala de cinza)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplica o filtro Laplaciano
    laplacian_filter = cv2.Laplacian(img, cv2.CV_64F)
    
    # Normaliza o resultado para valores no intervalo de 0 a 255
    laplacian_filter = cv2.normalize(laplacian_filter, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return laplacian_filter

def apply_low_pass_filter(img, ksize=(5, 5)):
    # Aplica o filtro Gaussiano
    return cv2.GaussianBlur(img, ksize, 0)

def apply_roberts_filter(image):
    # Lê a imagem em escala de cinza
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define os kernels do operador de Roberts
    roberts_kernel_x = np.array([[1, 0], [0, -1]])
    roberts_kernel_y = np.array([[0, 1], [-1, 0]])

    # Aplica os kernels na imagem
    roberts_x = cv2.filter2D(image, -1, roberts_kernel_x)
    roberts_y = cv2.filter2D(image, -1, roberts_kernel_y)

    # Combina os resultados para obter a magnitude do gradiente
    roberts_magnitude = np.hypot(roberts_x, roberts_y)

    # Normaliza para um intervalo de 0 a 255
    roberts_normalized = (roberts_magnitude / roberts_magnitude.max() * 255).astype(np.uint8)

    return roberts_normalized

def apply_prewitt_filter(image):
    # Lê a imagem em escala de cinza
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define os kernels do operador de Prewitt
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Aplica os kernels na imagem
    prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x)
    prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y)

    # Combina os resultados para obter a magnitude do gradiente
    prewitt_magnitude = np.hypot(prewitt_x, prewitt_y)

    # Normaliza para um intervalo de 0 a 255
    prewitt_normalized = (prewitt_magnitude / prewitt_magnitude.max() * 255).astype(np.uint8)

    return prewitt_normalized

def apply_sobel_filter(image):
    # Lê a imagem
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplica o filtro Sobel na direção x
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    # Aplica o filtro Sobel na direção y
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Combina os dois gradientes
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Normaliza para um intervalo de 0 a 255
    sobel_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return sobel_normalized

def apply_log_filter(image, ksize=3, sigma=0.5):
    # Lê a imagem em escala de cinza
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica o filtro Gaussiano para suavização
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    # Aplica o filtro Laplaciano para detecção de bordas
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Normaliza o resultado para um intervalo de 0 a 255
    laplacian_normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return laplacian_normalized

def zero_crossing(laplacian_img):
    output = np.zeros(laplacian_img.shape, dtype=np.uint8)
    rows, cols = laplacian_img.shape

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            patch = laplacian_img[y-1:y+2, x-1:x+2]
            min_val = patch.min()
            max_val = patch.max()
            if min_val < 0 and max_val > 0:
                output[y, x] = 255

    return output

def apply_zero_cross_filter(image, ksize=3, sigma=0.5):
    # Lê a imagem em escala de cinza
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica o filtro Gaussiano para suavização
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    # Aplica o filtro Laplaciano
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Aplica o método ZeroCross
    zerocross_img = zero_crossing(laplacian)

    return zerocross_img

def apply_canny_filter(image, low_threshold, high_threshold):
    # Lê a imagem em escala de cinza
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica o filtro Canny
    edges = cv2.Canny(image, low_threshold, high_threshold)

    return edges

def add_salt_and_pepper_noise(image, amount=0.2):
    # Lê a imagem
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Cria uma máscara de ruído "salt and pepper"
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    image[coords[0], coords[1]] = 255

    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    image[coords[0], coords[1]] = 0

    return image

def apply_watershed_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detecção de bordas usando o Canny
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remoção de pequenos ruídos brancos usando morfologia
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Área de fundo com certeza
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Distância transformada para encontrar a área de primeiro plano com certeza
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Rotulando os marcadores
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Aplica o algoritmo Watershed
    cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]

    return img

img_valve = cv2.imread('valve.png')
img_spider = cv2.imread('spider.jpeg')
img_lena = cv2.imread('lena.png')
img_woman = cv2.imread('woman.jpg') # Ja em grayscale
img_cow = cv2.imread('cow.png')
img_coin = cv2.imread('coin.jpg')

img_threshold = apply_threshold_filter(img_spider)
img_greyscale = apply_greyscale_filter(img_lena)
img_high_pass = apply_high_pass_filter(img_lena)
img_low_pass = apply_low_pass_filter(img_lena)
img_roberts = apply_roberts_filter(img_woman)
img_prewitt = apply_roberts_filter(img_woman)
img_sobel = apply_sobel_filter(img_valve)
img_log = apply_log_filter(img_cow)
img_zero = apply_zero_cross_filter(img_cow)
img_canny = apply_canny_filter(img_lena, 50, 150)
img_noise = add_salt_and_pepper_noise(img_lena)
img_watershed = apply_watershed_filter(img_coin)

cv2.imshow('Imagem Filtrada(Threshold)',img_threshold)
cv2.imshow('Imagem Filtrada(Greyscale)',img_greyscale)
cv2.imshow('Imagem Filtrada(High_Pass)',img_high_pass)
cv2.imshow('Imagem Filtrada(Low_Pass)',img_low_pass)
cv2.imshow('Imagem Filtrada(Roberts)',img_roberts)
cv2.imshow('Imagem Filtrada(Prewitt)',img_prewitt)
cv2.imshow('Imagem Filtrada(Sobel)',img_sobel)
cv2.imshow('Imagem Filtrada(LoG)',img_log)
cv2.imshow('Imagem Filtrada(Zerocross)',img_zero)
cv2.imshow('Imagem Filtrada(Canny)',img_canny)
cv2.imshow('Imagem com ruido(Salt and Pepper))',img_noise)
cv2.imshow('Imagem Filtrada(Watershed)',img_watershed)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image