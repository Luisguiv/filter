import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

def apply_threshold_filter(image, threshold_value):
    # Lê a imagem em escala de cinza
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica limiarização binária invertida
    _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    return thresh

def apply_greyscale_filter(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_high_pass_filter_basic(img):
    return img - cv2.GaussianBlur(img, (21, 21), 3)+127

def apply_high_pass_filter_boost(img, boost_factor):
    hipass = img - cv2.GaussianBlur(img, (7, 7), 3)+127
    hiboost = cv2.addWeighted(img, boost_factor, hipass, -1, 0)

    return hiboost

def apply_low_pass_filter_mean(img, kernel):
    kernel = int(kernel)
    return cv2.boxFilter(img, -1, (kernel,kernel))

def apply_low_pass_filter_median(img, kernel):
    kernel = int(kernel)
    return cv2.medianBlur(img,kernel)

def apply_roberts_filter(image):
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

def apply_log_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar suavização Gaussiana
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Aplicar o Laplaciano da Gaussiana
    log_image = cv2.Laplacian(blurred, cv2.CV_64F)
    
    return log_image

def find_zero_crossings(log_image):
    # Encontrar os pontos de cruzamento zero
    rows, cols = log_image.shape
    zero_crossings = np.zeros_like(log_image, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [
                log_image[i - 1, j],
                log_image[i + 1, j],
                log_image[i, j - 1],
                log_image[i, j + 1]
            ]
            center_value = log_image[i, j]

            # Verificar se há cruzamento zero
            if all(value < 0 for value in neighbors) or all(value > 0 for value in neighbors):
                zero_crossings[i, j] = 255

    return zero_crossings

def apply_canny_filter(image, low_threshold, high_threshold): # l=50 h=150
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica o filtro Canny
    edges = cv2.Canny(image, low_threshold, high_threshold)

    return edges

def add_salt_and_pepper_noise(image, amount):
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

def plot_histogram(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calcula o histograma da imagem
    hist = cv2.calcHist([image], [0], None, [256], [0,256])

    # Plota o histograma em barras
    plt.bar(range(256), hist.ravel(), width=1, color='magenta')
    plt.title('Histograma da Imagem(Grayscale)')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.show()

def equalize_histogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Equalizar o histograma
    equalized_img = cv2.equalizeHist(img)
    
    return equalized_img

def count_objects(image):
    gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_scaled, 225,225, cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    output = image.copy()
    for contour in contours:
        cv2.drawContours(output, [contour], -1,(240,0,159),3)

    text = "Tem {} objetos na imagem".format(len(contours))
    cv2.putText(output, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(240, 0, 159), 2)

    return output

def adjust_gamma(image, gamma): # gamma = 3.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)