import cv2
from skimage import data

# Da biblioteca
img_coffee = data.coffee()
img_astro = data.astronaut()
img_rocket = data.rocket()
img_text = data.text()

# Do diretorio
img_valve = cv2.imread('./images/valve.jpg')
img_spider = cv2.imread('./images/spider.jpg')
img_lena = cv2.imread('./images/lena.jpg')
img_woman = cv2.imread('./images/woman.jpg') # Ja em grayscale
img_cow = cv2.imread('./images/cow.jpg')
img_coin = cv2.imread('./images/coin.jpg')