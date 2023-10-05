import cv2
from skimage import data

# Da biblioteca
img_coffee = data.coffee()
img_astro = data.astronaut()
img_rocket = data.rocket()
img_text = data.text()

# Do diretorio
img_valve = cv2.imread('./images/valve.png')
img_spider = cv2.imread('./images/spider.jpeg')
img_lena = cv2.imread('./images/lena.png')
img_woman = cv2.imread('./images/woman.jpg') # Ja em grayscale
img_cow = cv2.imread('./images/cow.png')
img_coin = cv2.imread('./images/coin.jpg')