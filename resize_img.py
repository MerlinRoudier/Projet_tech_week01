# Redimensionner l'image à 80x80 pixels.
from PIL import Image

image = Image.open('robot.jpg')
n_image = image.resize((80,80))
n_image.save('robot.jpg')