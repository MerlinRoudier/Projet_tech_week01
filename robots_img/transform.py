from PIL import Image, ImageOps

img = Image.open("./10_robot.png").convert("L")


arr_img = []
colors = ["blue", "grey", "brown", "pink", "purple", "yellow", "cyan", "red", "green"]

for i in range(len(colors)):
    arr_img.append(ImageOps.colorize(img, black=colors[i], white="white"))


for i in range(len(colors)):
    arr_img[i].save(str(i)+"_robot.png")