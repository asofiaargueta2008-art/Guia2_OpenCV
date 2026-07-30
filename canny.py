import cv2
import matplotlib.pyplot as plt

imagen = cv2.imread("frutas.jpg")

if imagen is None:
    print("Error: no se encontró frutas.jpg")
else:
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    bordes = cv2.Canny(
        imagen_gris,
        threshold1=80,
        threshold2=255
    )

    figura, ejes = plt.subplots(1, 2, figsize=(12, 5))

    ejes[0].imshow(imagen_rgb)
    ejes[0].set_title("Imagen original")
    ejes[0].axis("off")

    ejes[1].imshow(bordes, cmap="gray")
    ejes[1].set_title("Bordes Canny")
    ejes[1].axis("off")

    plt.tight_layout()
    plt.show()