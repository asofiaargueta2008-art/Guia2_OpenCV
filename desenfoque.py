import cv2
import matplotlib.pyplot as plt

# Cargar imagen
image = cv2.imread("frutas.jpg")

if image is None:
    print("Error: no se encontró frutas.jpg")
else:
    # Reducir un poco la imagen para visualizarla mejor
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Desenfoque gaussiano
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Desenfoque mediano
    median_blur = cv2.medianBlur(image, 5)

    # Desenfoque bilateral
    bilateral_blur = cv2.bilateralFilter(image, 9, 75, 75)

    titulos = [
        "Imagen original",
        "Desenfoque gaussiano",
        "Desenfoque mediano",
        "Desenfoque bilateral"
    ]

    imagenes = [
        image,
        gaussian_blur,
        median_blur,
        bilateral_blur
    ]

    plt.figure(figsize=(12, 8))

    for i in range(4):
        plt.subplot(2, 2, i + 1)

        imagen_rgb = cv2.cvtColor(
            imagenes[i],
            cv2.COLOR_BGR2RGB
        )

        plt.imshow(imagen_rgb)
        plt.title(titulos[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()