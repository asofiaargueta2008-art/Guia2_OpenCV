import cv2
import matplotlib.pyplot as plt

image = cv2.imread("frutas.jpg")

if image is None:
    print("Error: no se encontró frutas.jpg")
else:
    pequena = cv2.resize(
        image,
        (0, 0),
        fx=0.1,
        fy=0.1
    )

    grande = cv2.resize(
        image,
        (600, 350)
    )

    interpolacion = cv2.resize(
        image,
        (780, 540),
        interpolation=cv2.INTER_LINEAR
    )

    titulos = [
        "Original",
        "10 % del tamaño",
        "Tamaño fijo",
        "Interpolación lineal"
    ]

    imagenes = [
        image,
        pequena,
        grande,
        interpolacion
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