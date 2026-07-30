import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread("universo.jpg")

if imagen is None:
    print("Error: no se encontró universo.jpg")

else:
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    gris_suave = cv2.GaussianBlur(gris, (5, 5), 0)

    # Bordess Canny

    bordes_canny = cv2.Canny(
        gris_suave,
        80,
        180
    )

    # Sobel

    sobel_x = cv2.Sobel(
        gris_suave,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gris_suave,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitud_sobel = cv2.magnitude(
        sobel_x.astype(np.float32),
        sobel_y.astype(np.float32)
    )

    magnitud_sobel = cv2.convertScaleAbs(
        magnitud_sobel
    )

    # laplaciano

    laplaciano = cv2.Laplacian(
        gris_suave,
        cv2.CV_64F
    )

    laplaciano = cv2.convertScaleAbs(
        laplaciano
    )

    # hough

    imagen_hough = imagen_rgb.copy()

    lineas = cv2.HoughLinesP(
        bordes_canny,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=40,
        maxLineGap=15
    )

    if lineas is not None:
        for linea in lineas:
            x1, y1, x2, y2 = linea

            cv2.line(
                imagen_hough,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

    # Resultado

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(imagen_rgb)
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(bordes_canny, cmap="gray")
    plt.title("Canny")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(magnitud_sobel, cmap="gray")
    plt.title("Sobel")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(laplaciano, cmap="gray")
    plt.title("Laplaciano")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(imagen_hough)
    plt.title("Transformada de Hough")
    plt.axis("off")

    plt.tight_layout()
    plt.show()