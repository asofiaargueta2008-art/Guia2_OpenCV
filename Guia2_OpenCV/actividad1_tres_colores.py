import cv2
import matplotlib.pyplot as plt

imagen_bgr = cv2.imread("huerto.jpg")

if imagen_bgr is None:
    print("Error: no se encontró huerto.jpg")
    print("Verifica que esté en la misma carpeta que este archivo.")

else:
    print("Imagen cargada correctamente.")
    print("Dimensiones:", imagen_bgr.shape)

    imagen_rgb = cv2.cvtColor(
        imagen_bgr,
        cv2.COLOR_BGR2RGB
    )


    imagen_hsv = cv2.cvtColor(
        imagen_bgr,
        cv2.COLOR_BGR2HSV
    )

    # Color Verde

    verde_bajo = (35, 40, 40)
    verde_alto = (85, 255, 255)

    mascara_verde = cv2.inRange(
        imagen_hsv,
        verde_bajo,
        verde_alto
    )

    resultado_verde = cv2.bitwise_and(
        imagen_bgr,
        imagen_bgr,
        mask=mascara_verde
    )

    resultado_verde_rgb = cv2.cvtColor(
        resultado_verde,
        cv2.COLOR_BGR2RGB
    )

    # Color Rojo

    rojo_bajo_1 = (0, 70, 50)
    rojo_alto_1 = (10, 255, 255)

    rojo_bajo_2 = (170, 70, 50)
    rojo_alto_2 = (179, 255, 255)

    mascara_rojo_1 = cv2.inRange(
        imagen_hsv,
        rojo_bajo_1,
        rojo_alto_1
    )

    mascara_rojo_2 = cv2.inRange(
        imagen_hsv,
        rojo_bajo_2,
        rojo_alto_2
    )

    mascara_rojo = cv2.bitwise_or(
        mascara_rojo_1,
        mascara_rojo_2
    )

    resultado_rojo = cv2.bitwise_and(
        imagen_bgr,
        imagen_bgr,
        mask=mascara_rojo
    )

    resultado_rojo_rgb = cv2.cvtColor(
        resultado_rojo,
        cv2.COLOR_BGR2RGB
    )

    # Color Amarillo

    amarillo_bajo = (20, 70, 70)
    amarillo_alto = (35, 255, 255)

    mascara_amarillo = cv2.inRange(
        imagen_hsv,
        amarillo_bajo,
        amarillo_alto
    )

    resultado_amarillo = cv2.bitwise_and(
        imagen_bgr,
        imagen_bgr,
        mask=mascara_amarillo
    )

    resultado_amarillo_rgb = cv2.cvtColor(
        resultado_amarillo,
        cv2.COLOR_BGR2RGB
    )

    # Resultado

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(imagen_rgb)
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(resultado_verde_rgb)
    plt.title("Color verde detectado")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(resultado_rojo_rgb)
    plt.title("Color rojo detectado")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(resultado_amarillo_rgb)
    plt.title("Color amarillo detectado")
    plt.axis("off")

    plt.tight_layout()
    plt.show()