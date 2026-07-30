import cv2
import matplotlib.pyplot as plt

filename = "frutas.jpg"
image_bgr = cv2.imread(filename)

if image_bgr is None:
    print("Error: no se encontró frutas.jpg")
else:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    umbral_bajo = (25, 40, 40)
    umbral_alto = (70, 255, 255)

    mask = cv2.inRange(image_hsv, umbral_bajo, umbral_alto)

    resultado_bgr = cv2.bitwise_and(
        image_bgr,
        image_bgr,
        mask=mask
    )

    resultado_rgb = cv2.cvtColor(
        resultado_bgr,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Máscara verde")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(resultado_rgb)
    plt.title("Color verde detectado")
    plt.axis("off")

    plt.tight_layout()
    plt.show()