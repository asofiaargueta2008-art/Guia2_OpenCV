import cv2

imagen1 = cv2.imread("frutas.jpg")
imagen2 = cv2.imread("huerto.jpg")

if imagen1 is None:
    print("Error: no se encontró frutas.jpg")

elif imagen2 is None:
    print("Error: no se encontró huerto.jpg")

else:
    imagen2_redimensionada = cv2.resize(
        imagen2,
        (imagen1.shape[1], imagen1.shape[0])
    )

    imagen_combinada = cv2.addWeighted(
        imagen1, 0.7,
        imagen2_redimensionada, 0.3,
        0
    )

    cv2.imshow("Imagen combinada", imagen_combinada)

    cv2.waitKey(0)

    cv2.destroyAllWindows()