import numpy as np
import cv2
from matplotlib import pyplot as plt
from a6_utils import drawEllipse
import glob


def first():

    file = open("data/points.txt", "r")
    pts = file.read()
    points = pts.split("\n")
    x = []
    y = []

    for p in points:
        temp = p.split(" ")
        x.append(int(temp[0]))
        y.append(int(temp[1]))

    rows = 2
    columns = len(x)
    N = columns

    mean = np.zeros((2, 1))
    mean[0] = (1 / N) * np.sum(x)
    mean[1] = (1 / N) * np.sum(y)

    Xd = np.zeros((rows, columns))
    Xd[0, :] = x - mean[0]
    Xd[1, :] = y - mean[1]

    C = np.dot(Xd, Xd.T)
    C *= (1 / (N - 1))

    U, D, V = np.linalg.svd(C)

    eigenValue1 = D[0]
    eigenValue2 = D[1]

    eigenVector1 = np.array(
        [[mean[0], mean[0] + (eigenValue1 * U[0][0])],
         [mean[1], mean[1] + (eigenValue1 * U[1][0])]],
        dtype=object
    )

    eigenVector2 = np.array(
        [[mean[0], mean[0] + (eigenValue2 * U[0][1])],
         [mean[1], mean[1] + (eigenValue2 * U[1][1])]],
        dtype=object
    )

    plt.plot(x, y, 'b.')
    drawEllipse(mean, C)
    plt.plot(eigenVector1[0], eigenVector1[1], "r")
    plt.plot(eigenVector2[0], eigenVector2[1], "g")
    plt.axis('equal')
    plt.axis([-10, 10, -10, 10])
    plt.show()

    # c)
    plt.bar(("First eigenvalue", "Second eigenvalue"), (eigenValue1 / eigenValue1, eigenValue2 / eigenValue1))
    plt.show()

    # d)
    yi = np.dot(U.T, Xd)
    yi[1, :] = 0
    xi = np.dot(U, yi) + mean

    plt.plot(xi[0, :], xi[1, :], 'b.')
    #drawEllipse(mean, C)
    #plt.plot(eigenVector1[0], eigenVector1[1], "r")
    #plt.plot(eigenVector2[0], eigenVector2[1], "g")
    plt.axis('equal')
    plt.axis([-10, 10, -10, 10])
    plt.show()

    # e)
    qPoint = [3, 6]
    minDist = -1
    closestX = 0
    closestY = 0

    for p, q in zip(x, y):
        dist = np.sqrt((p - qPoint[0])**2 + (q - qPoint[1])**2)

        if minDist == -1 or dist < minDist:
            minDist = dist
            closestX = p
            closestY = q


    print(minDist, closestX, closestY)
    # Which point is the closest?
    # The closest point is [5, 4], with distance 2.8283

    x.append(qPoint[0])
    y.append(qPoint[1])

    rows = 2
    columns = len(x)
    N = columns

    mean = np.zeros((2, 1))
    mean[0] = (1 / N) * np.sum(x)
    mean[1] = (1 / N) * np.sum(y)

    Xd = np.zeros((rows, columns))
    Xd[0, :] = x - mean[0]
    Xd[1, :] = y - mean[1]

    C = np.dot(Xd, Xd.T)
    C *= (1 / (N - 1))

    U, D, V = np.linalg.svd(C)

    yi = np.dot(U.T, Xd)
    yi[1, :] = 0
    xi = np.dot(U, yi) + mean
    #print(xi)

    minDist = -1
    closestX = 0
    closestY = 0

    for p, q in zip(xi[0], xi[1]):
        dist = np.sqrt((p - qPoint[0]) ** 2 + (q - qPoint[1]) ** 2)
        # print(dist)

        if minDist == -1 or dist < minDist:
            minDist = dist
            closestX = p
            closestY = q

    print(minDist, closestX, closestY)

    # After projecting all points (including qPoint) to PCA subspace and
    # calculating the distances to qPoint, the closest point is qPoint.


def second():

    file = open("data/points.txt", "r")
    pts = file.read()
    points = pts.split("\n")
    x = []
    y = []

    for p in points:
        temp = p.split(" ")
        x.append(int(temp[0]))
        y.append(int(temp[1]))

    rows = 2
    columns = len(x)
    N = columns
    m = rows

    mean = np.zeros((2, 1))
    mean[0] = (1 / N) * np.sum(x)
    mean[1] = (1 / N) * np.sum(y)

    Xd = np.zeros((rows, columns))
    Xd[0, :] = x - mean[0]
    Xd[1, :] = y - mean[1]

    C = np.dot(Xd.T, Xd)
    C *= (1 / (m - 1))

    U, D, V = np.linalg.svd(C)

    U = np.dot(Xd, U) / np.sqrt(D * (m - 1))

    yi = np.dot(U.T, Xd)
    xi = np.dot(U, yi) + mean
    print(xi)


def third():

    # a)
    path = "data/faces/1/*.png"
    M, m, n = data_prep(path)

    # b)
    U, mean = dual_pca(M)

    for i in range(0, 5):
        eigen = U[:, i]
        eigen = np.reshape(eigen, (m, n))
        plt.subplot(1, 5, i + 1)
        plt.imshow(eigen, cmap="gray")
        plt.axis("off")

    plt.show()

    I = cv2.imread("data/faces/1/001.png")
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    image = np.reshape(I, -1)
    Xd = image - mean

    yi = np.dot(U.T, Xd.T)
    xi = np.dot(U, yi) + mean

    imageV = np.reshape(xi, I.shape)
    imageC = xi.copy()
    imageC[4074] = 0
    imageC = np.reshape(imageC, I.shape)

    yi[4] = 0
    xi = np.dot(U, yi) + mean

    imageVC = np.reshape(xi, I.shape)

    plt.subplot(1, 4, 1)
    plt.imshow(I, cmap="gray")
    plt.subplot(1, 4, 2)
    plt.imshow(imageV, cmap="gray")
    plt.subplot(1, 4, 3)
    plt.imshow(imageC, cmap="gray")
    plt.subplot(1, 4, 4)
    plt.imshow(imageVC, cmap="gray")
    plt.show()

    # c)
    path = "data/faces/2/*.png"
    M, m, n = data_prep(path)
    U, mean = dual_pca(M)

    I = cv2.imread("data/faces/2/001.png")
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    image = np.reshape(I, -1)
    Xd = image - mean

    plt.subplot(1, 6, 1)
    plt.imshow(I, cmap="gray")
    plt.axis("off")

    keep = 32
    i = 2

    while keep != 1:

        yi = np.dot(U.T, Xd.T)
        yi[int(keep):] = 0
        xi = np.dot(U, yi) + mean

        imageC = np.reshape(xi, I.shape)

        plt.subplot(1, 6, i)
        plt.imshow(imageC, cmap="gray")
        plt.axis("off")

        i += 1
        keep /= 2

    plt.show()

    # e)
    path = "data/faces/1/*.png"
    M, m, n = data_prep(path)
    U, mean = dual_pca(M)

    I = cv2.imread("data/elephant.jpg")
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    image = np.reshape(I, -1)
    Xd = image - mean

    yi = np.dot(U.T, Xd.T)
    xi = np.dot(U, yi) + mean

    image = np.reshape(xi, I.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def data_prep(path):

    images = glob.glob(path)

    I = cv2.imread(images[0])
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    m, n = I.shape
    matrix = np.zeros((m * n, 64))

    for i, image in enumerate(images):

        I = cv2.imread(image)
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        matrix[:, i] = np.reshape(I, -1)

    return matrix, m, n


def dual_pca(matrix):

    m, N = matrix.shape
    mean = np.mean(matrix, axis=1)

    Xd = matrix.copy()

    for i in range(0, N):
        Xd[:, i] -= mean

    C = np.dot(Xd.T, Xd)
    C *= (1 / (m - 1))

    U, D, V = np.linalg.svd(C)
    D = D + 1e-15

    U = np.dot(Xd, U) / np.sqrt(D * (m - 1))

    return U, mean


def main():

    #first()
    #second()
    #third()

    run = input("Run first? (yes/any key) ")
    if run.lower() == "yes":
       first()

    run = input("Run second? (yes/any key) ")
    if run.lower() == "yes":
       second()

    run = input("Run third? (yes/any key) ")
    if run.lower() == "yes":
       third()


if __name__ == '__main__':

    main()
