import numpy as np
import cv2
from matplotlib import pyplot as plt
from a3_utils import draw_line

def first():

    # a)
    #   Iy(x,y)  = d/dy * g(y) * [g(x) * I(x,y)]
    #   Iyy(x,y) = d/dy * g(y) * [g(x) * Iy(x,y)]
    #   Ixy(x,y) = d/dy * g(y) * [g(x) * Ix(x,y)]

    # b) c)
    sigma = 2
    w = (2 * 3 * sigma) + 1

    G = gauss(w, sigma)
    d_noflip = gaussdx(w, sigma)
    D = np.flip(d_noflip)

    impulse = np.zeros((25, 25))
    impulse[12, 12] = 255

    a = cv2.filter2D(impulse, -1, G)
    a = cv2.filter2D(a, -1, G.T)

    b = cv2.filter2D(impulse, -1, G)
    b = cv2.filter2D(b, -1, D.T)

    c = cv2.filter2D(impulse, -1, D)
    c = cv2.filter2D(c, -1, G.T)

    d = cv2.filter2D(impulse, -1, G.T)
    d = cv2.filter2D(d, -1, D)

    e = cv2.filter2D(impulse, -1, D.T)
    e = cv2.filter2D(e, -1, G)

    plt.subplot(2, 3, 1)
    plt.imshow(impulse, cmap="gray")
    plt.title("Impulse")
    plt.axis("off")
    plt.subplot(2, 3, 2)
    plt.imshow(a, cmap="gray")
    plt.title("G, Gt")
    plt.axis("off")
    plt.subplot(2, 3, 3)
    plt.imshow(b, cmap="gray")
    plt.title("G, Dt")
    plt.axis("off")
    plt.subplot(2, 3, 4)
    plt.imshow(c, cmap="gray")
    plt.title("D, Gt")
    plt.axis("off")
    plt.subplot(2, 3, 5)
    plt.imshow(d, cmap="gray")
    plt.title("Gt, D")
    plt.axis("off")
    plt.subplot(2, 3, 6)
    plt.imshow(e, cmap="gray")
    plt.title("Dt, G")
    plt.axis("off")
    plt.show()

    # Is the order of operations important?
    # The order is not important, because convolution is associative.

    # d)
    image = cv2.imread('images/museum.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (Ix, Iy, Ixx, Iyy, Ixy) = partial_derivatives_image(image)
    (m, da) = gradient_magnitude(image)

    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(2, 4, 2)
    plt.imshow(Ix, cmap="gray")
    plt.title("Ix")
    plt.axis("off")
    plt.subplot(2, 4, 3)
    plt.imshow(Iy, cmap="gray")
    plt.title("Iy")
    plt.axis("off")
    plt.subplot(2, 4, 4)
    plt.imshow(m, cmap="gray")
    plt.title("Imag")
    plt.axis("off")
    plt.subplot(2, 4, 5)
    plt.imshow(Ixx, cmap="gray")
    plt.title("Ixx")
    plt.axis("off")
    plt.subplot(2, 4, 6)
    plt.imshow(Ixy, cmap="gray")
    plt.title("Ixy")
    plt.axis("off")
    plt.subplot(2, 4, 7)
    plt.imshow(Iyy, cmap="gray")
    plt.title("Iyy")
    plt.axis("off")
    plt.subplot(2, 4, 8)
    plt.imshow(da, cmap="gray")
    plt.title("Idir")
    plt.axis("off")
    plt.show()


def gaussdx(w, sigma):

    x = np.linspace(-(w - 1) / 2, (w - 1) / 2, w)

    gauss_kernel = -(1 / (np.sqrt(2 * np.pi) * sigma ** 3)) * x * np.exp(x ** 2 / (-2 * sigma ** 2))
    gauss_kernel /= np.sum(np.abs(gauss_kernel))

    gk = gauss_kernel.reshape(-1, 1).copy().T

    return gk


def gauss(size, sigma):

    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)

    gauss_kernel = (1 / np.sqrt(2 * np.pi) * sigma) * np.exp(x ** 2 / (-2 * sigma ** 2))
    gauss_kernel /= np.sum(gauss_kernel)

    gk = gauss_kernel.reshape(-1, 1).copy().T

    return gk


def partial_derivatives_image(image):

    image = image.astype(float)

    sigma = 2
    size = (2 * 3 * sigma) + 1

    # Ix
    y = gauss(size, sigma)
    x1 = gaussdx(size, sigma)
    x = np.flip(x1)
    Ix = cv2.filter2D(image.T, -1, y)
    Ix = cv2.filter2D(Ix.T, -1, x)

    # Iy
    x = gauss(size, sigma)
    y1 = gaussdx(size, sigma)
    y = np.flip(y1)
    Iy = cv2.filter2D(image, -1, x)
    Iy = cv2.filter2D(Iy.T, -1, y)
    Iy = Iy.T

    # Ixx
    x1 = gaussdx(size, sigma)
    x = np.flip(x1)
    y = gauss(size, sigma)
    Ixx = cv2.filter2D(Ix.T, -1, y)
    Ixx = cv2.filter2D(Ixx.T, -1, x)

    # Iyy
    x = gauss(size, sigma)
    y1 = gaussdx(size, sigma)
    y = np.flip(y1)
    Iyy = cv2.filter2D(Iy, -1, x)
    Iyy = cv2.filter2D(Iyy.T, -1, y)
    Iyy = Iyy.T

    # Ixy
    x = gauss(size, sigma)
    y1 = gaussdx(size, sigma)
    y = np.flip(y1)
    Ixy = cv2.filter2D(Ix, -1, x)
    Ixy = cv2.filter2D(Ixy.T, -1, y)
    Ixy = Ixy.T

    return (Ix, Iy, Ixx, Iyy, Ixy)


def gradient_magnitude(image):

    image = image.astype(float)

    sigma = 1
    size = (2 * 3 * sigma) + 1

    # Ix
    y = gauss(size, sigma)
    x1 = gaussdx(size, sigma)
    x = np.flip(x1)
    Ix = cv2.filter2D(image.T, -1, y)
    Ix = cv2.filter2D(Ix.T, -1, x)

    # Iy
    x = gauss(size, sigma)
    y1 = gaussdx(size, sigma)
    y = np.flip(y1)
    Iy = cv2.filter2D(image, -1, x)
    Iy = cv2.filter2D(Iy.T, -1, y)
    Iy = Iy.T

    m = np.sqrt((Ix ** 2 + Iy ** 2))
    da = np.arctan2(Iy, Ix)

    return (m, da)


def second():

    image = cv2.imread('images/museum.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sigma = 1
    theta = [10, 15, 20, 25]

    (_, _, binary_matrix) = findedges(image, sigma, theta[0])
    plt.subplot(2, 2, 1)
    plt.imshow(binary_matrix, cmap="gray")
    plt.title("Binary matrix, theta = 10")

    (_, _, binary_matrix) = findedges(image, sigma, theta[1])
    plt.subplot(2, 2, 2)
    plt.imshow(binary_matrix, cmap="gray")
    plt.title("Binary matrix, theta = 15")

    (_, _, binary_matrix) = findedges(image, sigma, theta[2])
    plt.subplot(2, 2, 3)
    plt.imshow(binary_matrix, cmap="gray")
    plt.title("Binary matrix, theta = 20")

    (mag, gradient, binary_matrix) = findedges(image, sigma, theta[3])
    plt.subplot(2, 2, 4)
    plt.imshow(binary_matrix, cmap="gray")
    plt.title("Binary matrix, theta = 25")

    #plt.imshow(binary_matrix, cmap="gray")
    #plt.title("Binary matrix")
    plt.show()

    # b)
    non_maxima = thin_edges(mag, gradient)

    # c)
    t_low = 10
    t_high = 15
    hysteresis(mag, non_maxima, t_low, t_high)


def findedges(I, sigma, theta):

    (m, da) = magnitude(I, sigma)

    binary_matrix = np.where(m < theta, 0, 1)

    return (m, da, binary_matrix)


def magnitude(image, sigma):

    image = image.astype(float)

    size = (2 * 3 * sigma) + 1

    # N = int((size - 1) / 2)
    # image = np.pad(image, ((N, N), (N, N)), constant_values=(0, 0))

    # Ix
    y = gauss(size, sigma)
    x1 = gaussdx(size, sigma)
    x = np.flip(x1)
    Ix = cv2.filter2D(image.T, -1, y)
    Ix = cv2.filter2D(Ix.T, -1, x)

    # Iy
    x = gauss(size, sigma)
    y1 = gaussdx(size, sigma)
    y = np.flip(y1)
    Iy = cv2.filter2D(image, -1, x)
    Iy = cv2.filter2D(Iy.T, -1, y)
    Iy = Iy.T

    m = np.sqrt((Ix ** 2 + Iy ** 2))
    da = np.arctan2(Iy, Ix)

    #da = da * 180 / np.pi

    return (m, da)


def thin_edges(mag, gradient):

    pos = [[1, 1, 0, -1, -1, -1, 0, 1, 1],
           [0, 1, 1, 1, 0, -1, -1, -1, 0]]

    non_maxima = np.zeros(mag.shape)

    (h, w) = mag.shape

    for y in range(0, h):
        for x in range(0, w):
            orientation = gradient[y][x]
            ix = round(np.abs((orientation / np.pi) * 4))

            x1 = x + pos[0][ix]
            y1 = y + pos[1][ix]
            x2 = x - pos[0][ix]
            y2 = y - pos[1][ix]

            if x1 < 0:
                x1 = 0
            elif x1 >= w:
                x1 = w - 1

            if x2 < 0:
                x2 = 0
            elif x2 >= w:
                x2 = w - 1

            if y1 < 0:
                y1 = 0
            elif y1 >= h:
                y1 = h - 1

            if y2 < 0:
                y2 = 0
            elif y2 >= h:
                y2 = h - 1

            if mag[y][x] >= mag[y1][x1] and mag[y][x] >= mag[y2][x2]:
                non_maxima[y][x] = mag[y][x]

    binary_matrix = np.where(non_maxima < 20, 0, 1)

    plt.imshow(binary_matrix, cmap="gray")
    plt.title("Non-maxima suppression")
    plt.show()

    return non_maxima


def hysteresis(mag, non_maxima, t_low, t_high):

    non_maxima[mag < t_low] = 0
    non_maxima = non_maxima.astype(np.uint8)

    stats = cv2.connectedComponentsWithStats(non_maxima)
    rows = stats[0]
    labels = stats[1]

    for i in range(1, rows):
        highestPx = np.max(mag[labels == i])

        if(highestPx > t_high):
            non_maxima[labels == i] = 255

    plt.imshow(non_maxima, cmap="gray")
    plt.title("Hysteresis")
    plt.show()


def third():

    # a)
    accumulator_matrix = lines_point(10, 10)
    plt.subplot(2, 2, 1)
    plt.imshow(accumulator_matrix, cmap="viridis")
    plt.title("x = 10, y = 10")

    accumulator_matrix = lines_point(30, 60)
    plt.subplot(2, 2, 2)
    plt.imshow(accumulator_matrix, cmap="viridis")
    plt.title("x = 30, y = 60")

    accumulator_matrix = lines_point(50, 20)
    plt.subplot(2, 2, 3)
    plt.imshow(accumulator_matrix, cmap="viridis")
    plt.title("x = 50, y = 20")

    accumulator_matrix = lines_point(80, 90)
    plt.subplot(2, 2, 4)
    plt.imshow(accumulator_matrix, cmap="viridis")
    plt.title("x = 80, y = 90")

    plt.tight_layout(pad=2.0)
    plt.show()

    # b) c) d)
    img_1 = cv2.imread('images/oneline.png')
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    (A, D) = hough_find_lines(img_1, 200, 300)
    plt.imshow(A, cmap="jet")
    plt.show()
    (A, D) = hough_find_lines(img_1, 200, 300, 10)
    plt.imshow(A, cmap="jet")
    plt.show()
    search_space(A, D, 1400, img_1, 200, 300)

    img_2 = cv2.imread('images/rectangle.png')
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    (A, D) = hough_find_lines(img_2, 200, 300)
    plt.imshow(A, cmap="jet")
    plt.show()
    (A, D) = hough_find_lines(img_2, 200, 300, 10)
    plt.imshow(A, cmap="jet")
    plt.show()
    search_space(A, D, 320, img_2, 200, 300)

    synthetic = np.zeros((100, 100))
    synthetic[80, 10] = 1
    synthetic[30, 50] = 1
    # (A, D) = hough_find_lines(synthetic, 300, 300)
    # search_space(A, D, 50, synthetic, 300, 300)

    # e)
    B = cv2.imread('images/bricks.jpg')
    B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    (_, _, binM) = findedges(B, 1, 35)
    B = cv2.imread('images/bricks.jpg')
    B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
    (A, D) = hough_find_lines(binM, 175, 750)
    #plt.imshow(A, cmap="jet")
    #plt.show()
    search_space_jpg(A, D, 500, binM, B, 175, 750)

    P = cv2.imread('images/pier.jpg')
    P = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY)
    (_, _, binM) = findedges(P, 1, 15)
    P = cv2.imread('images/pier.jpg')
    P = cv2.cvtColor(P, cv2.COLOR_BGR2RGB)
    (A, D) = hough_find_lines(binM, 175, 750)
    #plt.imshow(A, cmap="jet")
    #plt.show()
    search_space_jpg(A, D, 500, binM, P, 175, 750)

    # g)
    E = cv2.imread('images/eclipse.jpg')
    E = cv2.cvtColor(E, cv2.COLOR_BGR2GRAY)
    plt.subplot(1, 2, 1)
    plt.imshow(E, cmap="gray")
    plt.title("Original image")
    (_, _, binM) = findedges(E, 1, 10)
    r = 47
    A = circle_hough(binM, r)
    plt.subplot(1, 2, 2)
    plt.imshow(A)
    plt.title("Hough transform")
    plt.show()


def lines_point(x, y):

    t_bins = 300
    r_bins = 300

    accumulator_matrix = np.zeros((r_bins, t_bins))
    diagonal = 100

    theta_a = np.linspace(-np.pi / 2, np.pi / 2, t_bins)

    rho = x * np.cos(theta_a) + y * np.sin(theta_a)
    rho_moved = np.around(((rho + diagonal) / (2 * diagonal)) * r_bins)

    for i in range(0, t_bins):
        index = rho_moved[i]
        if 0 <= index < r_bins:
            accumulator_matrix[int(rho_moved[i])][i] += 1

    return accumulator_matrix


def hough_find_lines(image, t_bins, r_bins, threshold=0):

    if threshold != 0:
        (_, _, binary_matrix) = findedges(image, 1, threshold)
        (A, D) = hough_find_edges_p2(binary_matrix, t_bins, r_bins)
    else:
        (A, D) = hough_find_edges_p2(image, t_bins, r_bins)

    return (A, D)


def hough_find_edges_p2(image, t_bins, r_bins):

    A = np.zeros((r_bins, t_bins))
    (h, w) = image.shape

    D = round(np.sqrt(h ** 2 + w ** 2))

    theta_a = np.linspace(-np.pi / 2, np.pi / 2, t_bins)

    (ix, iy) = np.nonzero(image)

    for i in range(0, len(ix)):
        rho = ix[i] * np.cos(theta_a) + iy[i] * np.sin(theta_a)

        rho_moved = np.around(((rho + D) / (2 * D)) * r_bins)

        for j in range(0, t_bins):
            index = rho_moved[j]
            if 0 <= index < r_bins:
                A[int(index)][j] += 1

    return (A, D)


def nonmaxima_suppression_box(image, A):

    (h, w) = A.shape
    non_maxima = np.zeros(A.shape)

    pos = [[1, 1, 0, -1, -1, -1, 0, 1],
           [0, 1, 1, 1, 0, -1, -1, -1]]

    for y in range(1, h - 1):
        for x in range(1, w - 1):

            for i in range(0, 8):
                ix = x + pos[0][i]
                iy = y + pos[1][i]

                if A[y][x] <= A[iy][ix]:
                    non_maxima[y][x] = A[y][x]

    plt.imshow(non_maxima, cmap="jet")
    plt.show()


def search_space(A, D, threshold, image, t_bins, r_bins):

    (h, w) = image.shape

    res = np.where(A > threshold)

    plt.imshow(image, cmap="jet")

    rho = (res[0] * 2 * D / r_bins) - D
    theta_a = np.linspace(-np.pi / 2, np.pi / 2, t_bins)

    for i in range(0, len(res[0])):
        theta = theta_a[res[1][i]]
        draw_line(rho[i], theta, D)

    plt.xlim([0, w - 1])
    plt.ylim([h - 1, 0])
    plt.show()


def search_space_jpg(A, D, threshold, image, imageBGR, t_bins, r_bins):

    (h, w) = image.shape

    res = np.where(A > threshold)

    value = A[res]
    dict = {}

    for i in range(0, len(value)):

        dict[i] = value[i]

    dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    first10 = list(dict)[:10]
    index = []

    for tuple in first10:
        index.append(tuple[0])

    plt.imshow(imageBGR)

    rho = (res[0] * 2 * D / r_bins) - D
    theta_a = np.linspace(-np.pi / 2, np.pi / 2, t_bins)

    for i in range(0, len(index)):
        theta = theta_a[res[1][index[i]]]
        draw_line(rho[index[i]], theta, D)

    plt.xlim([0, w - 1])
    plt.ylim([h - 1, 0])
    plt.show()


def circle_hough(image, r):

    (y, x) = image.shape
    A = np.zeros((y, x))

    for i in range(0, y):
        for j in range(0, x):

            if image[i, j] == 1:

                for theta in range(0, 360):
                    a = i - round(r * np.cos(theta * np.pi / 180))
                    b = j + round(r * np.sin(theta * np.pi / 180))

                    if 0 <= a < y and 0 <= b < x:
                        A[a][b] += 1

    return A


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
