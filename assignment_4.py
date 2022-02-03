import numpy as np
import cv2
from matplotlib import pyplot as plt
from a4_utils import simple_descriptors
from a4_utils import display_matches
from scipy.linalg import svd

def first():

    #I = cv2.imread('data/graf/graf1.jpg')
    I = cv2.imread('data/test_points.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    # a)
    hessian_points(I, 3, 0, True)
    hessian_points(I, 6, 0, True)
    hessian_points(I, 9, 0, True)

    (xPoint, yPoint) = hessian_points(I, 1, 100)
    draw_points(I, xPoint, yPoint, 1, 100, "Hessian")
    (xPoint, yPoint) = hessian_points(I, 1, 200)
    draw_points(I, xPoint, yPoint, 1, 200, "Hessian")
    (xPoint, yPoint) = hessian_points(I, 1, 300)
    draw_points(I, xPoint, yPoint, 1, 300, "Hessian")

    # b)
    harris_points(I, 3, 0, True)
    harris_points(I, 6, 0, True)
    harris_points(I, 9, 0, True)

    (xPoint, yPoint) = harris_points(I, 1, 5000)
    draw_points(I, xPoint, yPoint, 1, 5000, "Harris")
    (xPoint, yPoint) = harris_points(I, 1, 7000)
    draw_points(I, xPoint, yPoint, 1, 7000, "Harris")
    (xPoint, yPoint) = harris_points(I, 1, 10000)
    draw_points(I, xPoint, yPoint, 1, 10000, "Harris")


def hessian_points(image, sigma, thresh=0, draw=False):

    (_, _, Ixx, Iyy, Ixy) = partial_derivatives_image(image, sigma)

    hessian_det = np.zeros(image.shape)

    hessian_det = sigma ** 4 * (Ixx * Iyy - Ixy ** 2)

    if draw:
        plt.imshow(hessian_det, cmap="gray")
        plt.title("Hessian sigma = " + str(sigma))
        plt.show()

    xPoint = []
    yPoint = []

    if thresh != 0:
        (xPoint, yPoint) = hessian_extended(hessian_det, thresh)

    return (xPoint, yPoint)

def hessian_extended(hessian_det, thresh):

    xPoint = []
    yPoint = []
    temp = []

    hessian_det = np.pad(hessian_det, [(1, 1), (1, 1)], mode="constant")
    (h, w) = hessian_det.shape

    pos = [[1, 1, 0, -1, -1, -1, 0, 1],
           [0, 1, 1, 1, 0, -1, -1, -1]]

    for y in range(0, h - 1):
        for x in range(0, w - 1):

            if hessian_det[y][x] >= thresh:
                for i in range(0, 8):
                    ix = x + pos[0][i]
                    iy = y + pos[1][i]

                    temp.append(hessian_det[iy][ix])

                maxP = np.max(temp)

                if maxP <= hessian_det[y][x]:
                    xPoint.append(x)
                    yPoint.append(y)

                temp.clear()

    return (xPoint, yPoint)


def harris_points(image, sigma, thresh=0, draw=False):

    harris = np.zeros(image.shape)

    sigma_ = 1.6 * sigma
    size = round(2 * 3 * sigma_ + 1)
    alfa = 0.06

    (Ix, Iy, _, _, Ixy) = partial_derivatives_image(image, sigma)

    gk = gauss(size, sigma_)

    gIx = cv2.filter2D(Ix**2, -1, gk)
    gIy = cv2.filter2D(Iy**2, -1, gk)
    gIxy = cv2.filter2D(Ixy, -1, gk)

    gIx = sigma**2 * gIx
    gIy = sigma**2 * gIy
    gIxy = sigma**2 * gIxy

    det = (gIx * gIy) - (gIxy * gIxy)
    trace = gIx + gIy

    harris = det - alfa * trace**2

    if draw:
        plt.imshow(harris, cmap="gray")
        plt.title("Harris sigma = " + str(sigma))
        plt.show()

    xPoint = []
    yPoint = []

    if thresh != 0:
        (xPoint, yPoint) = harris_extended(harris, thresh)

    return (xPoint, yPoint)


def harris_extended(harris, thresh):

    xPoint = []
    yPoint = []
    temp = []

    harris = np.pad(harris, [(1, 1), (1, 1)], mode="constant")
    (h, w) = harris.shape

    pos = [[1, 1, 0, -1, -1, -1, 0, 1],
           [0, 1, 1, 1, 0, -1, -1, -1]]

    for y in range(0, h - 1):
        for x in range(0, w - 1):

            if harris[y][x] >= thresh:
                for i in range(0, 8):
                    ix = x + pos[0][i]
                    iy = y + pos[1][i]

                    temp.append(harris[iy][ix])

                maxP = np.max(temp)

                if maxP < harris[y][x]:
                    xPoint.append(x)
                    yPoint.append(y)

                temp.clear()

    return (xPoint, yPoint)


def draw_points(image, xPoint, yPoint, sigma, thresh, name):

    plt.plot(xPoint, yPoint, marker='x', color="red", linestyle="None")
    plt.imshow(image, cmap="gray")
    plt.title(name + " sigma = " + str(sigma) + " and threshold = " + str(thresh))
    plt.show()


def second():

    I = cv2.imread('data/graf/graf1_small.jpg')
    #I = cv2.imread('data/graf/graf1.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    J = cv2.imread('data/graf/graf2_small.jpg')
    #J = cv2.imread('data/graf/graf2.jpg') #700
    J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)
    #J = np.pad(J, [(1, 2), (1, 1)], mode="constant")

    #print(I.shape, J.shape)

    (xPointI, yPointI) = hessian_points(I, 1, 400)
    #(xPointI, yPointI) = harris_points(I, 1, 50000)
    (xPointJ, yPointJ) = hessian_points(J, 1, 400)
    #(xPointJ, yPointJ) = harris_points(J, 1, 50000)

    pointsI = []
    pointsI = to_list(xPointI, yPointI)

    pointsJ = []
    pointsJ = to_list(xPointJ, yPointJ)

    #J = np.pad(J, [(1, 2), (1, 1)], mode="constant")
    I = np.copy(I[:289][:340])
    #print(I.shape, J.shape)

    descI = []
    descI = simple_descriptors(I, pointsI, 1)

    descJ = []
    descJ = simple_descriptors(J, pointsJ, 1)

    matches = find_correspondences(descI, descJ)

    display_matches(I, J, pointsI, pointsJ, matches)

    find_matches(I, J, 400)

    # testing with the same image
    #T = cv2.imread('data/test_points.jpg')
    #T = cv2.cvtColor(T, cv2.COLOR_RGB2GRAY)

    #(xPointT, yPointT) = hessian_points(T, 1, 300)
    #pointsT = []
    #pointsT = to_list(xPointT, yPointT)

    #descT = simple_descriptors(T, pointsT, 1)
    #matchesT = find_correspondences(descT, descT)
    #display_matches(T, T, pointsT, pointsT, matchesT)
    #find_matches(T, T, 100)


def find_correspondences(list1, list2):

    index = 0
    matchTemp = 0
    match = []
    minDist = 0
    first = True

    for x in list1:
        for y in list2:
            dist = np.sqrt(0.5 * np.sum((np.sqrt(x) - np.sqrt(y)) ** 2))
            #dist = np.linalg.norm(x - y)

            if first:
                minDist = dist
                first = False

            if dist <= minDist:
                minDist = dist
                matchTemp = index

            index += 1

        first = True
        index = 0
        match.append(matchTemp)

    matches = []
    iter = 0

    for x in match:

        matches.append((iter, x))
        iter += 1

    return matches


def find_matches(I, J, thresh, radius=40):

    (xPointI, yPointI) = hessian_points(I, 1, thresh)
    #(xPointI, yPointI) = harris_points(I, 1, 50000)
    (xPointJ, yPointJ) = hessian_points(J, 1, thresh)
    #(xPointJ, yPointJ) = harris_points(J, 1, 50000)

    pointsI = []
    pointsI = to_list(xPointI, yPointI)

    pointsJ = []
    pointsJ = to_list(xPointJ, yPointJ)

    descI = []
    descI = simple_descriptors(I, pointsI, 1, 16, radius)

    descJ = []
    descJ = simple_descriptors(J, pointsJ, 1, 16, radius)

    matchesLR = find_correspondences(descI, descJ)
    matchesRL = find_correspondences(descJ, descI)
    matchesRL = [t[::-1] for t in matchesRL]
    # display_matches(I, J, pointsI, pointsJ, matchesLR)
    #display_matches(I, J, pointsI, pointsJ, matchesRL)

    rez = list(set(matchesRL).intersection(matchesLR))
    #print(rez)
    display_matches(I, J, pointsI, pointsJ, rez)

    return (pointsI, pointsJ, rez)


def to_list(xPoint, yPoint):

    points = []

    for i in range(len(xPoint)):
        x = xPoint[i]
        y = yPoint[i]
        points.append((x, y))

    return points


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


def partial_derivatives_image(image, sigma):

    image = image.astype(float)

    #sigma = 2
    size = round((2 * 3 * sigma) + 1)

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


def third():

    # New York
    I = cv2.imread('data/newyork/newyork1.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    J = cv2.imread('data/newyork/newyork2.jpg')
    J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

    file = open('data/newyork/newyork.txt', 'r')

    (pointsI, pointsJ, matches) = read_data(file)
    H = estimate_homography(pointsI, pointsJ, matches)
    new = cv2.warpPerspective(I, H, I.shape)
    display_images(I, J, new)

    # Graf
    G1 = cv2.imread('data/graf/graf1.jpg')
    G1 = cv2.cvtColor(G1, cv2.COLOR_RGB2GRAY)

    G2 = cv2.imread('data/graf/graf2.jpg')
    G2 = cv2.cvtColor(G2, cv2.COLOR_RGB2GRAY)

    file = open('data/graf/graf.txt', 'r')

    (pointsG1, pointsG2, matches) = read_data(file)
    H = estimate_homography(pointsG1, pointsG2, matches)
    new = cv2.warpPerspective(G1, H, G1.shape)
    display_images(G1, G2, new)

    # c)
    (pointsI, pointsJ, matches) = find_matches(I, J, 400, 30)
    matches = matches[:10]
    H = estimate_homography(pointsI, pointsJ, matches)
    new = cv2.warpPerspective(I, H, I.shape)
    display_images(I, J, new)

    # e)
    (pointsI, pointsJ, matches) = find_matches(I, J, 400, 30)
    matches = matches[:10]
    H = estimate_homography(pointsI, pointsJ, matches)
    new = warp_perspective(I, H)
    display_images(I, J, new)


def estimate_homography(p1, p2, matches):

    rows = len(matches) * 2
    columns = 9

    A = np.zeros((rows, columns))

    Xr = []
    Xt = []

    for i in matches:

        Xr.append(p1[i[0]])
        Xt.append(p2[i[1]])

    j = 0
    k = 0

    for i in range(0, rows):

        if i % 2 == 0:
            A[i][0] = Xr[k][0]
            A[i][1] = Xr[k][1]
            A[i][2] = 1
            A[i][6] = -Xt[k][0] * Xr[k][0]
            A[i][7] = -Xt[k][0] * Xr[k][1]
            A[i][8] = -Xt[k][0]
            k += 1
        else:
            A[i][3] = Xr[j][0]
            A[i][4] = Xr[j][1]
            A[i][5] = 1
            A[i][6] = -Xt[j][1] * Xr[j][0]
            A[i][7] = -Xt[j][1] * Xr[j][1]
            A[i][8] = -Xt[j][1]
            j += 1

    #[U, S, V] = svd(A)
    [U, S, V] = np.linalg.svd(A)
    V = V.T

    h = []

    for i in range(0, len(V)):
        h.append(V[i][len(V) - 1])

    h = np.array(h)
    h /= V[len(V) - 1][len(V) - 1]
    H = h.reshape((3, 3))

    return H


def warp_perspective(I, H):

    homogeneous = []
    h, w = I.shape

    for i in range(0, 250):
        for j in range(0, 250):
            homogeneous.append((i, j, 1))

    Xt = np.dot(H, np.array(homogeneous).T)

    Xt = Xt / Xt[-1]

    x = Xt[:1]
    y = Xt[1:2]

    x = x.reshape(h, w)
    x = np.copy(x.astype(np.float32))
    y = y.reshape(h, w)
    y = np.copy(y.astype(np.float32))

    new = cv2.remap(I, y, x, cv2.INTER_LINEAR)

    return new


def read_data(file):

    data = file.read()
    data = data.split()
    data = np.array(data)
    data = np.copy(data.astype(np.float64))
    data = data.reshape((4, 4))

    pointsItemp = data[:, :2]
    pointsJtemp = data[:, -2:]

    pointsI = []
    pointsJ = []

    for i in range(0, 4):
        pointsI.append((pointsItemp[i][0], pointsItemp[i][1]))
        pointsJ.append((pointsJtemp[i][0], pointsJtemp[i][1]))

    matches = []

    for i in range(0, 4):
        matches.append((i, i))

    return (pointsI, pointsJ, matches)


def display_images(I, J, new):

    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(I, cmap="gray")
    plt.title("I1")
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(J, cmap="gray")
    plt.title("I2")
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(new, cmap="gray")
    plt.title("I1 to I2")
    plt.show()


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
