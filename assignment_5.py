import numpy as np
import cv2
from matplotlib import pyplot as plt
from a5_utils import normalize_points
from a5_utils import draw_epiline
from a5_utils import draw_epiline_2
from a5_utils import get_grid
from a4_utils import simple_descriptors
from a4_utils import display_matches
import random
import glob


def first():

    f = 0.0025
    T = 0.12

    pz = np.linspace(0.1, 10, num=100)

    d = (f / pz) * T

    plt.plot(d)
    plt.show()


def second():

    file = open('data/epipolar/house_points.txt', 'r')

    (points1, points2, matches) = read_data(file)
    file.close()
    F = fundamental_matrix(points1, points2, matches)

    point1 = (85, 233, 1)
    point2 = (67, 219, 1)
    average = reprojection_error(F, point1, point2)
    print("First average: ", average)

    pts1 = to_homogeneous(points1)
    pts2 = to_homogeneous(points2)
    average = 0

    for i in range(0, len(pts1)):
        average += reprojection_error(F, pts1[i], pts2[i])

    average /= len(pts1)
    print("Second average: ", average)

    inliers, outliers = get_inliers(F, pts1, pts2, 1)

    file = open('data/epipolar/house_matches.txt')
    correspondences = file.read()
    ransac_fundamental(correspondences, 1, 500)
    file.close()

    I = cv2.imread('data/epipolar/house1.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    J = cv2.imread('data/epipolar/house2.jpg')
    J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)
    (pointsI, pointsJ, rez) = find_matches(I, J, 70, 31)

    if len(pointsI) > len(pointsJ):
        pointsI = pointsI[:len(pointsJ)].copy()
    else:
        pointsJ = pointsJ[:len(pointsI)].copy()

    ransac_fundamental_v2(pointsI, pointsJ, rez, 5, 900)


def fundamental_matrix(points1, points2, matches, draw=True):

    rows = len(matches)
    columns = 9

    A = np.zeros((rows, columns))

    Xr = []
    Xt = []

    transformed1, T1 = normalize_points(np.array(points1))
    transformed2, T2 = normalize_points(np.array(points2))

    for i in matches:
        Xr.append(transformed1[i[0]])
        Xt.append(transformed2[i[1]])

    k = 0

    for i in range(0, rows):

        A[i, 0] = Xr[k][0] * Xt[k][0]
        A[i, 1] = Xr[k][0] * Xt[k][1]
        A[i, 2] = Xr[k][0]
        A[i, 3] = Xr[k][1] * Xt[k][0]
        A[i, 4] = Xr[k][1] * Xt[k][1]
        A[i, 5] = Xr[k][1]
        A[i, 6] = Xt[k][0]
        A[i, 7] = Xt[k][1]
        A[i, 8] = 1
        k += 1

    [U, D, V] = np.linalg.svd(A)

    V = V.T
    v9 = V[:, 8]
    Ft = v9.reshape((3, 3))

    [U, D, V] = np.linalg.svd(Ft)
    D = np.diag(D)
    D[-1] = 0

    #F_ = U * D * V.T
    F_ = np.dot(U, D)
    F_ = np.dot(F_, V)

    #F = T2.T * F_.T * T1
    F = np.dot(T2.T, F_.T)
    F = np.dot(F, T1)
    #print(F)

    [U, D, V] = np.linalg.svd(Ft)

    V = V.T
    e1 = V[:, 2] / V[2, 2]
    e2 = U[:, 2] / U[2, 2]

    if draw:
        I = cv2.imread('data/epipolar/house1.jpg')
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        J = cv2.imread('data/epipolar/house2.jpg')
        J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

        plt.subplot(1, 2, 1)
        plt.imshow(I, cmap="gray")

        h, w = I.shape

        for i in range(0, len(points2)):
            x = (points2[i][0], points2[i][1], 1)
            line = np.dot(F.T, x)
            draw_epiline(line, h, w)

        plt.subplot(1, 2, 2)
        plt.imshow(J, cmap="gray")

        h, w = J.shape

        for i in range(0, len(points1)):
            x = (points1[i][0], points1[i][1], 1)
            line = np.dot(F, x)
            draw_epiline(line, h, w)

        plt.show()

    return F


def reprojection_error(F, point1, point2):

    line1 = np.dot(F.T, point2)
    line2 = np.dot(F, point1)
    a1 = line1[0]
    b1 = line1[1]
    c1 = line1[2]
    a2 = line2[0]
    b2 = line2[1]
    c2 = line2[2]

    distance1 = (np.abs(a1 * point1[0] + b1 * point1[1] + c1) / (np.sqrt(a1**2 + b1**2)))
    distance2 = (np.abs(a2 * point2[0] + b2 * point2[1] + c2) / (np.sqrt(a2**2 + b2**2)))
    average = (distance1 + distance2) / 2

    return average


def get_inliers(F, pts1, pts2, e):

    inliers = []
    outliers = []
    average = 0

    for i in range(0, len(pts1)):

        line1 = np.dot(F.T, pts2[i])
        x1 = pts1[i]

        line2 = np.dot(F, pts1[i])
        x2 = pts2[i]

        distance1 = (np.abs(line1[0] * x1[0] + line1[1] * x1[1] + line1[2]) / (np.sqrt(line1[0]**2 + line1[1]**2)))
        distance2 = (np.abs(line2[0] * x2[0] + line2[1] * x2[1] + line2[2]) / (np.sqrt(line2[0]**2 + line2[1]**2)))

        if distance1 < e and distance2 < e:
            inliers.append((x1, x2))

        else:
            outliers.append((x1, x2))

    return inliers, outliers


def ransac_fundamental(correspondences, e, k):

    points = correspondences.strip().split("\n")
    pts1 = []
    pts2 = []
    inlierSize = 0
    bestInliers = []
    bestOutliers = []
    bestF = []

    for i in range(0, len(points)):

        pts = points[i].strip().split("  ")
        pts1.append((float(pts[0]), float(pts[1])))
        pts2.append((float(pts[2]), float(pts[3])))

    for i in range(0, k):

        x = []
        y = []

        rand = random.sample(range(0, len(pts1) - 1), 8)

        for r in range(0, 8):
            x.append(pts1[rand[r]])
            y.append(pts2[rand[r]])

        matches = []

        for m in range(0, 8):
            matches.append((m, m))

        F = fundamental_matrix(x, y, matches, False)

        p1 = to_homogeneous(pts1)
        p2 = to_homogeneous(pts2)
        inliers, outliers = get_inliers(F, p1, p2, e)

        if len(inliers) > inlierSize:
            inlierSize = len(inliers)
            bestInliers = inliers
            bestOutliers = outliers
            bestF = F

    percentage = inlierSize / len(points)
    percentage = round(percentage, 2)

    inX1 = []
    inY1 = []
    inX2 = []
    inY2 = []
    inliers1 = []
    inliers2 = []
    outX1 = []
    outY1 = []
    outX2 = []
    outY2 = []

    for i in range(0, len(bestInliers)):
        inX1.append(bestInliers[i][0][0])
        inY1.append(bestInliers[i][0][1])
        inliers1.append((bestInliers[i][0][0], bestInliers[i][0][1], 1))
        inX2.append(bestInliers[i][1][0])
        inY2.append(bestInliers[i][1][1])
        inliers2.append((bestInliers[i][1][0], bestInliers[i][1][1], 1))

    for i in range(0, len(bestOutliers)):
        outX1.append(bestOutliers[i][0][0])
        outY1.append(bestOutliers[i][0][1])
        outX2.append(bestOutliers[i][1][0])
        outY2.append(bestOutliers[i][1][1])

    error = 0

    for i in range(0, len(inliers1)):
        error += reprojection_error(bestF, inliers1[i], inliers2[i])

    error /= len(inliers1)
    error = round(error, 2)

    I = cv2.imread('data/epipolar/house1.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    J = cv2.imread('data/epipolar/house2.jpg')
    J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

    pointL = [inX1[0], inY1[0], 1]
    pointR = [inX2[0], inY2[0], 1]
    line1 = np.dot(bestF.T, pointR)
    line2 = np.dot(bestF, pointL)
    h, w = J.shape

    plt.plot(outX1, outY1, 'r.')
    plt.plot(inX1, inY1, 'b.')
    plt.plot(pointL[0], pointL[1], 'g.')
    plt.imshow(I, cmap="gray")
    plt.title("Outliers (red), Inliers (blue), Selected (green)")
    draw_epiline_2(line1, h, w)
    plt.show()

    plt.plot(outX2, outY2, 'r.')
    plt.plot(inX2, inY2, 'b.')
    plt.plot(pointR[0], pointR[1], 'g.')
    plt.imshow(J, cmap="gray")
    plt.title("Inliers: " + str(percentage) + " Error: " + str(error))
    draw_epiline_2(line2, h, w)
    plt.show()


def ransac_fundamental_v2(points1, points2, matchesP, e, k):

    inlierSize = 0
    bestInliers = []
    bestOutliers = []
    bestF = []

    for i in range(0, k):

        x = []
        y = []

        rand = random.sample(range(0, len(matchesP) - 1), 8)
        #rand = random.sample(range(0, len(points1) - 1), 8)

        for r in range(0, 8):
            x.append(points1[rand[r]])
            y.append(points2[rand[r]])

        matches = []

        for m in range(0, 8):
            matches.append((m, m))

        F = fundamental_matrix(x, y, matches, False)

        p1 = to_homogeneous(points1)
        p2 = to_homogeneous(points2)
        inliers, outliers = get_inliers(F, p1, p2, e)

        if len(inliers) > inlierSize:
            inlierSize = len(inliers)
            bestInliers = inliers
            bestOutliers = outliers
            bestF = F

    percentage = inlierSize / len(points1)
    percentage = round(percentage, 2)

    inX1 = []
    inY1 = []
    inX2 = []
    inY2 = []
    inliers1 = []
    inliers2 = []
    outX1 = []
    outY1 = []
    outX2 = []
    outY2 = []

    for i in range(0, len(bestInliers)):
        inX1.append(bestInliers[i][0][0])
        inY1.append(bestInliers[i][0][1])
        inliers1.append((bestInliers[i][0][0], bestInliers[i][0][1], 1))
        inX2.append(bestInliers[i][1][0])
        inY2.append(bestInliers[i][1][1])
        inliers2.append((bestInliers[i][1][0], bestInliers[i][1][1], 1))

    for i in range(0, len(bestOutliers)):
        outX1.append(bestOutliers[i][0][0])
        outY1.append(bestOutliers[i][0][1])
        outX2.append(bestOutliers[i][1][0])
        outY2.append(bestOutliers[i][1][1])

    error = 0

    for i in range(0, len(inliers1)):
        error += reprojection_error(bestF, inliers1[i], inliers2[i])

    error /= len(inliers1)
    error = round(error, 2)

    I = cv2.imread('data/epipolar/house1.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    J = cv2.imread('data/epipolar/house2.jpg')
    J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

    matchesLR = find_correspondences(inliers1, inliers2)
    matchesRL = find_correspondences(inliers2, inliers1)
    matchesRL = [t[::-1] for t in matchesRL]

    rez = list(set(matchesRL).intersection(matchesLR))

    bestL = inliers1[rez[0][0]]
    bestR = inliers2[rez[0][1]]
    errorMin = reprojection_error(bestF, bestL, bestR)

    for i in range(0, len(rez)):

        pointL = inliers1[rez[i][0]]
        pointR = inliers2[rez[i][1]]

        err = reprojection_error(bestF, pointL, pointR)

        if err < errorMin:
            errorMin = err
            bestL = pointL
            bestR = pointR

    line1 = np.dot(bestF.T, bestR)
    line2 = np.dot(bestF, bestL)

    h, w = J.shape

    plt.plot(outX1, outY1, 'r.')
    plt.plot(inX1, inY1, 'b.')
    plt.plot(bestL[0], bestL[1], 'g.')
    plt.imshow(I, cmap="gray")
    plt.title("Outliers (red), Inliers (blue), Selected (green)")
    draw_epiline_2(line1, h, w)
    plt.show()

    plt.plot(outX2, outY2, 'r.')
    plt.plot(inX2, inY2, 'b.')
    plt.plot(bestR[0], bestR[1], 'g.')
    plt.imshow(J, cmap="gray")
    plt.title("Inliers: " + str(percentage))
    draw_epiline_2(line2, h, w)
    plt.show()


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
    descI = simple_descriptors(I, pointsI, 2.5, 16, radius)

    descJ = []
    descJ = simple_descriptors(J, pointsJ, 2.5, 16, radius)

    matchesLR = find_correspondences(descI, descJ)
    matchesRL = find_correspondences(descJ, descI)
    matchesRL = [t[::-1] for t in matchesRL]
    #display_matches(I, J, pointsI, pointsJ, matchesLR)
    #display_matches(I, J, pointsI, pointsJ, matchesRL)

    rez = list(set(matchesRL).intersection(matchesLR))

    display_matches(I, J, pointsI, pointsJ, rez)

    return (pointsI, pointsJ, rez)


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


def to_list(xPoint, yPoint):

    points = []

    for i in range(len(xPoint)):
        x = xPoint[i]
        y = yPoint[i]
        points.append((x, y))

    return points


def third():

    file = open('data/epipolar/house_points.txt')
    correspondences = file.read()
    file.close()

    file = open('data/epipolar/house1_camera.txt')
    camera = file.read().strip().split("\n")
    C1 = np.zeros((3, 4))

    for i in range(0, 3):
        line = camera[i].strip().split(" ")
        for j in range(0, 4):
            C1[i][j] = line[j]

    file.close()

    file = open('data/epipolar/house2_camera.txt')
    camera = file.read().strip().split("\n")
    C2 = np.zeros((3, 4))

    for i in range(0, 3):
        line = camera[i].strip().split(" ")
        for j in range(0, 4):
            C2[i][j] = line[j]

    file.close()

    res = triangulation(correspondences, C1, C2)

    T = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    res = np.dot(res, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, pt in enumerate(res):
        plt.plot([pt[0]], [pt[1]], [pt[2]], 'r.')
        ax.text(pt[0], pt[1], pt[2], str(i))

    plt.show()

    reconstruction()


def triangulation(correspondences, P1, P2):

    points = correspondences.strip().split("\n")
    pts1 = []
    pts2 = []
    matches = []
    xPoint1 = []
    xPoint2 = []
    yPoint1 = []
    yPoint2 = []

    for i in range(0, len(points)):
        pts = points[i].strip().split(" ")
        pts1.append((float(pts[0]), float(pts[1])))
        pts2.append((float(pts[2]), float(pts[3])))
        matches.append((i, i))
        xPoint1.append(float(pts[0]))
        yPoint1.append(float(pts[1]))
        xPoint2.append(float(pts[2]))
        yPoint2.append(float(pts[3]))

    F = fundamental_matrix(pts1, pts2, matches, False)

    res = np.zeros((10, 3))

    for i in range(0, len(pts1)):

        x1x = np.zeros((3, 3))
        x1x[0][1] = -1
        x1x[0][2] = pts1[i][1]
        x1x[1][0] = 1
        x1x[1][2] = -pts1[i][0]
        x1x[2][0] = -pts1[i][1]
        x1x[2][1] = pts1[i][0]

        A1 = np.dot(x1x, P1)

        x2x = np.zeros((3, 3))
        x2x[0][1] = -1
        x2x[0][2] = pts2[i][1]
        x2x[1][0] = 1
        x2x[1][2] = -pts2[i][0]
        x2x[2][0] = -pts2[i][1]
        x2x[2][1] = pts2[i][0]

        A2 = np.dot(x2x, P2)

        A = np.zeros((4, 4))
        A[:2, :] = A1[0:2, :].copy()
        A[2:, :] = A2[0:2, :].copy()


        U, D, V = np.linalg.svd(A)
        D = np.diag(D)
        V = V.T

        eigenvector = V[:, 3] / V[3, 3]
        res[i] = eigenvector[:3]

    I = cv2.imread('data/epipolar/house1.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    J = cv2.imread('data/epipolar/house2.jpg')
    J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

    plt.plot(xPoint1, yPoint1, 'r.')
    plt.imshow(I, cmap="gray")
    plt.show()

    plt.plot(xPoint2, yPoint2, 'r.')
    plt.imshow(J, cmap="gray")
    plt.show()

    return res


def reconstruction():

    images = glob.glob('data/camera/*.jpg')

    objPoints = []
    imgObj = []

    for image in images:

        I = cv2.imread(image)
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        #plt.imshow(I, cmap="gray")
        #plt.show()

        ret, corners = cv2.findCirclesGrid(I, (4, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        test = cv2.drawChessboardCorners(I, (4, 11), corners, ret)

        grid = get_grid()
        objPoints.append(grid)
        imgObj.append(corners)

        #plt.imshow(test, cmap="gray")
        #plt.show()

    ret, C, distortion, r, t = cv2.calibrateCamera(objPoints, imgObj, I.shape[::-1], None, None)

    first = cv2.imread('data/object/IMG_20211219_155417.jpg')
    f = cv2.imread('data/object/IMG_20211219_155417.jpg')
    first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second = cv2.imread('data/object/IMG_20211219_155411.jpg')
    s = cv2.imread('data/object/IMG_20211219_155411.jpg')
    second = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)

    #plt.imshow(first, cmap="gray")
    #plt.show()
    #plt.imshow(second, cmap="gray")
    #plt.show()

    h, w = first.shape[:2]
    newC, roi = cv2.getOptimalNewCameraMatrix(C, distortion, (w, h), 1, (w, h))

    fst = cv2.undistort(first, C, distortion, None, newC)
    x, y, w, h = roi
    fst = fst[y:y+h, x:x+w]
    #plt.imshow(fst, cmap="gray")
    #plt.show()

    snd = cv2.undistort(second, C, distortion, None, newC)
    x, y, w, h = roi
    snd = snd[y:y + h, x:x + w]
    #plt.imshow(snd, cmap="gray")
    #plt.show()

    (pointsI, pointsJ, matches) = find_matches(fst, snd, 300, radius=30)

    F = fundamental_matrix(pointsI, pointsJ, matches, False)

    E = np.dot(newC.T, F)
    E = np.dot(E, newC)

    pI, pJ = get_matches(pointsI, pointsJ, matches)
    pI = np.array(pI)
    pI = pI.astype(np.int32).copy()
    pJ = np.array(pJ)
    pJ = pJ.astype(np.int32).copy()

    ret, R, T, _ = cv2.recoverPose(E, pI, pJ)

    RT1 = np.zeros((3, 4))
    RT1[:, :3] = R.copy()
    RT1[:, 3:] = T.copy()

    P1 = np.dot(newC, RT1)

    Id = [1, 1, 1]
    Id = np.diag(Id)
    RT2 = np.zeros((3, 4))
    RT2[:, :3] = Id.copy()

    P2 = np.dot(newC, RT2)

    res = np.zeros((len(pI), 3))

    for i in range(0, len(pI)):

        x1x = np.zeros((3, 3))
        x1x[0][1] = -1
        x1x[0][2] = pI[i][1]
        x1x[1][0] = 1
        x1x[1][2] = -pI[i][0]
        x1x[2][0] = -pI[i][1]
        x1x[2][1] = pI[i][0]

        A1 = np.dot(x1x, P1)

        x2x = np.zeros((3, 3))
        x2x[0][1] = -1
        x2x[0][2] = pJ[i][1]
        x2x[1][0] = 1
        x2x[1][2] = -pJ[i][0]
        x2x[2][0] = -pJ[i][1]
        x2x[2][1] = pJ[i][0]

        A2 = np.dot(x2x, P2)

        A = np.zeros((4, 4))
        A[:2, :] = A1[0:2, :].copy()
        A[2:, :] = A2[0:2, :].copy()

        U, D, V = np.linalg.svd(A)
        D = np.diag(D)
        V = V.T

        eigenvector = V[:, 3] / V[3, 3]
        res[i] = eigenvector[:3]


    fig, ax = plt.subplots()

    for i, pt in enumerate(pI):
        plt.plot([pI[i][0]], [pI[i][1]], 'r.')
        ax.text(pI[i][0], pI[i][1], str(i))

    plt.imshow(fst, cmap="gray")
    plt.show()

    fig, ax = plt.subplots()

    for i, pt in enumerate(pJ):
        plt.plot([pJ[i][0]], [pJ[i][1]], 'r.')
        ax.text(pJ[i][0], pJ[i][1], str(i))

    plt.imshow(snd, cmap="gray")
    plt.show()

    T = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, -1]])
    res = np.dot(res, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, pt in enumerate(res):
        plt.plot([pt[0]], [pt[1]], [pt[2]], 'r.')
        ax.text(pt[0], pt[1], pt[2], str(i))

    plt.show()


def get_matches(points1, points2, matches):

    p1 = []
    p2 = []

    for i in matches:

        p1.append(points1[i[0]])
        p2.append(points2[i[1]])

    return p1, p2


def to_homogeneous(points):

    pts = []

    for i in range(0, len(points)):
        pts.append((points[i][0], points[i][1], 1))

    return pts


def read_data(file):

    data = file.read()
    data = data.split()
    data = np.array(data)
    data = np.copy(data.astype(np.float64))
    data = data.reshape((10, 4))

    pointsItemp = data[:, :2]
    pointsJtemp = data[:, -2:]

    pointsI = []
    pointsJ = []

    for i in range(0, 10):
        pointsI.append((pointsItemp[i][0], pointsItemp[i][1]))
        pointsJ.append((pointsJtemp[i][0], pointsJtemp[i][1]))

    matches = []

    for i in range(0, 10):
        matches.append((i, i))

    return (pointsI, pointsJ, matches)


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
