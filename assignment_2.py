import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from a2_utils import read_data
from a2_utils import gauss_noise
from a2_utils import sp_noise

def first():

    I = cv2.imread('dataset/object_01_1.png')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    J = cv2.imread('dataset/object_02_1.png')
    J = cv2.cvtColor(J, cv2.COLOR_BGR2RGB)

    K = cv2.imread('dataset/object_03_1.png')
    K = cv2.cvtColor(K, cv2.COLOR_BGR2RGB)

    histogram1 = myhist3(I, 8)
    histogram2 = myhist3(J, 8)
    histogram3 = myhist3(K, 8)

    # Euclidian distance
    l2_1 = compare_histograms(histogram1, histogram1, "euclidian")
    print("L2: ", l2_1)
    l2_2 = compare_histograms(histogram1, histogram2, "euclidian")
    print("L2: ", l2_2)
    l2_3 = compare_histograms(histogram1, histogram3, "euclidian")
    print("L2: ", l2_3)

    # Chi-square distance
    distance = compare_histograms(histogram1, histogram1, "chi-sqare")
    print("Chi-square: ", distance)
    distance = compare_histograms(histogram1, histogram2, "chi-sqare")
    print("Chi-square: ", distance)
    distance = compare_histograms(histogram1, histogram3, "chi-sqare")
    print("Chi-square: ", distance)

    # Intersection distance
    distance = compare_histograms(histogram1, histogram1, "intersection")
    print("Intersection: ", distance)
    distance = compare_histograms(histogram1, histogram2, "intersection")
    print("Intersection: ", distance)
    distance = compare_histograms(histogram1, histogram3, "intersection")
    print("Intersection: ", distance)

    # Hellinger distance
    distance = compare_histograms(histogram1, histogram1, "hellinger")
    print("Hellinger: ", distance)
    distance = compare_histograms(histogram1, histogram2, "hellinger")
    print("Hellinger: ", distance)
    distance = compare_histograms(histogram1, histogram3, "hellinger")
    print("Hellinger: ", distance)

    plt.subplot(2, 3, 1)
    plt.axis("off")
    plt.imshow(I)
    plt.subplot(2, 3, 2)
    plt.axis("off")
    plt.imshow(J)
    plt.subplot(2, 3, 3)
    plt.axis("off")
    plt.imshow(K)

    histogram1 = histogram1.reshape(-1)
    plt.subplot(2, 3, 4)
    plt.bar(range(0, 512), histogram1, width=5)
    #plt.title("l2(H1, H1) = " + str(l2_1))

    histogram2 = histogram2.reshape(-1)
    plt.subplot(2, 3, 5)
    plt.bar(range(0, 512), histogram2, width=5)
    #plt.title("l2(H1, H2) = " + str(l2_2))

    histogram3 = histogram3.reshape(-1)
    plt.subplot(2, 3, 6)
    plt.bar(range(0, 512), histogram3, width=5)
    #plt.title("l2(H1, H3) = " + str(l2_3))

    #plt.subplots_adjust(wspace=0.6)
    plt.show()

    # Which image (object_02_1.png or object_03_1.png) is more similar to image object_01_1.png considering the L2 distance
    # Euclidian distance between images 1 and 3 is smaller than between images 1 and 2, so we can conclude that they are more similar.
    # We can see that all three histograms contain a strongly expressed component (one bin has a much higher value than the others). Which color does this bin represent
    # The bin with the highest value represents the black colour of the background.

    image_retrieval("C:\\Users\\matej\\PycharmProjects\\assignment 2\\dataset", 8)

def myhist3(I, n_bins):

    H = np.zeros((n_bins, n_bins, n_bins))

    im0 = I[:, :, 0].reshape(-1)
    im1 = I[:, :, 1].reshape(-1)
    im2 = I[:, :, 2].reshape(-1)

    bin_size = 255 / n_bins

    for i in range(0, im0.size):

        x = im0[i]
        y = im1[i]
        z = im2[i]

        ix0 = int(x / bin_size)
        ix1 = int(y / bin_size)
        ix2 = int(z / bin_size)

        if ix0 >= n_bins:
            ix0 = n_bins - 1

        if ix1 >= n_bins:
            ix1 = n_bins - 1

        if ix2 >= n_bins:
            ix2 = n_bins - 1

        H[ix0, ix1, ix2] += 1

    H /= I.size
    H *= 3

    return H

def compare_histograms(hist1, hist2, distName):

    dist = 0

    if distName == "euclidian":

        #dist = np.linalg.norm(hist1 - hist2)
        dist = np.sqrt(np.sum((hist1 - hist2)**2))

    elif distName == "chi-sqare":

        e0 = 0.0000000001
        dist = 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2 + e0))

    elif distName == "intersection":

        dist = 1 - np.sum(np.minimum(hist1, hist2))

    elif distName == "hellinger":

        dist = np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2))**2))

    return dist

def image_retrieval(path, n_bins):

    listH = []
    distance_list = []
    dict = {}

    for filename in os.listdir(path):
        dir = "dataset/" + filename

        I = cv2.imread(dir)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

        hist = myhist3(I, n_bins)
        hist = hist.reshape(-1)

        listH.append(hist)

    image = cv2.imread("dataset/object_05_4.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    histogram = myhist3(image, n_bins)
    histogram = histogram.reshape(-1)

    for i in range(0, len(listH)):

        distanceH = compare_histograms(histogram, listH[i], "hellinger")
        distance_list.append(distanceH)
        dict.update({(i + 1): distanceH})

    dictU = dict
    dict = sorted(dict.items(), key=lambda x: x[1], reverse=False)

    temp = list(dict)[0:6]
    points = [temp[0][0]]

    plt.subplot(2, 6, 1)
    plt.axis("off")
    plt.imshow(image)

    imageHist = listH[0]
    plt.subplot(2, 6, 7)
    plt.bar(range(0, 512), imageHist, width=3)

    imageNames = os.listdir(path)

    for i in range(1, 6):
        ixIm = temp[i][0]
        points.append(ixIm)

        filename = imageNames[ixIm - 1]
        imageHist = listH[ixIm - 1]

        dir = "dataset/" + filename

        img = cv2.imread(dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 6, i + 1)
        plt.axis("off")
        plt.imshow(img)

        plt.subplot(2, 6, 7 + i)
        plt.bar(range(0, 512), imageHist, width=3)

    #plt.subplots_adjust(wspace=0.6)
    plt.show()

    dictS = []

    for i in range(0, 120):
        dictS.append(dict[i][1])

    x0 = list(range(1, 121))
    y0 = list(dictU.values())

    plt.plot(x0, y0)
    plt.show()

    x1 = list(range(1, 121))
    y1 = dictS

    close = np.partition(y1, 5)
    points = close[0:5]

    plt.plot(x1, y1)
    plt.scatter(x1[0:5], points, marker='o')
    plt.show()


def second():

    # b)
    (signal, kernel, C) = simple_convolution()
    plot_convolution(signal, kernel, C)
    # c)
    (signal, kernel, C) = simple_convolution_with_edge()
    plot_convolution(signal, kernel, C)

    # d)
    sigma = [0.5, 1, 2, 3, 4]
    color = ['b', 'y', 'g', 'r', 'm']
    label = ['sigma = 0.5', 'sigma = 1', 'sigma = 2', 'sigma = 3', 'sigma = 4']

    for i in range(0, 5):
        size = int(2 * 3 * sigma[i] + 1)

        if size % 2 == 0:
            size += 1

        (x, gauss) = gaussian_kernel(size, sigma[i])
        plt.plot(x, gauss, color=color[i], label=label[i])

    plt.axis([-15, 15, 0, 0.8])
    plt.legend()
    plt.show()

    # e)
    test_associativity()

def simple_convolution():

    signal = read_data("signal.txt")
    kernel = read_data("kernel.txt")

    N = int((len(kernel) - 1) / 2)
    #new_signal = signal[N:]
    #new_signal = new_signal[:(len(new_signal) - N)]

    reversed = kernel[::-1].copy()

    numb_of_steps = len(signal) - len(kernel) + 1
    C = np.zeros(numb_of_steps)
    C_2 = np.zeros(len(signal))

    #C = np.zeros(len(signal) - 2*N - len(kernel))

    for i in range(0, numb_of_steps):
        C[i] = np.dot(signal[i: i + len(kernel)], reversed)
        C_2[i + N] = np.dot(signal[i: i + len(kernel)], reversed)

    #C = np.convolve(new_signal, kernel)

    #comparison = cv2.filter2D(signal, -1, kernel)
    #plt.plot(comparison)
    #plt.show()

    return (signal, kernel, C_2)

    # Can you recognize the shape of the kernel?
    # The kernel looks like a Gaussian kernel.
    # What is the sum of the elements in the kernel?
    # The sum is very close to 1. np.sum(kernel) = 0.9999999974
    # How does the kernel affect the signal?

def simple_convolution_with_edge():

    signal = read_data("signal.txt")
    kernel = read_data("kernel.txt")

    N = int((len(kernel) - 1) / 2)

    new_signal = np.pad(signal, (N, N), constant_values=(0, 0))

    reversed = kernel[::-1].copy()

    numb_of_steps = len(new_signal) - len(kernel) + 1
    C = np.zeros(numb_of_steps)

    for i in range(numb_of_steps):
        C[i] = np.dot(new_signal[i: i + len(kernel)], reversed)

    #C = np.convolve(new_signal, kernel, mode='valid')

    return (signal, kernel, C)

def plot_convolution(signal, kernel, C):

    plt.plot(signal, color='b', label='original')
    plt.plot(kernel, color='g', label='kernel')
    plt.plot(C, color='r', label='result')
    plt.axis([0, 40, 0, 6])
    plt.legend()
    plt.show()

def gaussian_kernel(size, sigma):

    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)

    gauss_kernel = (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(x**2 / (-2 * sigma**2))
    gauss_kernel /= np.sum(gauss_kernel)

    return (x, gauss_kernel)

def test_associativity():

    signal = read_data("signal.txt")
    #plt.plot(signal)
    #plt.show()

    sigma = 2
    size_k1 = (2 * 3 * sigma) + 1

    (x, k1) = gaussian_kernel(size_k1, sigma)
    k2 = [0.1, 0.6, 0.4]
    rev_k2 = k2[::-1].copy()

    # (s * k1) * k2
    N = int((len(k1) - 1) / 2)
    new_signal = np.pad(signal, (N, N), constant_values=(0, 0))
    numb_of_steps = len(new_signal) - len(k1) + 1
    C = np.zeros(numb_of_steps)

    for i in range(numb_of_steps):
        C[i] = np.dot(new_signal[i: i + len(k1)], k1)

    N = int((len(k2) - 1) / 2)
    C = np.pad(C, (N, N), constant_values=(0, 0))
    numb_of_steps = len(C) - len(k2) + 1
    C_1 = np.zeros(numb_of_steps)

    for i in range(numb_of_steps):
        C_1[i] = np.dot(C[i: i + len(k2)], rev_k2)

    # (s * k2) * k1
    N = int((len(k2) - 1) / 2)
    new_signal = np.pad(signal, (N, N), constant_values=(0, 0))
    numb_of_steps = len(new_signal) - len(k2) + 1
    C_4 = np.zeros(numb_of_steps)

    for i in range(numb_of_steps):
        C_4[i] = np.dot(new_signal[i: i + len(k2)], rev_k2)

    N = int((len(k1) - 1) / 2)
    C_4 = np.pad(C_4, (N, N), constant_values=(0, 0))
    numb_of_steps = len(C_4) - len(k1) + 1
    C_5 = np.zeros(numb_of_steps)

    for i in range(numb_of_steps):
        C_5[i] = np.dot(C_4[i: i + len(k1)], k1)

    # s * (k1 * k2)
    N = int((len(k2) - 1) / 2)
    new_k1 = np.pad(k1, (N, N), constant_values=(0, 0))
    numb_of_steps = len(new_k1) - len(k2) + 1
    C_2 = np.zeros(numb_of_steps)

    for i in range(numb_of_steps):
        C_2[i] = np.dot(new_k1[i: i + len(k2)], rev_k2)

    N = int((len(C_2) - 1) / 2)
    new_sig = np.pad(signal, (N, N), constant_values=(0, 0))
    numb_of_steps = len(new_sig) - len(C_2) + 1
    C_3 = np.zeros(numb_of_steps)
    rev_C2 = C_2[::-1].copy()

    for i in range(numb_of_steps):
        C_3[i] = np.dot(new_sig[i: i + len(C_2)], rev_C2)

    plt.plot(C_1)
    plt.axis([0, 40, 0, 2])
    plt.title("(s * k1) * k2")
    plt.show()
    plt.plot(C_5)
    plt.axis([0, 40, 0, 2])
    plt.title("(s * k2) * k1")
    plt.show()
    plt.plot(C_3)
    plt.axis([0, 40, 0, 2])
    plt.title("s * (k1 * k2)")
    plt.show()

def third():

    # a)
    I = cv2.imread('images/lena.png')
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    I = gauss_noise(I, 150)

    J = cv2.imread('images/lena.png')
    J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)
    J = sp_noise(J, 0.1)

    (gI, filteredI) = gaussfilter(I, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(filteredI, cmap="gray")
    plt.show()

    (gJ, filteredJ) = gaussfilter(J, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(J, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(filteredJ, cmap="gray")
    plt.show()

    # Which noise is better removed using the Gaussian filter?
    # Gaussian noise.

    # b)
    M = cv2.imread('images/museum.jpg')
    M = cv2.cvtColor(M, cv2.COLOR_RGB2GRAY)
    image_sharpening(M)

    # c)
    signal = read_data("signal.txt")
    filter_width = 3
    simple_median(signal, filter_width)

    # d)
    L_1 = cv2.imread('images/lena.png')
    L_1 = cv2.cvtColor(L_1, cv2.COLOR_RGB2GRAY)
    L_1 = gauss_noise(L_1, 50)
    median_2D(L_1)
    L_2 = cv2.imread('images/lena.png')
    L_2 = cv2.cvtColor(L_2, cv2.COLOR_RGB2GRAY)
    L_2 = sp_noise(L_2, 0.1)
    median_2D(L_2)

    # e)
    cat1 = cv2.imread('images/cat1.jpg')
    cat1 = cv2.cvtColor(cat1, cv2.COLOR_BGR2RGB)
    cat2 = cv2.imread('images/cat2.jpg')
    cat2 = cv2.cvtColor(cat2, cv2.COLOR_BGR2RGB)
    hybrid_image(cat1, cat2)


def gaussfilter(image, sigma):

    #plt.subplot(1, 2, 1)
    #plt.imshow(image, cmap="gray")

    #sigma = 2
    size = (2 * 3 * sigma) + 1
    (x, gauss) = gaussian_kernel(size, sigma)

    filtered = cv2.filter2D(image, -1, gauss)
    gaussT = gauss.T
    filtered = cv2.filter2D(filtered, -1, gaussT)

    #plt.subplot(1, 2, 2)
    #plt.imshow(filtered, cmap="gray")
    #plt.show()

    return (gauss, filtered)

def image_sharpening(image):

    kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - ((1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    sharpened = cv2.filter2D(image, -1, kernel)

    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(image, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Sharpened image")
    plt.imshow(sharpened, cmap="gray")
    plt.show()

def simple_median(I, w):

    corrupted = I.copy()
    I[10] = 0
    I[19] = 1

    sigma = 1.2
    size = int((2 * 3 * sigma) + 1)

    if size % 2 == 0:
        size += 1

    (x, g) = gaussian_kernel(size, sigma)
    gauss = cv2.filter2D(corrupted, -1, g)

    median_filter = np.zeros(w)
    res_length = len(corrupted) - w - 1
    res = np.zeros(res_length)

    for i in range(0, len(corrupted) - w - 1):
        median_filter = corrupted[i: i + 3].copy()

        median_filter.sort()
        mid = int(len(median_filter) / 2)

        res[i] = median_filter[mid]

    median = cv2.filter2D(corrupted, -1, res)

    plt.plot(I)
    plt.axis([0, 40, 0, 6])
    plt.title("Input signal")
    plt.show()
    plt.plot(corrupted)
    plt.title("Corrupted signal")
    plt.axis([0, 40, 0, 6])
    plt.show()
    plt.plot(gauss)
    plt.title("Gauss")
    plt.axis([0, 40, 0, 6])
    plt.show()
    plt.plot(res)
    plt.axis([0, 40, 0, 6])
    plt.title("Median")
    plt.show()

    # Which filter performs better at this specific task?
    # Median filter works better with salt and pepper noise.
    # In comparison to Gaussian filter that can be applied multiple times in any order,
    # does the order matter in case of median filter?
    # The order does matter when it comes to median filter.
    # What is the name of filters like this?
    # Nonlinear filters.

def median_2D(image):

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")

    (g, image_G) = gaussfilter(image, 2)
    plt.subplot(1, 3, 2)
    plt.imshow(image_G, cmap="gray")
    plt.title("Gauss filtered")

    w = 5
    h = 5

    N = int(w / 2)
    img = np.pad(image, ((N, N), (N, N)), mode="constant")

    filtered = np.zeros((image.shape[0], image.shape[1]))
    mid = int((w * h) / 2)

    for i in range(0, img.shape[0] - w - 1):
        for j in range(0, img.shape[1] - h - 1):
            window = np.zeros(w * h)
            ix = 0
            for x in range(0, w):
                for y in range(0, h):
                    window[ix] = img[i + x][j + y].copy()
                    ix += 1

            window.sort()
            filtered[i][j] = window[mid]

    filtered = np.copy(filtered.astype(np.uint8))

    plt.subplot(1, 3, 3)
    plt.imshow(filtered, cmap="gray")
    plt.title("Median filtered")
    plt.show()

def hybrid_image(img1, img2):

    sigma = 15
    (g, img1) = gaussfilter(img1, sigma)

    size = (2 * 3 * sigma) + 1
    unit_impulse = np.zeros(size)
    mid = int(size / 2)
    unit_impulse[mid] = 1

    laplace = np.zeros(size)
    laplace = unit_impulse - g

    filtered = cv2.filter2D(img2, -1, laplace)
    laplaceT = laplace.T
    filtered = cv2.filter2D(filtered, -1, laplaceT)

    img1_f = np.copy(img1.astype(np.float64))
    filtered_f = np.copy(filtered.astype(np.float64))

    newImg = img1_f + filtered_f
    newImg = (newImg - np.min(newImg)) / np.ptp(newImg)

    plt.imshow(newImg)
    plt.title("Hybrid image")
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

