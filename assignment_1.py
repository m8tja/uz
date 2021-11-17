import numpy as np
import cv2
from matplotlib import pyplot as plt

def first():

    # a)
    I = cv2.imread('images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    plt.imshow(I)
    plt.title("Original image")
    plt.show()

    # b)
    I_float = np.copy(I.astype(np.float64))
    red = I_float[:, :, 0]
    green = I_float[:, :, 1]
    blue = I_float[:, :, 2]
    gray = (red + green + blue) / 3
    plt.imshow(gray, cmap='gray')
    plt.title("Image converted into grayscale")
    plt.show()

    # c)
    cutout = I[130:260, 240:450, 1]
    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(I)
    plt.subplot(1, 2, 2)
    plt.imshow(cutout, cmap='gray')
    plt.title("A section of the image in grayscale")
    plt.show()

    # d)
    I[125:250, 220:420, :] = cv2.bitwise_not(I[125:250, 220:420, :])
    #I[125:250, 220:420, :] = 255 - I[125:250, 220:420, :]
    plt.imshow(I)
    plt.title("Image with a rectangular section inverted")
    plt.show()

    # How is inverting a grayscale value defined for uint8?
    # Each pixel value is subtracted from the max pixel value, which is 255 for uint8.

    # e)
    J = cv2.imread('images/umbrellas.jpg')
    J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)
    plt.imshow(J, cmap='gray')
    plt.title("Image in grayscale")
    plt.show()
    J_float = np.copy(J.astype(np.float64))
    J_float /= 255
    J_float *= 63
    plt.imshow(J_float, vmax=255, cmap='gray')
    plt.title("Rescaled image")
    plt.show()

def second():

    # a)
    I = cv2.imread('images/bird.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    #plt.imshow(I)
    #plt.show()
    I_float = np.copy(I.astype(np.float64))
    red = I_float[:, :, 0]
    green = I_float[:, :, 1]
    blue = I_float[:, :, 2]
    gray = (red + green + blue) / 3
    gray2 = (red + green + blue) / 3

    threshold = 55

    gray[gray < threshold] = 0
    gray[gray >= threshold] = 1
    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(I)
    plt.subplot(1, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Binary mask")
    plt.show()

    mask = np.where(gray2 < threshold, 0, 1)
    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(I)
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Binary mask using np.where()")
    plt.show()

    # b)
    J = cv2.imread('images/bird.jpg')
    J = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)

    #numbBins = input("Input number of bins: ")
    #numbBins = int(numbBins)
    numbBins = 20
    histogram = myhist(J, numbBins)
    plt.bar(range(numbBins), histogram)
    plt.title("Histogram made with myhist() function on original image with 20 bins")
    plt.show()

    numbBins = 100
    histogram = myhist(J, numbBins)
    plt.bar(range(numbBins), histogram)
    plt.title("Histogram made with myhist() function on original image with 100 bins")
    plt.show()

    # c)
    M = cv2.imread('images/bird.jpg')
    M = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    M[M < 10] = 10
    M[M > 200] = 200

    histogram_2 = myhist(M, numbBins)
    plt.bar(range(numbBins), histogram_2)
    plt.title("Histogram made with myhist() function")
    plt.show()

    histogram_3 = myhist_2(M, numbBins)
    plt.bar(range(numbBins), histogram_3)
    plt.title("Histogram made with altered myhist_2() function")
    plt.show()

    # d)
    test_bins = 100
    U = cv2.imread('images/toy_1.jpg')
    V = cv2.imread('images/toy_2.jpg')
    W = cv2.imread('images/toy_3.jpg')
    histU = myhist(U, test_bins)
    plt.bar(range(test_bins), histU)
    plt.title("Histogram of the lightest image")
    plt.show()
    histV = myhist(V, test_bins)
    plt.bar(range(test_bins), histV)
    plt.title("Histogram of the second lightest image")
    plt.show()
    histW = myhist(W, test_bins)
    plt.bar(range(test_bins), histW)
    plt.title("Histogram of the darkest image")
    plt.show()

    # In the histogram of the lightest image we can see that the intensity of the majority of the pixels
    # is in the mid range and leaning towards the right side of the graph, meaning there are indeed more
    # light pixels and very few dark ones.
    # The other two images have similar histograms, which shows that the lightness of the images was pretty close.
    # In the histogram of the mid range image we can see it still has some lighter pixels, whereas in the histogram
    # of the darkest image we can see there are no high intensity pixels in the image.

    # e)
    birdT = otsu(J)
    maskB = np.where(J < birdT, 0, 1)
    plt.imshow(maskB, cmap="gray")
    plt.title("Binary mask using Otsu's method")
    plt.show()

    E = cv2.imread('images/eagle.jpg')
    E = cv2.cvtColor(E, cv2.COLOR_BGR2GRAY)
    eagleT = otsu(E)
    maskE = np.where(E < eagleT, 0, 1)
    plt.imshow(maskE, cmap="gray")
    plt.title("Binary mask using Otsu's method")
    plt.show()

def myhist(imageG, numbBins):

    imageG = imageG.reshape(-1)
    H = np.zeros(numbBins)
    bin_size = (255/numbBins)

    for i in imageG:
        index = int(i / bin_size)
        if index >= numbBins:
            index = numbBins - 1

        H[index] += 1

    sum = np.sum(H)

    H /= sum

    return H

def myhist_2(imageG, numbBins):

    imageG = imageG.reshape(-1)
    H = np.zeros(numbBins)
    min = np.amin(imageG)
    max = np.amax(imageG)

    valueForBins = max - min

    bin_size = (valueForBins / numbBins)

    for i in imageG:
        index = int(i / bin_size)
        if index >= numbBins:
            index = numbBins - 1

        H[index] += 1

    sum = np.sum(H)

    H /= sum

    return H

def otsu(imageG):

    numbBins = 256
    hist = myhist(imageG, numbBins)
    top = numbBins
    max = 0
    threshold = 0

    for i in range(1, top):
        if(not(np.sum(hist[:i]) == 0 or np.sum(hist[i:]) == 0)):
            weight0 = np.sum(hist[:i])
            weight1 = np.sum(hist[i:])
            mean0 = np.sum(np.arange(i) * hist[:i]) / weight0
            mean1 = np.sum(np.arange(i, top) * hist[i:]) /weight1

            variance = weight0 * weight1 * ((mean0 - mean1)**2)

            if(variance > max):
                threshold = i
                max = variance

    #mask = np.where(imageG < threshold, 0, 1)
    #plt.imshow(mask, cmap="gray")
    #plt.show()

    return threshold

def third():

    # a)
    I = cv2.imread('images/mask.png')
    plt.imshow(I)
    plt.title("Original image")
    plt.show()

    n = input("Input size of structuring element: ")
    n = int(n)
    #n = 5
    SE = np.ones((n, n), np.uint8)
    I_eroded = cv2.erode(I, SE)
    I_dilated = cv2.dilate(I, SE)
    plt.imshow(I_eroded)
    plt.title("Erosion")
    plt.show()
    plt.imshow(I_dilated)
    plt.title("Dilation")
    plt.show()

    I_open = cv2.morphologyEx(I, cv2.MORPH_OPEN, SE)
    I_openM = cv2.dilate(I_eroded, SE)
    I_close = cv2.morphologyEx(I, cv2.MORPH_CLOSE, SE)
    I_closeM = cv2.erode(I_dilated, SE)
    plt.imshow(I_open)
    plt.title("Opening using built-in function")
    plt.show()
    plt.imshow(I_openM)
    plt.title("Erosion followed by dilation")
    plt.show()
    plt.imshow(I_close)
    plt.title("Closing using built-in function")
    plt.show()
    plt.imshow(I_closeM)
    plt.title("Dilation followed by erosion")
    plt.show()

    # Based on the results, which order of erosion and dilation operations produces opening and which closing?
    # Opening is achieved by first applying erosion then dilation.
    # Closing is achieved by first applying dilation then erosion.

    # b)
    I = cv2.imread('images/bird.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    plt.imshow(I)
    plt.title("Original image")
    plt.show()
    I_float = np.copy(I.astype(np.float64))
    red = I_float[:, :, 0]
    green = I_float[:, :, 1]
    blue = I_float[:, :, 2]
    gray = (red + green + blue) / 3

    threshold = 55
    mask = np.where(gray < threshold, 0, 1)
    mask = mask.astype(np.uint8)
    #plt.imshow(mask, cmap='gray')
    #plt.show()
    #print(mask.dtype)

    n = 22
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    mask_c = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, SE)
    #n = 2
    #SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    #mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_OPEN, SE)
    plt.imshow(mask_c, cmap='gray')
    plt.title("Cleaned up mask using morphological operations")
    plt.show()

    # c)
    immask(I, mask_c)

    # d)
    E = cv2.imread('images/eagle.jpg')
    E = cv2.cvtColor(E, cv2.COLOR_BGR2GRAY)
    eagleT = otsu(E)
    maskE = np.where(E < eagleT, 0, 1)
    plt.imshow(maskE, cmap="gray")
    plt.title("Binary mask using Otsu's method")
    plt.show()

    E_2 = cv2.imread('images/eagle.jpg')
    E_2 = cv2.cvtColor(E_2, cv2.COLOR_BGR2RGB)
    immask(E_2, maskE)

    F = cv2.imread('images/eagle.jpg')
    F = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
    F = 255 - F
    eagleF = otsu(F)
    maskF = np.where(F < eagleF, 0, 1)
    plt.imshow(maskF, cmap="gray")
    plt.title("Binary mask with Otsu's method on the inverted image")
    plt.show()

    # Why is the background included in the mask and not the object? How would you fix that in general?
    # The background is included because in the original image its intensity is lower than the object's intensity.
    # We can invert the image, then object will be included in the mask and the background won't be.

    # e)
    C = cv2.imread('images/coins.jpg')
    C = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY)
    C = 255 - C
    coinsT = otsu(C)
    maskC = np.where(C < coinsT, 0, 1)
    maskC = maskC.astype(np.uint8)
    n = 10
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    maskC = cv2.morphologyEx(maskC, cv2.MORPH_CLOSE, SE)
    #plt.imshow(maskC, cmap="gray")
    #plt.show()

    stats = cv2.connectedComponentsWithStats(maskC)

    rows = stats[0]
    labels = stats[1]
    arr = stats[2]

    for i in range(1, rows):
        if arr[i][4] >= 700:
            tempM = (labels == i).astype("uint8") * 255
            tempM = 255 - tempM
            maskC = maskC * tempM

    C_2 = cv2.imread('images/coins.jpg')
    C_2 = cv2.cvtColor(C_2, cv2.COLOR_BGR2RGB)
    immask(C_2, maskC)

def immask(image, mask):

    I_float = np.copy(image.astype(np.float64))
    red = I_float[:, :, 0]
    green = I_float[:, :, 1]
    blue = I_float[:, :, 2]
    gray = (red + green + blue) / 3

    rgb = gray * mask
    rgb = np.dstack((red, green, blue))
    rgb = 255 - rgb
    rgb[mask == 0] = 0

    plt.imshow((rgb * 255).astype(np.uint8))
    plt.title("Image processed with immask() function")
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

if __name__ == "__main__":
    main()
