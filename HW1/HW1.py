import cv2 as cv
import numpy as np

if __name__ == "__main__":
    cv.setUseOptimized(True)
    # cv.setNumThreads(1)

    print("Optimized: " + str(cv.useOptimized()))
    img = cv.imread('../test/AtoZ_invert.png', 0)

    k = 10
    kernel = np.ones((k, k), np.uint8)

    # erosion = cv.erode(img, kernel, iterations=1)
    # dilation = cv.dilate(img, kernel, iterations=1)
    erosionTime = 0
    dilationTime = 0

    for i in range(3, 120, 20):
        erosionTime = 0
        dilationTime = 0
        count = 0
        kernel = np.ones((i, i), np.uint8)

        for j in range(0, 1000):
            T1 = cv.getTickCount()
            erosion = cv.erode(img, kernel, iterations=1)
            T2 = cv.getTickCount()
            dilation = cv.dilate(img, kernel, iterations=1)
            T3 = cv.getTickCount()

            erosionTime += ((T2 - T1) / cv.getTickFrequency())
            dilationTime += ((T3 - T2) / cv.getTickFrequency())

        print(f"====================================================\n")
        print(f"Kernel size: {i}x{i}, Repeat: {j + 1} times")
        print("Average Erosion time:  %s ms " % ((erosionTime / j) * 1000))
        print("Average Dilation time: %s ms \n" % ((dilationTime / j) * 1000))

    # cv.imshow('Original', img)
    # cv.imshow('Erosion', erosion)
    # cv.imshow('Dilation', dilation)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
