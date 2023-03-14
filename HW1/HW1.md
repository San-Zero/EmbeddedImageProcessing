# HW1

在x86 PC上的執行morphalogy的範例
比對:
- 1. kernel大小在無AVX的情況下對於效能的影響
Ans: kernel越大，運算需要的時間越多
- 2. 有AVX和無AVX的效能差異
Ans: 發現開啟AVX後，運算的時間比關閉AVX還多

## Code
```python
# HW1.py
import cv2 as cv
import numpy as np
import time

if __name__ == "__main__":
    cv.setUseOptimized(False)
    # cv.setNumThreads(1)

    print("Optimized: " + str(cv.useOptimized()))
    img = cv.imread('../test/AtoZ_invert.png', 0)

    k = 10
    kernel = np.ones((k, k), np.uint8)

    # erosion = cv.erode(img, kernel, iterations=1)
    # dilation = cv.dilate(img, kernel, iterations=1)
    erosionTime = 0
    dilationTime = 0

    for i in range(3, 30, 2):
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

```
    
## 關閉AVX
```
Optimized: False
====================================================

Kernel size: 3x3, Repeat: 1000 times
Average Erosion time:  0.06668068068068075 ms 
Average Dilation time: 0.06577387387387387 ms 

====================================================

Kernel size: 5x5, Repeat: 1000 times
Average Erosion time:  0.08001601601601607 ms 
Average Dilation time: 0.08085625625625632 ms 

====================================================

Kernel size: 7x7, Repeat: 1000 times
Average Erosion time:  0.09199059059059059 ms 
Average Dilation time: 0.09371171171171162 ms 

====================================================

Kernel size: 9x9, Repeat: 1000 times
Average Erosion time:  0.0897820820820821 ms 
Average Dilation time: 0.09131591591591587 ms 

====================================================

Kernel size: 11x11, Repeat: 1000 times
Average Erosion time:  0.0887703703703704 ms 
Average Dilation time: 0.09416606606606602 ms 

====================================================

Kernel size: 13x13, Repeat: 1000 times
Average Erosion time:  0.12876186186186186 ms 
Average Dilation time: 0.13376806806806793 ms 

====================================================

Kernel size: 15x15, Repeat: 1000 times
Average Erosion time:  0.13952072072072058 ms 
Average Dilation time: 0.1435651651651651 ms 

====================================================

Kernel size: 17x17, Repeat: 1000 times
Average Erosion time:  0.14343253253253258 ms 
Average Dilation time: 0.15092152152152155 ms 

====================================================

Kernel size: 19x19, Repeat: 1000 times
Average Erosion time:  0.14340290290290292 ms 
Average Dilation time: 0.15235195195195234 ms 

====================================================

Kernel size: 21x21, Repeat: 1000 times
Average Erosion time:  0.14371171171171168 ms 
Average Dilation time: 0.15658088088088087 ms 

====================================================

Kernel size: 23x23, Repeat: 1000 times
Average Erosion time:  0.16225285285285293 ms 
Average Dilation time: 0.17549019019019027 ms 

====================================================

Kernel size: 25x25, Repeat: 1000 times
Average Erosion time:  0.17369649649649668 ms 
Average Dilation time: 0.18523333333333306 ms 

====================================================

Kernel size: 27x27, Repeat: 1000 times
Average Erosion time:  0.18999189189189233 ms 
Average Dilation time: 0.1998548548548549 ms 

====================================================

Kernel size: 29x29, Repeat: 1000 times
Average Erosion time:  0.19823253253253265 ms 
Average Dilation time: 0.2134692692692694 ms 

```

## 開啟AVX
```
Optimized: True
====================================================

Kernel size: 3x3, Repeat: 1000 times
Average Erosion time:  0.059294194194194245 ms 
Average Dilation time: 0.05999949949949952 ms 

====================================================

Kernel size: 5x5, Repeat: 1000 times
Average Erosion time:  0.07402962962962957 ms 
Average Dilation time: 0.07444174174174158 ms 

====================================================

Kernel size: 7x7, Repeat: 1000 times
Average Erosion time:  0.08659759759759761 ms 
Average Dilation time: 0.08542452452452447 ms 

====================================================

Kernel size: 9x9, Repeat: 1000 times
Average Erosion time:  0.10523583583583587 ms 
Average Dilation time: 0.10591171171171158 ms 

====================================================

Kernel size: 11x11, Repeat: 1000 times
Average Erosion time:  0.16509229229229258 ms 
Average Dilation time: 0.16416596596596617 ms 

====================================================

Kernel size: 13x13, Repeat: 1000 times
Average Erosion time:  0.17903473473473472 ms 
Average Dilation time: 0.17957647647647654 ms 

====================================================

Kernel size: 15x15, Repeat: 1000 times
Average Erosion time:  0.187602202202202 ms 
Average Dilation time: 0.18613273273273276 ms 

====================================================

Kernel size: 17x17, Repeat: 1000 times
Average Erosion time:  0.19687567567567574 ms 
Average Dilation time: 0.19891691691691687 ms 

====================================================

Kernel size: 19x19, Repeat: 1000 times
Average Erosion time:  0.19994804804804814 ms 
Average Dilation time: 0.2001218218218219 ms 

====================================================

Kernel size: 21x21, Repeat: 1000 times
Average Erosion time:  0.20471101101101088 ms 
Average Dilation time: 0.2045893893893892 ms 

====================================================

Kernel size: 23x23, Repeat: 1000 times
Average Erosion time:  0.25090260260260283 ms 
Average Dilation time: 0.2527126126126128 ms 

====================================================

Kernel size: 25x25, Repeat: 1000 times
Average Erosion time:  0.29210830830830825 ms 
Average Dilation time: 0.2928415415415413 ms 

====================================================

Kernel size: 27x27, Repeat: 1000 times
Average Erosion time:  0.29339339339339415 ms 
Average Dilation time: 0.2927599599599593 ms 

====================================================

Kernel size: 29x29, Repeat: 1000 times
Average Erosion time:  0.27810390390390366 ms 
Average Dilation time: 0.2791626626626622 ms 
 
```