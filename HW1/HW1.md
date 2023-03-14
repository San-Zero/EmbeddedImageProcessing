# HW1

在x86 PC上的執行morphalogy的範例
比對:
1. kernel大小在無AVX的情況下對於效能的影響
    >kernel越大，運算需要的時間越多
2. 有AVX和無AVX的效能差異
    >發現開啟AVX後，運算的時間比關閉AVX還多

## Code
```python
# HW1.py
import cv2 as cv
import numpy as np
import time

if __name__ == "__main__":
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
```
    
## 關閉AVX
```
Optimized: False
====================================================

Kernel size: 3x3, Repeat: 1000 times
Average Erosion time:  0.06999239239239233 ms 
Average Dilation time: 0.06709519519519516 ms 

====================================================

Kernel size: 23x23, Repeat: 1000 times
Average Erosion time:  0.19022632632632616 ms 
Average Dilation time: 0.2022 ms 

====================================================

Kernel size: 43x43, Repeat: 1000 times
Average Erosion time:  0.341203103103103 ms 
Average Dilation time: 0.35394504504504487 ms 

====================================================

Kernel size: 63x63, Repeat: 1000 times
Average Erosion time:  0.46508158158158247 ms 
Average Dilation time: 0.4677838838838835 ms 

====================================================

Kernel size: 83x83, Repeat: 1000 times
Average Erosion time:  0.7074678678678674 ms 
Average Dilation time: 0.7004507507507503 ms 

====================================================

Kernel size: 103x103, Repeat: 1000 times
Average Erosion time:  0.8810900900900892 ms 
Average Dilation time: 0.8830812812812816 ms 
```

## 開啟AVX
```
Optimized: True
====================================================

Kernel size: 3x3, Repeat: 1000 times
Average Erosion time:  0.07148168168168169 ms 
Average Dilation time: 0.07104434434434433 ms 

====================================================

Kernel size: 23x23, Repeat: 1000 times
Average Erosion time:  0.2934177177177177 ms 
Average Dilation time: 0.29765505505505496 ms 

====================================================

Kernel size: 43x43, Repeat: 1000 times
Average Erosion time:  0.481787787787788 ms 
Average Dilation time: 0.48253373373373337 ms 

====================================================

Kernel size: 63x63, Repeat: 1000 times
Average Erosion time:  0.6419330330330325 ms 
Average Dilation time: 0.6304997997998003 ms 

====================================================

Kernel size: 83x83, Repeat: 1000 times
Average Erosion time:  0.8882032032032022 ms 
Average Dilation time: 0.8709841841841838 ms 

====================================================

Kernel size: 103x103, Repeat: 1000 times
Average Erosion time:  1.0836530530530515 ms 
Average Dilation time: 1.057748648648649 ms 
```