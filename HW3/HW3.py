import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_blocks(img: np.ndarray, block_size: int) -> np.ndarray:
    # 獲取圖像大小
    height, width = img.shape

    # 定義區塊大小
    # block_size = 32
    # 計算區塊的行數和列數
    n_blocks_height = int(np.ceil(height / block_size))
    n_blocks_width = int(np.ceil(width / block_size))

    # 創建一個空的NumPy數組來保存所有區塊
    blocks = np.zeros((n_blocks_height, n_blocks_width, block_size, block_size), dtype=np.float32)

    # 將圖像分成區塊並存儲到NumPy數組中
    for i in range(n_blocks_height):
        for j in range(n_blocks_width):
            block = gray[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            blocks[i, j, :, :] = block

    return blocks


def blocks_to_hist(blocks: np.ndarray) -> np.ndarray:
    # 獲取區塊的行數和列數
    num_blocks_y, num_blocks_x, block_size, _ = blocks.shape

    # 創建一個空的NumPy數組來保存所有區塊的直方圖
    hist = np.zeros((num_blocks_y, num_blocks_x, 256), dtype=np.float32)

    # 計算每個區塊的直方圖
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            hist[y, x, :] = np.histogram(blocks[y, x, :, :], bins=256, range=(0, 256))[0]

    return hist


def blocksToImg(_blocks: np.ndarray) -> np.ndarray:
    _num_blocks_y, _num_blocks_x, _block_size, _ = _blocks.shape

    # 創建一個新的圖像來存儲所有區塊
    img = np.zeros((_num_blocks_y * _block_size, _num_blocks_x * _block_size), dtype=np.uint8)

    # 將每個區塊放回到新圖像中
    for y in range(_num_blocks_y):
        for x in range(_num_blocks_x):
            img[y * _block_size:(y + 1) * _block_size, x * _block_size:(x + 1) * _block_size] = _blocks[y, x, :, :]

    return img


def bfs(_x: int, _y: int, _hist: np.ndarray, similarity: float) -> list:
    _num_blocks_y, _num_blocks_x, _block_size, _ = _hist.shape
    queue = [(_x, _y)]
    _result = []
    visited = set()
    visited.add((_x, _y))

    while len(queue) > 0:
        currentVertex = queue.pop(0)

        for _i, _j in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # 搜尋該節點的上右下左區塊
            new_x, new_y = currentVertex[0] + _i, currentVertex[1] + _j
            if new_x < 0 or new_x >= _num_blocks_x or new_y < 0 or new_y >= _num_blocks_y:  # 如果超出範圍則跳過
                continue
            compare_result = cv2.compareHist(_hist[currentVertex[1], currentVertex[0]], _hist[new_y, new_x],
                                             cv2.HISTCMP_CORREL)
            if compare_result >= similarity:
                _result.append((new_x, new_y))
            if (new_x, new_y) not in visited:
                queue.append((new_x, new_y))
                visited.add((new_x, new_y))

    return _result


def getMaxIndex(list: list, num: int):
    return [i for i, x in enumerate(list) if x in sorted(list, reverse=True)[:num]]


def compareHist(_hist: np.ndarray):
    _num_blocks_y, _num_blocks_x, _, _ = _hist.shape
    _result = []

    # 獲取前幾多的值
    # list1 = getMaxIndex(_hist[_num_blocks_y - 1, 0].ravel(), 10)
    # list2 = getMaxIndex(_hist[_num_blocks_y - 1, 0].ravel(), 10)

    counter = 0
    wrong_amount = 2
    correct_amount = 0
    total_amount = 10

    for i in range(num_blocks_x):
        list1 = getMaxIndex(_hist[_num_blocks_y - 1, i].ravel(), total_amount)
        for j in range(num_blocks_x):
            if i == j:
                continue
            list2 = getMaxIndex(_hist[_num_blocks_y - 1, j].ravel(), total_amount)
            for k in range(total_amount):
                if list1[k] == list2[k]:
                    correct_amount += 1
            if wrong_amount <= total_amount - correct_amount:
                counter += 1
                correct_amount = 0
        _result.append(counter)
        counter = 0

    return _result


def display(img, cmap=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)
    plt.show()


if __name__ == '__main__':
    # 讀取圖像
    img = cv2.imread('imgs/road.jpg')

    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    # 將圖像轉換為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 將圖像分成數個區塊
    blocks = to_blocks(gray, 16)

    # 獲取區塊的行數和列數
    num_blocks_y, num_blocks_x, block_size, _ = blocks.shape
    # print(blocks.shape)

    # 創建一個空的NumPy數組來保存histogram
    hist = np.zeros((num_blocks_y, num_blocks_x, 256, 1), dtype=np.float32)

    # 計算histogram
    for i in range(0, num_blocks_y):
        for j in range(0, num_blocks_x):
            # blocks[i, j]是一個32x32的小圖，calcHist計算這張圖的histogram
            hist[i, j] = cv2.calcHist([blocks[i, j]], [0], None, [256], [0, 256])

    # 找出最後一排最適合當作基準的區塊，第一層for用來指定要比較的區塊，第二層則是和其他區塊比較
    # 將被比較與比較的區塊作排序後，找出前五個大的值，如果比較的次數 - 比較後正確的數量 <= 允許的誤差數量，則認為是相似的
    compare_results = []
    compare_counter = 0

    for i in range(0, num_blocks_x):
        for j in range(0, num_blocks_x):
            results = cv2.compareHist(hist[num_blocks_y - 1, i], hist[num_blocks_y - 1, j], cv2.HISTCMP_CORREL)
            if results >= 0.99:  # 0.99是閾值，可以自己調整
                compare_counter += 1

        compare_counter -= 1  # 減去自己本身
        compare_results.append(compare_counter)
        compare_counter = 0  # 重置計數器

    # 取出最大的當作基準的區塊，若有複數個則隨機選擇一個
    max_value = max(compare_results)
    max_index = [i for i, j in enumerate(compare_results) if j == max_value]
    max_index = random.choice(max_index)
    # print(num_blocks_y - 1, max_index)

    img2 = np.zeros((num_blocks_y * block_size, num_blocks_x * block_size, 3), dtype=np.uint8)

    # 將每個區塊放回到新圖像中
    # for y in range(num_blocks_y):
    #     for x in range(num_blocks_x):
    #         if x == max_index and y == num_blocks_y - 1:
    #             # 將這個區塊變成藍色
    #             img2[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size, 0] = 255
    #         else:
    #             pass
    #             # img2[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = blocks[y, x, :, :]

    # 將兩張影像重疊
    # result = cv2.add(img, img2)

    # cv2.imshow('img', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 藉由BFS走訪整張圖片，前一個與後一個比較，找出所有與相似的區塊
    result = bfs(max_index, num_blocks_y - 1, hist, similarity=0.91)
    # print(len(result))
    # print(result)

    # result裡有所有與基準區塊相似的區塊的座標，將result還原成圖像
    for i, j in result:
        img2[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size, 0] = 255

    print(img2.shape)
    print(img.shape)

    result = cv2.add(img, img2)
    cv2.imshow('lbp', result)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding unknown region
    sure_fg = np.uint8(img2[:, :, 0])
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(img2[:, :, 0])

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    display(markers)

    # finding contours on markers
    contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(img, contours, i, (255, 0, 0), 5)

    display(img)

    cv2.imshow('watershed', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

