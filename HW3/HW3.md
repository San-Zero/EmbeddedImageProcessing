# HW3

## LBP抓馬路區域

將圖片切割成數個區塊，在畫面最下面選出一個區塊做基準，
透過BFS走訪整張圖片，前一個區塊與後一個區塊，若兩者相似，將後一個區塊記錄起來，並把基準換做下一塊，待後續比較使用
最後將記錄好的所有相似的區塊還原成圖片

![image](./lbp_result.png)
![image](./lbp_result2.png)

## 將lbp當作watershed的marker
![image](./watershed_result1.png)
![image](./watershed_result2.png)

