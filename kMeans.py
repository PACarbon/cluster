from numpy import *
import requests
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#读取一个文本文件，解析制表符分隔的浮点数数据，返回一个列表
def loadDataSet(fileName):  # 通用函数，用于解析制表符分隔的浮点数
    dataarray = []  # 假设最后一列是目标值
    fr = open(fileName)  #打开指定的文件 fileName
    for line in fr.readlines():  #逐行读取文件内容
        curLine = line.strip().split('\t')  #去除每行的首尾空白字符，并按制表符 \t 分割
        fltLine = list(map(float, curLine))  # 将所有元素转换为浮点数
        dataarray.append(fltLine)  #将每个 fltLine 添加到 dataarray 中
    return dataarray

#计算两个向量之间的欧几里得距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # 计算欧几里得距离

#随机生成 k 个质心，初始质心的位置在数据集的最小值和最大值之间随机选择
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]  # 获取数据的维度
    centroids = np.zeros((k, n))  # 创建一个 k×n 的零矩阵作为初始质心
    for j in range(n):  # 创建随机质心
        minJ = np.min(dataSet[:, j])  # 获取列的最小值
        maxJ = np.max(dataSet[:, j])  # 获取列的最大值
        centroids[:, j] = minJ + (maxJ - minJ) * np.random.rand(k)  # 在范围内随机生成质心
    return centroids

#实现 K-means 聚类算法，返回最终的质心和簇分配结果
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):   #dataSet：要聚类的数据集  k：希望聚成的簇的数量  distMeas：距离计算函数  createCent：质心初始化
    m = np.shape(dataSet)[0]  # 获取数据的行数
    clusterAssment = np.zeros((m, 2))  # 创建一个 m×2 的矩阵用于存储每个点的簇分配结果和误差平方
    centroids = createCent(dataSet, k)  # 随机生成初始质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        #clusterChanged 控制整个 K-means 的迭代是否继续。只要有任何一个点被重新分配簇，就继续迭代
        for i in range(m):  # 遍历每个数据点
            minDist = np.inf  # 初始化最小距离为无穷大
            minIndex = -1  # 初始化最小距离对应的质心索引为 -1
            for j in range(k):   # 遍历所有质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])  # 计算点 i 到质心 j 的距离
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 如果点 i 被分配到新的簇，则更新 flag
                clusterChanged = True   # 设置标志，说明有变化，继续下一轮迭代
            clusterAssment[i, :] = minIndex, minDist ** 2  # 更新点 i 的簇分配结果和误差平方
        print(centroids)  # 打印当前的质心
        for cent in range(k):  # 更新质心的位置
            if len(dataSet[clusterAssment[:, 0] == cent]) > 0:   #布尔索引，用来选出属于第 cent 个簇的所有点
                centroids[cent, :] = np.mean(dataSet[clusterAssment[:, 0] == cent], axis=0)  # 计算质心的新位置
    return centroids, clusterAssment  #centroids：最终的质心坐标数组   clusterAssment：每个数据点的簇分配情况和对应误差平方

#实现二分 K-means 聚类算法，从一个簇开始，逐步分裂直到达到 k 个簇
def biKmeans(dataSet, k, distMeas=distEclud):  #dataSet: 原始数据集（二维数组） k: 最终要划分的簇个数  distMeas: 距离计算函数 默认是欧几里得距离 distEclud
    m = np.shape(dataSet)[0]  # 获取数据点的数量
    clusterAssment = np.zeros((m, 2))  # 初始化一个 m×2 的数组
    centroid0 = np.mean(dataSet, axis=0).tolist()   # 整个数据集的均值，作为初始质心
    centList = [centroid0]  # 创建一个列表，初始时包含一个质心
    for j in range(m):  # 计算初始误差
        clusterAssment[j, 1] = distMeas(np.array(centroid0), dataSet[j, :]) ** 2  #将所有数据点看作是一个大簇，记录它们到初始质心的误差平方
    while len(centList) < k:
        lowestSSE = np.inf   # 初始化最小误差和
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.where(clusterAssment[:, 0] == i)[0], :]  # 获取当前簇的所有点
            if len(ptsInCurrCluster) == 0:
                continue   # 如果当前簇是空的，跳过
            centroidarray, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = np.sum(splitClustAss[:, 1])  # 计算分裂后的误差
            sseNotSplit = np.sum(clusterAssment[np.where(clusterAssment[:, 0] != i)[0], 1])   # 其他簇的总误差
            print(f"sseSplit: {sseSplit}, sseNotSplit: {sseNotSplit}")
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i  # 当前最优的待拆分簇编号
                bestNewCents = centroidarray  # 二分后产生的两个质心
                bestClustAss = splitClustAss.copy()  # 二分后的簇分配结果
                lowestSSE = sseSplit + sseNotSplit   # 当前最小误差和
        bestClustAss[np.where(bestClustAss[:, 0] == 1)[0], 0] = len(centList)  # 更新簇索引
        bestClustAss[np.where(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit   # 保留原编号
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()   # 替换原来的质心
        centList.append(bestNewCents[1, :].tolist())  # 添加新质心
        clusterAssment[np.where(clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClustAss  # 更新簇分配结果和误差平方
    return np.array(centList), clusterAssment

import requests
from time import sleep

#从 OpenStreetMap 的 Nominatim 服务获取地理位置信息
def geoGrab(stAddress, city):  #stAddress: 街道地址  city: 城市名称
    address = f"{stAddress} {city}"
    url = "https://nominatim.openstreetmap.org/search"  # 正确的 URL
    params = {'q': address, 'format': 'json'}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        if data:
            return {'latitude': data[0]['lat'], 'longitude': data[0]['lon']}
    except requests.RequestException as e:
        print(f"请求失败: {e}")
    return None

#批量获取地址的经纬度信息，并将结果写入 places.txt 文件
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict:
            lat = float(retDict['latitude'])
            lng = float(retDict['longitude'])
            print(f"{lineArr[0]}\t{lat}\t{lng}")
            fw.write(f"{line}\t{lat}\t{lng}\n")
        else:
            print("error fetching")
        sleep(1)
    fw.close()

#使用球面余弦定理计算两个地理位置之间的距离
def distSLC(vecA, vecB):  # Spherical Law of Cosines
    latA = np.radians(vecA[1])
    latB = np.radians(vecB[1])
    lonA = np.radians(vecA[0])
    lonB = np.radians(vecB[0])
    a = np.sin(latA) * np.sin(latB)
    b = np.cos(latA) * np.cos(latB) * np.cos(lonB - lonA)
    return np.arccos(a + b) * 6371.0

#从 places.txt 文件中加载数据，执行二分 K-means 聚类，并在地图上绘制聚类结果
def clusterClubs(numClust=5):  #参数 numClust：希望聚成的簇（默认是5）
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[3]), float(lineArr[4])])  # 确保纬度和经度的顺序正确
    datarray = np.array(datList)
    myCentroids, clustAssing = biKmeans(datarray, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')  # 替换为你的图像文件路径
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datarray[np.where(clustAssing[:, 0] == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten(), ptsInCurrCluster[:, 1].flatten(), marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:, 0].flatten(), myCentroids[:, 1].flatten(), marker='+', s=300)
    plt.show()

    # 绘制K-means聚类结果
def plotKMeans(dataSet, centroids, clusterAssing):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    for i in range(centroids.shape[0]):
        ptsInCurrCluster = dataSet[clusterAssing[:, 0] == i]
        ax.scatter(ptsInCurrCluster[:, 0], ptsInCurrCluster[:, 1], marker=scatterMarkers[i % len(scatterMarkers)], s=90)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=300)
    plt.show()


    # 绘制二分K-means聚类结果
def plotBiKmeans2(dataSet, centroids, clusterAssing):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    for i in range(5):
        ptsInCurrCluster = dataSet[clusterAssing[:, 0] == i]
        ax.scatter(ptsInCurrCluster[:, 0], ptsInCurrCluster[:, 1], marker=scatterMarkers[i % len(scatterMarkers)], s=90)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=300)
    plt.show()




# 聚类并显示结果
dataSet = loadDataSet('testSet.txt')
dataSet = np.array(dataSet)  # 确保是 numpy 数组

# 可以调用 clusterClubs 函数来显示包含图像的聚类结果
#clusterClubs(numClust=5)


dataSet2 = loadDataSet('testSet2.txt')
dataSet2 = np.array(dataSet2)  # 确保是 numpy 数组

#clusterClubs(numClust=5)
centroids, clusterAssing = kMeans(dataSet, 3)
print("质心坐标:")
print(centroids)
print("簇分配结果:")
print(clusterAssing)
plotKMeans(dataSet, centroids, clusterAssing)
centroids2, clusterAssing2 = biKmeans(dataSet2, 4, distMeas=distEclud)
plotBiKmeans2(dataSet2, centroids2, clusterAssing2)

clusterClubs(numClust=5)