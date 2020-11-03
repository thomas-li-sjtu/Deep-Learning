from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class SOMnet(object):
    def __init__(self):      # 设计网络参数初始值
        self.lratemax = 0.8; # 最大学习率--欧式距离
        self.lratemin = 0.05 # 最小学习率--欧式距离
        self.rmax = 5.0      # 最大聚类半径--根据数据集
        self.rmin = 0.5      # 最小聚类半径--根据数据集
        self.Steps = 1000    # 迭代次数
        self.lratrlist = []  # 学习率收敛曲线
        self.rlist = []      # 学习半径收敛曲线
        self.w = []          # 权重向量组
        self.M = 2           # M*N 表示聚类总数
        self.N = 2           # M,N表示领域的参数
        self.dataMat = []    # 外部导入数据集
        self.classLabel = [] # 聚类后的类别标签
    """
    def loadDataSet(self,filename): # 加载数据文件
        numFeat = len(open(filename).readline().split('\t ')) -1
        fr = open(filename)
        for line in fr.readline():
            lineArr = []
            curLine = line.strip().split("\t")
            lineArr.append(float(curLine[0]))
            lineArr.append(float(curLine[1]))
            #print(shape(lineArr))
            self.dataMat.append(lineArr)
        self.dataMat = mat(self.dataMat)
    """
    def loadDataSet(self, iris):  # 加载数据集
        # 将sklearn中的数据集datasets.load_iris()的特征作分类,每班样品50,样品总数150,维数4
        self.dataMat = iris["data"]
        self.dataMat = mat(self.dataMat)

    """
    def file2matrix(self,path,delimiter):
        recordlsit = []
        fp = open(path)
        content = fp.read()
        fp.close()
        rowlist = content.splitlines()#按行转换为一维表
        # 逐行遍历 结果按分隔符分隔为行向量
        recordlsit = list(map(eval,row.split(delimiter)) for row in rowlist if row.strip())
        self.dataMat = mat(recordlsit)
    """
    def normalize(self, dataMat):  # 数据归一化
        [m, n] = shape(dataMat)
        for i in arange(n - 1):
            dataMat[:, i] = (dataMat[:, i] - mean(dataMat[:, i])) / (std(dataMat[:, i]) + 1.0e-10)
        return dataMat

    def distEclud(self, matA, matB):  # 计算欧式距离
        ma,na = shape(matA)
        mb,nb = shape(matB)
        rtnmat = zeros((ma,nb))
        for i in arange(ma):
            for j in arange(nb):
                rtnmat[i,j]=np.linalg.norm(matA[i,:]-matB[:,j].T)
        return rtnmat


    def init_grid(self):  # 初始化第二层网络
        k=0 # 构建第二层网络模型
        grid = mat(zeros((self.M * self.N,2)))
        for i in arange(self.M):
            for j in arange(self.N):
                grid[k,:] = [i,j]
                k += 1
        print(grid.shape)
        return grid

    def ratecalc(self,i): # 学习率 和 聚类半径
        lrate = self.lratemax - ((i+1.0) * (self.lratemax - self.lratemin)) / self.Steps
        r = self.rmax - ((i+1.0) * (self.rmax - self.rmin)) / self.Steps
        return lrate,r

    def trainSOM(self): # SOM网络的实现
        dm,dn = shape(self.dataMat) # 1. 构建输入层网络
        normDataset = self.normalize(self.dataMat) # 归一化数据
        grid = self.init_grid() # 2. 初始化第二层分类网络
        self.w = np.random.rand(dn,self.M*self.N) # 3. 随机初始化两层之间的权值向量
        distM = self.distEclud # 确定距离公式
        # 4. 迭代求解
        if self.Steps < 5*dm:self.Steps = 5*dm  # 设定最小迭代次数
        for i in arange(self.Steps):
            lrate,r = self.ratecalc(i) # 1) 计算当前迭代次数下的学习率和分类半径
            self.lratrlist.append(lrate);self.rlist.append(r)
            # 2) 随机生成样本索引，并抽取一个样本
            k = np.random.randint(0,dm)
            mySample = normDataset[k,:]
            # 3) 计算最优节点：返回最小距离的索引值
            minIndx = (distM(mySample,self.w)).argmin()
            # 4) 计算领域
            d1 = ceil(minIndx/self.M) # 计算此节点在第二层矩阵中的位置
            d2 = mod(minIndx,self.M)
            distMat = distM(mat([d1,d2]),grid.T)
            #nodelindx = (distMat < r).nonzeor()[1] # 获取领域内的所有节点
            nodelindx = np.nonzero((distMat < r))[1] # 获取领域内的所有节点
            for j in arange(shape(self.w)[1]): # 5) 案列更新权重
                if sum(nodelindx == j):
                    self.w[:,j] = self.w[:,j] + lrate * (mySample[0]-self.w[:,j])
        self.classLabel = list(range(dm)) # 分配和存储聚类后的类别标签
        for i in arange(dm):
            dist=distM(normDataset[i, :], self.w)
            print(type(dist))
            #self.classLabel[i] = argmin(distM(normDataset[i,:],self.w))# np.argmin()求最小值的坐标
            #self.classLabel[i] = distM(normDataset[i,:],self.w).argmin()# np.argmin()求最小值的坐标
            self.classLabel[i] = np.argmin(dist)
        self.classLabel = mat(self.classLabel)

    def showCluster(self,plt): # 绘图  显示聚类结果
        lst = unique(self.classLabel.tolist()[0]) # 去除
        i = 0
        for cindx in lst:
            myclass = nonzero(self.classLabel == cindx)[1]
            xx = self.dataMat[myclass].copy()
            if i == 0:plt.plot(xx[:,0],xx[:,1],'bo')
            if i == 1:plt.plot(xx[:,0],xx[:,1],'rd')
            if i == 2: plt.plot(xx[:, 0], xx[:, 1], 'gD')
            if i == 3: plt.plot(xx[:, 0], xx[:, 1], 'c^')
            i += 1
        plt.show()
if __name__ == "__main__":
    # 加载
    SOMNet =  SOMnet()
    iris = load_iris()
    SOMNet.loadDataSet(iris)
    #SOMNet.file2matrix("test.txt",'/t')
    SOMNet.trainSOM()
    print(SOMNet.w)
SOMNet.showCluster(plt)