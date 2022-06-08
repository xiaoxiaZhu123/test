import random
import sympy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

# 返回两城市距离矩阵
# type：0 返回距离，1 返回距离的倒数
def cityDistance(city, type):
    # 两城市距离矩阵，默认为0
    city_distance = np.zeros((cityCnt, cityCnt), dtype=np.float)
    for i in range(cityCnt):
        for j in range(i+1, cityCnt):
            if type == 1:
                # 计算各个城市的距离的倒数：||city[i] - city[j]||
                city_distance[i, j] = round(1/np.linalg.norm(city[i] - city[j]), 5)
                
            else:
                # 计算各个城市的距离：||city[i] - city[j]||
                city_distance[i, j] = round(np.linalg.norm(city[i] - city[j]), 5)
            city_distance[j, i] = city_distance[i, j]
    return city_distance

# 返回城市坐标矩阵
def cityDot(cityCnt):
    # 城市坐标矩阵，默认为0
    city = np.zeros((cityCnt, 2), dtype=np.float)
    # 随机数种子
    seed = random.seed(100)
    for i in range(cityCnt):
        # 产生0-100之间的数
        city[i, 0] = round(random.random()*100, 2)
        city[i, 1] = round(random.random()*100, 2)
    return city

# 寻找最佳路径
def findBestRoute(city_dist):  
    # m只蚂蚁的起点城市，默认为0
    ant_startCity = np.zeros((m, 1), dtype=np.int)
    
    # m只蚂蚁经过的城市路径
    ant_city = np.zeros((m, cityCnt+1), dtype=np.int)

    # m只蚂蚁经过所有城市的路径长度
    ant_city_length = np.zeros(m, dtype=np.float)

    # 各个迭代的最佳路径
    route_best = np.zeros((NC_max, cityCnt+1), dtype=np.int)
    
    # 各个迭代的最佳路径长度
    route_best_length = np.zeros((NC_max, 1), dtype=np.float)

    # 所有城市
    all_city = np.array([i for i in range(cityCnt)])

    # 全局变量（修改函数外的全局变量，要用global）
    global density

    # 初始迭代次数
    NC = 0
    while NC < NC_max:
        # 随机产生m只蚂蚁的起点城市
        for i in range(m):
            ant_startCity[i,0] = random.randint(0, cityCnt-1)

        # 给m只蚂蚁的城市路径矩阵赋上初值
        ant_city[:, [0]] = ant_startCity[:, [0]]

        # 每只蚂蚁依次寻找最佳路径
        for i in range(m):
            # 依次访问每个城市
            for j in range(1, cityCnt):
                # 已访问的城市
                visitedCity = ant_city[i, :j]
                # visitedCity = ant_city[0, :1]
                # 未访问的城市=所有城市-已访问城市
                unvistedCity = set(all_city) - set(visitedCity)

                # 已访问城市个数
                visLen = visitedCity.size
                # 已访问城市i到未访问城市j的概率，默念为0
                p = {}
                for k in unvistedCity:
                    test = density[visitedCity[visLen-1], k]**alpha * city_dist[visitedCity[visLen-1], k]**belta
                    # 已访问城市i到未访问城市j的概率
                    p[k] = test
                # 已访问城市i到所有未访问城市j的概率
                p_values = np.array(list(p.values()))
                sumP = np.sum(p_values)
                for a in p:
                    p[a] /= sumP
                
                # 轮盘赌法选择下一个城市
                # 随机产生一个数
                rand = random.random()
                # 记录轮盘每个区域边界
                s = 0.0
                for b in p:
                    # 轮盘每个区域边界=已访问城市i到所有未访问城市j概率之和
                    s += p[b]
                    if rand <= s:
                        # 选择的下个城市
                        nextCity = b
                        break
                ant_city[i, j] = nextCity
            # 每只蚂蚁最后一个城市回到原点
            ant_city[i, cityCnt] =  ant_city[i, 0]
            
            # 每只蚂蚁完成一次迭代走的路径
            for Len in range(ant_city[i].size-1):
                ant_city_length[i] += city_distance[ant_city[i, Len], ant_city[i, Len+1]]

        # 更新信息素
        delta_density = np.zeros((cityCnt, cityCnt))
        # m只蚂蚁在j到j+1路径上增加的信息素
        for i in range(m):
            for j in range(cityCnt):
                delta = Q/city_distance[ant_city[i, j], ant_city[i, j+1]]
                delta_density[ant_city[i, j], ant_city[i, j+1]] += round(delta, 3)
                delta_density[ant_city[i, j+1], ant_city[i, j]] = delta_density[ant_city[i, j], ant_city[i, j+1]]        
        density = (1-ro)*density + delta_density

        # 每轮迭代最佳路径
        route_best[NC] = ant_city[np.argmin(ant_city_length)]    
        # 每轮迭代最佳路径长度
        route_best_length[NC] = np.min(ant_city_length)  

        # 迭代次数增1
        NC = NC + 1
        # 数组清0
        ant_startCity.fill(0)
        ant_city.fill(0)
        ant_city_length.fill(0)
    # 所有迭代最佳路径
    bestRoute = route_best[np.argmin(route_best_length[np.where(route_best_length>0)])]

    # 所有迭代最佳路径长度
    bestLength = np.min(route_best_length[np.where(route_best_length>0)])

    # 画图，每只蚂蚁走的路径
    #设置字体
    xn = city[:, [0]]
    fn = city[:, [1]]
    xn_test =[city[item, 0] for item in bestRoute]
    fn_test = [city[item, 1] for item in bestRoute]

    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['font.sans-serif'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = 'False'
    plt.plot(xn, fn, 'ro', label='原始数据')
    plt.plot(xn_test, fn_test, 'g-', label='最佳路径')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('最佳路径选择')
    plt.legend()
    plt.show()
    
    return bestRoute, bestLength


# 初始化参数
# 城市数量
cityCnt = 50

# 蚂蚁数量
m = int(cityCnt * 1.5)

# 信息素浓度
# 全1矩阵
density = np.ones((cityCnt, cityCnt))
# 对角线为0
density[np.eye(cityCnt, dtype=np.bool)] = 0

# 信息素因子,[1,4]比较合适
alpha = 1

# 启发函数因子,[1, 5]比较合适
belta = 4

# 信息素挥发因子,[0.2, 0.5]比较合适
ro = 0.7

# 信息素常数，[10, 1000]比较合适
Q = 10

# 最大迭代次数，[100, 500]比较合适
NC_max = 100

# 产生城市坐标
city = cityDot(cityCnt)

# 各城市的距离的倒数
city_dist = cityDistance(city, 1)

# 各城市的距离
city_distance = cityDistance(city, 0)

#最佳路径
bestRoute, bestLength = findBestRoute(city_dist)
print('最佳路径：{}\n'.format(bestRoute))
print('最佳路径长度：{:.2f}\n'.format(bestLength))

