import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import numpy
from pyecharts import Pie
from mpl_toolkits.mplot3d import Axes3D


data = pandas.read_csv("./dataset/data.csv")
raw_data = pandas.read_csv("./dataset/raw.csv")

def fig_1(raw_data):
    plt.figure(figsize=(10, 5))
    print("skew: ", raw_data.price.skew())
    sns.distplot(raw_data['price'], vertical=True)
    plt.savefig("./整体房价统计.png")

    area = []
    for i in raw_data['area'].values:
        area.append(float(i.split("平米")[0]))  # 获得平米数
    raw_data = raw_data.drop(columns=["area"])
    raw_data["area"] = area
    plt.figure(figsize=(10, 5))
    print("skew: ", raw_data.area.skew())
    sns.distplot(raw_data['area'])
    plt.savefig("./整体面积统计.png")



def fig_2(data, raw_data):
    tmp = []
    for i in data["region"].values:
        if i == 0:
            tmp.append("长宁")
        elif i == 1:
            tmp.append("杨埔")
        elif i == 2:
            tmp.append("宝山")
        elif i == 3:
            tmp.append("松江")
        elif i == 4:
            tmp.append("黄埔")
        elif i == 5:
            tmp.append("浦东")
        elif i == 6:
            tmp.append("虹口")
        elif i == 7:
            tmp.append("闵行")
        elif i == 8:
            tmp.append("青浦")
        elif i == 9:
            tmp.append("普陀")
        elif i == 10:
            tmp.append("徐汇")
        elif i == 11:
            tmp.append("嘉定")
        elif i == 12:
            tmp.append("静安")
        else:
            tmp.append("未知")
    data = data.drop(columns=["region"])
    data["region"] = tmp
    plt.figure(figsize=(15, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(data.region, raw_data.price)
    plt.savefig("./各区房价箱线图.png")

    plt.figure(figsize=(15, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(data.region, (raw_data.price)/(data.area))
    plt.savefig("./各区房均价箱线图.png")

    plt.figure(figsize=(15, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(data.region, data.area)
    plt.savefig("./各区房屋面积箱线图.png")

    plt.figure(figsize=(15, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(data.room_num, data.price)
    plt.savefig("./房价房间数箱线图.png")

    plt.figure(figsize=(15, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(data.hall_num, data.price)
    plt.savefig("./房价大厅数箱线图.png")


def fig_3(data):
    plt.figure(figsize=(20, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(data.elevator, data.price)
    plt.savefig("./电梯与房价箱线图.png")

    plt.figure(figsize=(20, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(data.parking, data.price)
    plt.savefig("./车位与房价箱线图.png")

    plt.figure(figsize=(20, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(data.decoration, data.price)
    plt.savefig("./装修与房价箱线图.png")

    plt.figure(figsize=(20, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(data.traffic, data.price)
    plt.savefig("./交通与房价箱线图.png")


def fig_4(data):
    fig1 = plt.figure(figsize=(15, 15))  # 绘制三维散点图
    z = data["price"]
    y = (data["area"].values - numpy.min(data["area"].values))/(numpy.max(data["area"].values) - numpy.min(data["area"].values))
    x = data["region"]
    ax = Axes3D(fig1)
    ax.scatter(x, y, z, s=5, marker='d')
    plt.savefig("./房价、面积、地区三维散点图.png")


def fig_5(data):
    """绘制饼图"""
    traffic_plot_x = data.traffic.value_counts().index
    traffic_plot_y = data.traffic.value_counts().values
    traffic_plot = Pie('traffic', width=900)
    traffic_plot.add(name='traffic', attr=traffic_plot_x, value=traffic_plot_y, center=[50, 60], radius=[40, 80],
                     is_random=True, rosetype='radius', is_label_show=True)
    traffic_plot.render(path="饼图traffic.html")

    room_plot_x = data.room_num.value_counts().index
    room_plot_y = data.room_num.value_counts().values
    room_plot = Pie('room', width=900)
    room_plot.add(name='room', attr=room_plot_x, value=room_plot_y, center=[50, 60], radius=[40, 80],
                     is_random=True, rosetype='radius', is_label_show=True)
    room_plot.render(path="饼图room.html")

    hall_num_plot_x = data.hall_num.value_counts().index
    hall_num_plot_y = data.hall_num.value_counts().values
    hall_num_plot = Pie('hall', width=900)
    hall_num_plot.add(name='hall', attr=hall_num_plot_x, value=hall_num_plot_y, center=[50, 60], radius=[40, 80],
                     is_random=True, rosetype='radius', is_label_show=True)
    hall_num_plot.render(path="饼图hall.html")

    decoration_plot_x = data.decoration.value_counts().index
    decoration_plot_y = data.decoration.value_counts().values
    decoration_plot = Pie('decoration', width=900)
    decoration_plot.add(name='decoration', attr=decoration_plot_x, value=decoration_plot_y, center=[50, 60], radius=[40, 80],
                     is_random=True, rosetype='radius', is_label_show=True)
    decoration_plot.render(path="饼图decoration.html")

    parking_plot_x = data.parking.value_counts().index
    parking_plot_y = data.parking.value_counts().values
    parking_plot = Pie('parking', width=900)
    parking_plot.add(name='parking', attr=parking_plot_x, value=parking_plot_y, center=[50, 60], radius=[40, 80],
                     is_random=True, rosetype='radius', is_label_show=True)
    parking_plot.render(path="饼图parking.html")


def fig_6(raw_data):
    price = []
    for i in raw_data["price"].values:
        price.append(float(i))
    data_location = raw_data["location"].values
    elem_ring = [data_location[i].split("\xa0")[2] for i in range(len(data_location))]
    plt.figure(figsize=(15, 8))
    plt.rc("font", family="SimHei", size="15")  # 解决中文乱码问题
    sns.boxplot(numpy.array(elem_ring), numpy.array(price))
    plt.savefig("./各环房价箱线图.png")
    plt.show()


fig_1(raw_data)
fig_2(data, raw_data)
fig_3(data)
fig_4(data)
fig_5(data)
fig_6(raw_data)