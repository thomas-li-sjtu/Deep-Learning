from sklearn import preprocessing
from sklearn.impute import KNNImputer
import pandas
import numpy
import re
import seaborn as sns
import matplotlib.pyplot as plt


def add_columns():
    """添加列名"""
    raw_data = pandas.read_csv("./dataset/1.csv")
    columns = ["id", "title", "intro", "price", "location", "position", "room_num", "community", "area", "listing_time",
               "last_trade", "house_year", "layer"]
    raw_data = pandas.DataFrame(raw_data.values, columns=columns)
    raw_data.to_csv("./dataset/raw.csv", index=False)
    return raw_data

raw_data = add_columns()
raw_data = pandas.read_csv("./dataset/raw.csv")
tmp = raw_data["id"].values
data = []
for i in range(len(tmp)):
    data.append(tmp[i].split("举报")[0])
data = pandas.DataFrame(numpy.array(data), columns=["id"])


def layers(raw_data):
    """处理列layer"""
    layer = []  # 楼层位置
    total_layers = []  # 总楼层数
    lay_0 = []
    lay_1 = []
    lay_2 = []
    lay_3 = []
    lay_4 = []
    lay_5 = []
    lay_6 = []
    lay_7 = []
    lay_8 = []
    layers = raw_data["layer"].values
    for i in range(len(layers)):
        tmplayer = layers[i].split("/")
        if len(tmplayer) != 2:
            digit = ""
            for j in tmplayer[0]:
                if j.isdigit():
                    digit += j
            total_layers.append(float(digit))
            layer.append(-1)

            lay_0.append(-1)
            lay_1.append(-1)
            lay_2.append(-1)
            lay_3.append(-1)
            lay_4.append(-1)
            lay_5.append(-1)
            lay_6.append(-1)
            lay_7.append(-1)
            lay_8.append(-1)
        else:
            if tmplayer[0] == "低楼层":  # 获得楼层位置
                layer.append(0)

                lay_0.append(1)
                lay_1.append(0)
                lay_2.append(0)
                lay_3.append(0)
                lay_4.append(0)
                lay_5.append(0)
                lay_6.append(0)
                lay_7.append(0)
                lay_8.append(0)
            elif tmplayer[0] == "中楼层":
                layer.append(1)

                lay_0.append(0)
                lay_1.append(1)
                lay_2.append(0)
                lay_3.append(0)
                lay_4.append(0)
                lay_5.append(0)
                lay_6.append(0)
                lay_7.append(0)
                lay_8.append(0)
            elif tmplayer[0] == "高楼层":
                layer.append(2)

                lay_0.append(0)
                lay_1.append(0)
                lay_2.append(1)
                lay_3.append(0)
                lay_4.append(0)
                lay_5.append(0)
                lay_6.append(0)
                lay_7.append(0)
                lay_8.append(0)
            elif tmplayer[0] == "联排":
                layer.append(3)

                lay_0.append(0)
                lay_1.append(0)
                lay_2.append(0)
                lay_3.append(1)
                lay_4.append(0)
                lay_5.append(0)
                lay_6.append(0)
                lay_7.append(0)
                lay_8.append(0)
            elif tmplayer[0] == "下叠别墅":
                layer.append(4)

                lay_0.append(0)
                lay_1.append(0)
                lay_2.append(0)
                lay_3.append(0)
                lay_4.append(1)
                lay_5.append(0)
                lay_6.append(0)
                lay_7.append(0)
                lay_8.append(0)
            elif tmplayer[0] == "双拼":
                layer.append(5)

                lay_0.append(0)
                lay_1.append(0)
                lay_2.append(0)
                lay_3.append(0)
                lay_4.append(0)
                lay_5.append(1)
                lay_6.append(0)
                lay_7.append(0)
                lay_8.append(0)
            elif tmplayer[0] == "独栋":
                layer.append(6)

                lay_0.append(0)
                lay_1.append(0)
                lay_2.append(0)
                lay_3.append(0)
                lay_4.append(0)
                lay_5.append(0)
                lay_6.append(1)
                lay_7.append(0)
                lay_8.append(0)
            elif tmplayer[0] == "上叠别墅":
                layer.append(7)

                lay_0.append(0)
                lay_1.append(0)
                lay_2.append(0)
                lay_3.append(0)
                lay_4.append(0)
                lay_5.append(0)
                lay_6.append(0)
                lay_7.append(1)
                lay_8.append(0)
            elif tmplayer[0] == "地下室":
                layer.append(8)

                lay_0.append(0)
                lay_1.append(0)
                lay_2.append(0)
                lay_3.append(0)
                lay_4.append(0)
                lay_5.append(0)
                lay_6.append(0)
                lay_7.append(0)
                lay_8.append(1)
            else:
                print(tmplayer[0])
                raise IndexError
            digit = ""  # 获得总楼层
            for j in tmplayer[1]:
                if j.isdigit():
                    digit += j
            total_layers.append(float(digit))

    print(len(layers))

    data["layer"] = layer
    data["layer0"] = lay_0  # 独热编码
    data["layer1"] = lay_1
    data["layer2"] = lay_2
    data["layer3"] = lay_3
    data["layer4"] = lay_4
    data["layer5"] = lay_5
    data["layer6"] = lay_6
    data["layer7"] = lay_7
    data["layer8"] = lay_8
    data["total_layers"] = total_layers


def price_area_room(raw_data):
    """处理列price,area,room"""
    price = []
    area = []
    num_room = []
    num_hall = []
    data_price = raw_data["price"].values
    data_area = raw_data["area"].values
    data_room = raw_data["room_num"].values
    for i in range(len(data_area)):
        area.append(float(data_area[i].split("平米")[0]))  # 获得平米数
        numbers = re.findall(r'\d', data_room[i])  # 正则匹配获得厅室数目
        num_room.append(int(numbers[0]))  # 获得房间数目
        num_hall.append(int(numbers[1]))  # 获得厅数目
        price.append(float(data_price[i]))  # 获得房屋价格

    data["price"] = (price - numpy.min(price))/(numpy.max(price) - numpy.min(price))
    data["area"] = numpy.array(area)
    data["room_num"] = numpy.array(num_room)
    data["hall_num"] = numpy.array(num_hall)


def position_location(raw_data):
    """处理列position,location"""
    global data
    position = []
    ring = []
    region = []
    data_position = raw_data["position"].values
    data_location = raw_data["location"].values

    elem_position = list(set(data_position))  # 检索position的属性取值范围
    for i in range(len(elem_position)):
        tmp = sorted(elem_position[i].split(" "))
        elem_position[i] = ' '.join(tmp)
    elem_position = list(set(elem_position))
    for i in range(len(elem_position)):  # "暂无数据"放到前面
        if elem_position[i] == "暂无数据":
            elem_position[0], elem_position[i] = elem_position[i], elem_position[0]

    with open("./dataset/position.txt", "w" ,encoding="utf-8") as file:  # 存储映射关系
        for i in range(len(elem_position)):
            file.write(elem_position[i] + "\t" + str(i) + "\n")

    for i in range(len(data_position)):  # 映射
        tmp = sorted(data_position[i].split(" "))
        tmp = ' '.join(tmp)
        position.append(elem_position.index(tmp)-1)
    data["position"] = position

    elem_region = [data_location[i].split("\xa0")[0] for i in range(len(data_location))] # 检索location的属性取值范围
    elem_ring = [data_location[i].split("\xa0")[2] for i in range(len(data_location))]
    elem_region = list(set(elem_region))
    elem_ring = list(set(elem_ring))

    with open("./dataset/region.txt", "w" ,encoding="utf-8") as file:  # 存储映射关系
        for i in range(len(elem_region)):
            file.write(elem_region[i] + "\t" + str(i) + "\n")
    with open("./dataset/ring.txt", "w" ,encoding="utf-8") as file:  # 存储映射关系
        for i in range(len(elem_ring)):
            file.write(elem_ring[i] + "\t" + str(i) + "\n")

    for i in range(len(data_location)):  # 映射
        tmpregion = data_location[i].split("\xa0")[0]
        tmpring = data_location[i].split("\xa0")[2]
        region.append(elem_region.index(tmpregion))
        ring.append(elem_ring.index(tmpring)-1)
    data["ring"] = ring
    data["region"] = region

    ring_0 = []
    ring_1 = []
    ring_2 = []
    ring_3 = []
    for i in data["ring"].values:
        if i == 0:
            ring_0.append(1)
            ring_1.append(0)
            ring_2.append(0)
            ring_3.append(0)
        elif i == 1:
            ring_0.append(0)
            ring_1.append(1)
            ring_2.append(0)
            ring_3.append(0)
        elif i == 2:
            ring_0.append(0)
            ring_1.append(0)
            ring_2.append(1)
            ring_3.append(0)
        elif i == 3:
            ring_0.append(0)
            ring_1.append(0)
            ring_2.append(0)
            ring_3.append(1)
        elif i == -1:
            ring_0.append(-1)
            ring_1.append(-1)
            ring_2.append(-1)
            ring_3.append(-1)
    data["ring0"] = ring_0
    data["ring1"] = ring_1
    data["ring2"] = ring_2
    data["ring3"] = ring_3
    data = data.drop(columns=["ring"])


def description(raw_data):
    """从描述中获取信息"""
    traffic = []
    light = []
    parking = []
    elevator = []
    decoration = []
    title = raw_data["title"].values
    intro = raw_data["intro"].values

    for i in range(len(title)):
        if "地铁" in title[i] or "号线" in title[i] or "交通" in title[i] \
                or "地铁" in intro[i] or "号线" in intro[i] or "交通" in intro[i]:
            traffic.append(1)
        else:
            traffic.append(0)

        if "采光" in title[i] or "采光" in intro[i]:
            light.append(1)
        else:
            light.append(0)

        if "车位" in title[i] or "车位" in intro[i]:
            parking.append(1)
        else:
            parking.append(0)

        if "电梯" in title[i] or "电梯" in intro[i] \
                or "一梯" in title[i] or "一梯" in intro[i]:
            elevator.append(1)
        else:
            elevator.append(0)

        if "精装" in title[i] or "精装" in intro[i] \
                or "装修" in title[i] or "装修" in intro[i] \
                or "拎包" in title[i] or "拎包" in intro[i]:
            decoration.append(1)
        else:
            decoration.append(0)

    data["traffic"] = numpy.array(traffic)
    data["parking"] = numpy.array(parking)
    data["elevator"] = numpy.array(elevator)
    data["decoration"] = numpy.array(decoration)
    data["light"] = numpy.array(light)


def fill_missing(data):
    """补充缺失值，缺失值为-1"""
    imputer = KNNImputer(missing_values=-1, n_neighbors=20, weights='distance', metric='nan_euclidean')  # 用knn来填补缺失值
    tmp = imputer.fit_transform(data)
    tmp = pandas.DataFrame(data=tmp, index=None, columns=[i for i in data])
    return tmp


layers(raw_data)
price_area_room(raw_data)
position_location(raw_data)
description(raw_data)
data = fill_missing(data)

data.to_csv("./dataset/data.csv", index=False)


