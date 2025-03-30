import matplotlib.pyplot as plt
import numpy as np
from Separate_regions import *
from Convert_lines import *
from Count_line import *
import multiprocessing
from sklearn.cluster import KMeans
import sys
import time
import os
import re
import pandas as pd


# i = 0
# j = 0
def CountGrain(file_name):
    picture = (np.loadtxt(file_name).reshape([800, 800]) * 255).astype("uint8")

    ## Separate regions
    my_separator = Separator()
    bi_pic = my_separator.preprocess(
        picture, binary_threshold=51, kernel_radius=2, min_distance=2
    )
    labels, n = my_separator.Separate(bi_pic, valid_area=32)
    regions, regions_c = my_separator.Generate_regions(labels, picture)

    outs = []
    ks = []
    for i in range(len(regions)):
        ## Convert lines
        skeleton = Skeletonize(regions[i])  # Skeletonize the regions
        lines = SearchLines(
            skeleton
        )  # Separate lines that cross each other in a region

        ## Count line
        my_counter = Counter()
        total_lines = []
        for j in range(len(lines)):

            pixels = len(lines[j].memory)  # the length of the lines
            if pixels < 5:  # ignore the lines that are too short
                continue

            converted_lines = my_counter.HoughTransform(
                lines[j].memory, theta_sep=1 / 180 * np.pi, rho_sep=1, rho_amplitude=100
            )

            if converted_lines is not None:
                converted_lines = [
                    grain[0] for grain in converted_lines
                ]  # reorganize the dimension of the array
                total_lines.extend(converted_lines)
            else:
                continue

        all_lines = my_counter.Approximate(
            total_lines, rho_error=3, theta_error=15 / 180 * np.pi
        )  # merge the lines belonging the same lines
        valid_grains = my_counter.Vote(
            all_lines, voter_threshold=5
        )  # filter the final lines that are too short

        outs.append(len(valid_grains))
        ks.extend([valid_grains[i][0][1] for i in range(len(valid_grains))])

        # print(outs)
        # print(ks)

    ns = sum(outs)
    return ns, ks


def aggreagtion(ks):
    """
    考虑颜色需要考虑如何经过霍夫转换后保留原像素的位置，需要在文件中重写霍夫转换代码
    工程量较大，暂时不考虑
    有更多更灵活的方式，可以查看Object detection相关的文献等


    各个通道晶粒数量在总晶粒的分类基础上进行k=2的k均值聚类
    """
    ks = np.array(ks).reshape([-1, 1])
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(
        ks.reshape([-1, 1])
    )  # 由于只有两种方向，所以分为两类。考虑晶粒颜色的需要再进一步改动
    max_n = kmeans.labels_.max()

    statistics = []
    for i in range(max_n + 1):
        statistics.append(
            sum(kmeans.labels_ == i)
        )  # acquire the number of grains in separate classies

    centers = np.abs(
        kmeans.cluster_centers_.ravel()
    )  # use the center to make sure all the classes output by different picture are the same classes( the same class share the same center, at least the same order of their center)
    statistics = np.array(statistics)[centers.argsort()]
    # print(centers[centers.argsort()])
    # print(statistics)
    return statistics


def main():
    try:
        dir_p = sys.argv[1]
    except IndexError:  # if dir_path is not given, then use the default one
        dir_p = "data"

    # sort the files in a chronological order
    files = os.listdir(dir_p)
    order_key = [int(re.findall("\d+(?=\.)", files[i])[0]) for i in range(len(files))]
    files = np.array(files)[np.array(order_key).argsort()]
    file_name = [dir_p + "/" + str(files[i]) for i in range(len(files))]

    global_n = []
    global_k = []

    time_start_t = time.time()
    for i in range(len(file_name)):
        time_md1 = time.time()
        print("目标文件%d开始执行" % (i + 1))
        ns, ks = CountGrain(file_name[i])  #  ns: n grains, ks: their slopes
        statistics = aggreagtion(ks)  #  separate each channel
        time_md2 = time.time()
        print("目标文件%d执行完毕,耗时%fs" % (i + 1, time_md2 - time_md1))

        global_n.append(ns)
        global_k.append(statistics)

    global_k = np.array(global_k)
    global_n = np.array(global_n)
    total = np.vstack([global_n, global_k.T])

    data = pd.DataFrame(total.T, columns=["Sum", "Channel 1", "Channel 2"])
    data.index = files
    data.to_excel("output.xlsx")

    time_end_t = time.time()
    print("共耗时%f s" % (time_end_t - time_start_t))


if __name__ == "__main__":
    main()
    pass
