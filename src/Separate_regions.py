"""
This module is used to separate the grains from its background, and try to disconnect some of the grains.

The rest connected grains are left to be solved in the following modules
"""

import cv2
import numpy as np


class Separator(object):

    def __init__(self):
        pass

    def preprocess(
        self, pic, binary_threshold=51, kernel_radius=2, min_distance=2
    ):  # 这些参数是经过调试之后选择的，如效果不理想，首先考虑调整这些参数
        """
        二值化、开运算+距离转换

        开运算和距离转换的目的都是尽量“拆开”黏连的晶粒，原理也相似。
        两个步骤都进行是非必需的，可以选择只进行其中一个。
        """

        def addrim(pic):
            """
            Adding margins to the picture in case there much pixels in the edge, which will be abnormally calculated by distanceTransform

            """
            m, n = pic.shape
            temp = np.zeros((m + 2, n + 2))
            temp[1:-1, 1:-1] = pic[:]
            pic = temp
            pic = pic.astype("uint8")
            return pic

        ret, bi_pic = cv2.threshold(
            pic, binary_threshold, 255, cv2.THRESH_BINARY
        )  # Binarize

        kernel = np.ones(
            (kernel_radius, kernel_radius), np.uint8
        )  # the kernel applied to morphological calculation
        open_bi_pic = cv2.morphologyEx(bi_pic, cv2.MORPH_OPEN, kernel)

        rim_open_bi_pic = addrim(open_bi_pic)  # adding rims
        distance = cv2.distanceTransform(
            rim_open_bi_pic, distanceType=cv2.DIST_L2, maskSize=5
        )  # 值为1的像素的值转换为到最近的值为0的像素的距离
        final_pic = (
            distance > min_distance
        ) * 255  # 低于min_distance的像素会删除，即会删除晶粒的边缘像素。
        final_pic = bi_pic.astype("uint8")

        return final_pic

    def Separate(self, pic, valid_area=32):

        n_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(pic)
        valid_grains = np.argwhere(stats[:, -1] > valid_area).ravel()
        map_to_filter = np.zeros(labels.max() + 1)
        map_to_filter[valid_grains] = np.arange(len(valid_grains))
        labels = map_to_filter[labels]

        return labels, len(valid_grains)

    def Generate_regions(self, labels, pic):
        """
        Dividing the pictures into many regions containing one or more connected grains

        This function is used to generate regions to be analysised in the following modules
        """
        n_objects = int(labels.max())
        get_boundary = lambda x: [
            x[:, 0].min(),
            x[:, 0].max(),
            x[:, 1].min(),
            x[:, 1].max(),
        ]
        region_of_grains_binary = []
        region_of_grains_original = []
        for ind in range(1, n_objects + 1):
            min_x, max_x, min_y, max_y = get_boundary(np.argwhere(labels == ind))
            binary_region = (labels == ind)[min_x:max_x, min_y:max_y].astype(
                "uint8"
            ) * 255
            original_region = pic[min_x:max_x, min_y:max_y].astype("uint8")
            original_region = cv2.bitwise_and(
                original_region, original_region, mask=binary_region.astype("uint8")
            )

            region_of_grains_binary.append(binary_region)
            region_of_grains_original.append(original_region)
        return region_of_grains_binary, region_of_grains_original


if __name__ == "__main__":

    # test code
    from matplotlib import pyplot as plt

    file_name = "precipitation_grains/ita12-5000.txt"
    picture = (np.loadtxt(file_name).reshape([800, 800]) * 255).astype("uint8")
    my_separator = Separator()
    bi_pic = my_separator.preprocess(picture)
    labels, n = my_separator.Separate(bi_pic)
    regions, _ = my_separator.Generate_regions(labels, picture)
    plt.imshow(regions[0])
    plt.savefig("test.png")
