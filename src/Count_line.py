'''
构建霍夫转换后不同直线的距离矩阵，

算法思路借鉴了Dijkstra算法和Q聚类, 先阅读相关资料有助于理解。
'''
import time

import numpy as np
import cv2

class Counter():
    def __init__(self):  # rho is the distance between o and the line, it should be equal to the diagonal
        pass


    def HoughTransform(self,footsteps, theta_sep=1, rho_sep=1, rho_amplitude=100):
        '''
        https://docs.opencv.org/3.4/d3/de6/tutorial_js_houghlines.html#:~:text=Hough%20Transform%20in%20OpenCV%20Everything%20explained%20above%20is,use%20canny%20edge%20detection%20before%20applying%20hough%20transform.
        '''

        footsteps = np.array(footsteps)
        footsteps = np.float32(footsteps[:,np.newaxis,:])
        lines = cv2.HoughLinesPointSet(footsteps, 5, 5, min_rho=-rho_amplitude ,max_rho=rho_amplitude, rho_step=rho_sep, min_theta=-np.pi/2, max_theta=np.pi/2, theta_step=theta_sep)
        return lines


    def Approximate(self, lines, rho_error=3, theta_error=15/180*np.pi):  
        '''
        There are many lines with slight difference are distinguished by HoughTransform are actually the same lines.
        This Function is intended to classify the lines with similar parameters
        '''
        def weighted_distence(l1, l2, mode = 'p2p'):  
            '''
            This function is used to generate the distance between two lines which may or may not actually the same line.
            the "distance" here is actually the error of the deviation of their corrosponding parameters: k and b.
            k and b are diveded by a weight to equal their influence to the error
            '''
            if mode == 'p2p':  # point to point
                x1,y1 = l1[1], l1[2]
                x2,y2 = l2[1], l2[2]
            elif mode == 's2p':  # point sets to point
                l1 = np.array(l1)
                x1,y1 = l1[:,1].mean(),l1[:,2].mean()
                x2,y2 = l2[1], l2[2]
            else:
                print('mode undefined')

            result = np.sqrt(((x1-x2)/rho_error)**2+((y1-y2)/theta_error)**2) 

            return result


        n = len(lines)  # lines to be merged
        dist = np.ones((n,n))*np.inf  # initialize dist
        for i in range(n):
            for j in range(n):
                if i!=j:
                    dist[i,j] = weighted_distence(lines[i],lines[j])  
                else:
                    dist[i,j] = np.inf  # dist[i,i] = np.inf, so (dist[i,:]).argmin() can be actually the most similar line to lines[i]

        threshold = np.sqrt(2)  # above which is considered to be diffenrent lines.  # The error of rho and theta are weighted so it's ok to set threshold to sqrt(2)
        all_lines_ind = []

        while not (dist==np.inf).all():  # start classification
            '''
            all_lines_ind = [same_lines_ind1, same_lines_ind2, ...]
            same_lines_ind = [lines_ind1, lines_ind2, ...]

            there two loops:
            1. The first loop is used to put same_lines_ind into all_lines_ind
            2. The Second one is used to put lines_ind into same_lines_ind
            
            The Second one:
            1. if same_lines_ind(or if dist == np.inf) True goto 2 False break
            2. for all lines in same_lines_ind, find their separately the closest line
            3. get the distance from the center of the same_lines_ind to each closest line
            4. if all distance is above threshold Break, else only put the closest line into the same_lines_ind
            5. goto 1
            '''
            same_lines_ind = []  
            i,j = np.argwhere(dist==dist.min())[0]  # two lines with most similar parameters in current iteration  # i:row_number, j:column_number
            dist[:,j]=np.inf  # when update distance, we only set column_number to be np.inf
            same_lines_ind.append(j)  

            while True:

                if (dist==np.inf).all():
                    break
                # the lines that close to each lines in the 'same_lines_ind'
                same_lines = [lines[k] for k in same_lines_ind]
                potential_lines_ind = [dist[k,:].argmin() for k in same_lines_ind]
                # deviation of the each 'potential_lines' to the 'same_lines'
                potential_deviation = [weighted_distence(same_lines,lines[p],mode='s2p') for p in potential_lines_ind]

                if (np.array(potential_deviation)>=threshold).all() == True:  # if deviation are not tolerable
                    break
                else:
                    p = np.array(potential_deviation).argmin()  # only add the most close line into 'same_lines_ind'
                    same_lines_ind.append(potential_lines_ind[p])
                    dist[:,potential_lines_ind[p]]=np.inf  # in case it will be found again by other LineBud
            all_lines_ind.append(same_lines_ind)  

        # convert all_lines_ind to all_lines
        all_lines = []
        for i in range(len(all_lines_ind)):
            tmp = []
            for j in range(len(all_lines_ind[i])):
                tmp.append(list(lines[all_lines_ind[i][j]]))
            all_lines.append(tmp)
        return all_lines

    # sum the count of the lines belonging to the same line.
    # filter lines below the self.voter_threshold
    def Vote(self,all_lines, voter_threshold = 5):
        out_come = np.zeros(len(all_lines))
        for i in range(len(all_lines)):
            for j in range(len(all_lines[i])):
                out_come[i] += all_lines[i][j][0]

        valid_linds =np.argwhere(out_come>voter_threshold).ravel()

        all_lines = [all_lines[i] for i in valid_linds]
        return all_lines

if __name__ == '__main__':
    pass




