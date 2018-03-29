import numpy as np
import matplotlib.pyplot as plt

from robot_config import *
from robot_environ import *
#z# from robot_client import *

from state_machines.norm_attack import NormAttack

import pdb

if __name__ == '__main__':

    #z# pygame.init()
    #z# screen = pygame.display.set_mode((800, 500), 0, 32)
    raw_map_img, bars = create_raw_map_img()
    flag = config()
    #z# draw_init(screen, bars)

    # temp
    # act_1_p = np.array([[0,0,0,0,0,0,1,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,1,0,0,0,0,0,0,1,1]])
    act_2_p = np.zeros((2, flag.mov_num * 2 + 1), np.int64)
    act_2_p[:, 0] = 1
    act_2_p[:, -1] = 1
    act_2_p[0, -2] = 1
    act_2_p[1, flag.mov_num] = 1

    # test
    info_1, info_2, map_img = get_init()

    norm_attack_1 = NormAttack(0, info_1, info_2)
    norm_attack_2 = NormAttack(1, info_1, info_2)

    from time import time, sleep

    t = time()

    def show(way1, way2, self_info, enemy_info):
        img = raw_map_img.copy()
        img[self_info[0][0]-25:self_info[0][0]+25, self_info[0][1]-25:self_info[0][1]+25] = np.array([0,255,0], np.uint8)
        img[self_info[1][0]-25:self_info[1][0]+25, self_info[1][1]-25:self_info[1][1]+25] = np.array([0,0,255], np.uint8)
        img[enemy_info[0][0]-25:enemy_info[0][0]+25, enemy_info[0][1]-25:enemy_info[0][1]+25] = np.array([255,0,0], np.uint8)
        img[enemy_info[1][0]-25:enemy_info[1][0]+25, enemy_info[1][1]-25:enemy_info[1][1]+25] = np.array([255,0,0], np.uint8)
        for i,w in enumerate(way1[1:-1]):
            img[w[0]-5:w[0]+5, w[1]-5:w[1]+5] = np.array([0,120,0], np.uint8)
        for i,w in enumerate(way2[1:-1]):
            img[w[0]-5:w[0]+5, w[1]-5:w[1]+5] = np.array([0,0,120], np.uint8)
        img = img.transpose((1, 0, 2))
        plt.imshow(img)
        plt.show()


    for global_step in range(flag.steps):
        if global_step%1000==0:
            info_1, info_2, map_img = get_init()
            info_2[0:1][0:1] = np.zeros((2, 2))
        print('time spent ', time() - t)
        t = time()
        act_1_p = np.array([norm_attack_1.run(info_1, info_2), norm_attack_2.run(info_1, info_2)])
        show(norm_attack_1.path_planner.way, norm_attack_2.path_planner.way, info_1, info_2)

        # environ and optimize
        info_1, info_2, r1, r2, map_img_new = environ(flag, info_1, info_2, act_1_p, act_2_p, raw_map_img)
        #pdb.set_trace()

        #draw_state(screen, bars, info_1, info_2)

