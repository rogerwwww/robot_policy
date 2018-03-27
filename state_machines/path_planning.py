import numpy as np
import time
import pdb
import random

scale = 0.1
robot_size = [int(500 * scale), int(500 * scale)]
d_size = [_ // 2 for _ in robot_size]

rescale = 0.1
d_size = [int(_ * rescale) for _ in d_size]
act_dict = [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12]


class PathPlanning:
    """
    Path Planning Method class
    """

    def __init__(self, cur_pos, dst_pos):
        self.cur_pos = cur_pos
        self.dst_pos = dst_pos  # pdb.set_trace()

    def update_dst(self, update_pos):
        """
        Update destination position
        :param update_pos: updated destination position
        :return: None
        """
        self.dst_pos = update_pos

    @property
    def get_dst(self):
        """
        Get current destination
        :return: destination coordinate
        """
        return self.dst_pos

    def naive(self, step, weight_map):
        """
        Naive path planning method. It takes a straight line towards the destination.
        :param step: step length
        :parma weight_map: input weight map (no effect in naive)
        :return: a movement in two directions
        """
        return (self.dst_pos - self.cur_pos) / np.linalg.norm(self.dst_pos - self.cur_pos) * step

    def is_pos(self, pos, dir_tmp, weight_map):
        """
        Judge corners' position whether illegal
        """
        h, w = weight_map.shape
        pos_tmp = (pos[0] + dir_tmp[0], pos[1] + dir_tmp[1])
        corners = [[pos_tmp[0] + d_size[0], pos_tmp[1] + d_size[1]], [pos_tmp[0] + d_size[0], pos_tmp[1] - d_size[1]],
                   [pos_tmp[0] - d_size[0], pos_tmp[1] + d_size[1]], [pos_tmp[0] - d_size[0], pos_tmp[1] - d_size[1]]]
        for corner in corners:
            if corner[0] < 0 or corner[0] >= h or corner[1] < 0 or corner[1] >= w:
                return False
        if sum(sum(weight_map[pos_tmp[0] - d_size[0]:pos_tmp[0] + d_size[0],
                   pos_tmp[1] - d_size[1]:pos_tmp[1] + d_size[1]])) < 3 * 255 * 4 * d_size[0] * d_size[1]:
            return False
        return True

    def get_children(self, pos, weight_map, map_father, map_dist):
        """
        get children and update its father and distance
        """
        threshold = 3 * 255
        dir_10 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        dir_14 = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

        h, w = weight_map.shape
        pos_children = []
        # up,down,left,right
        for dir_tmp in dir_10:
            if not self.is_pos(pos, dir_tmp, weight_map):
                continue
            pos_tmp = (pos[0] + dir_tmp[0], pos[1] + dir_tmp[1])
            if weight_map[pos_tmp] >= threshold and map_dist[pos_tmp] < 0:
                map_father[pos_tmp] = pos
                map_dist[pos_tmp] = map_dist[pos] + 10
                pos_children.append(pos_tmp)

        # left-up,right-down,left-down,right-up
        for dir_tmp in dir_14:
            if not self.is_pos(pos, dir_tmp, weight_map):
                continue
            pos_tmp = (pos[0] + dir_tmp[0], pos[1] + dir_tmp[1])
            if weight_map[pos_tmp] >= threshold and map_dist[pos_tmp] < 0:
                map_father[pos_tmp] = pos
                map_dist[pos_tmp] = map_dist[pos] + 14
                pos_children.append(pos_tmp)
        return pos_children

    def reshapeMap(self, weight_map):
        """
        resize map
        """
        while len(weight_map.shape) > 2:
            weight_map = np.sum(weight_map, -1)

        rescale = 0.1

        new_weight_map = weight_map[::int(1 / rescale), ::int(1 / rescale)]

        return new_weight_map

    def a_star(self, step, weight_map):
        """
        simple A-star
        """
        # cur_pos_init = tuple(self.cur_pos[::-1])
        # dst_pos_init = tuple(self.dst_pos[::-1])
        cur_pos_init = tuple(self.cur_pos)
        dst_pos_init = tuple(self.dst_pos)
        weight_map_init = weight_map

        # resize
        weight_map = self.reshapeMap(weight_map_init)
        h, w = weight_map.shape
        cur_pos = [int(rescale * _) for _ in cur_pos_init]
        dst_pos = [int(rescale * _) for _ in dst_pos_init]

        cur_pos = tuple(cur_pos)
        dst_pos = tuple(dst_pos)

        # init
        map_father = -1 * np.ones((h, w, 2), np.int16)
        map_dist = -1 * np.ones((h, w), np.int16)
        pos_tbd = [cur_pos]

        # map
        map_father[cur_pos] = cur_pos
        map_dist[cur_pos] = 0
        not_end = True
        pos_did = []
        while len(pos_tbd) > 0 and not_end:
            pos = pos_tbd.pop(0)
            pos_did.append(pos)
            # get child
            pos_children = self.get_children(pos, weight_map, map_father, map_dist)
            if len(pos_children) > 0:
                pos_tbd += pos_children
                for pos_tmp in pos_children:
                    if pos_tmp[0] == dst_pos[0] and pos_tmp[1] == dst_pos[1]:
                        not_end = False

        # get way
        way = []

        # change dst to ground
        pos_father = ori_pos_father = dst_pos
        dist = 2
        pos_father_list = []
        while True:
            try:
                if map_father[pos_father][0] != -1:
                    break
            except IndexError:
                pass
            if len(pos_father_list) == 0:
                dist += 1
                for _ in [(dist, dist), (dist, -dist), (-dist, dist), (-dist, -dist)]:
                    pos_father_list.append(tuple(np.array(ori_pos_father) + _))
            pos_father = pos_father_list.pop()

        while pos_father[0] != cur_pos[0] or pos_father[1] != cur_pos[1]:
            way.append(tuple([int(_ / rescale) for _ in pos_father]))
            # get father
            pos_father = tuple(map_father[pos_father])
            if pos_father[0] < 0:
                pdb.set_trace()
        way.append(tuple([int(_ / rescale) for _ in cur_pos]))
        try:
            if np.linalg.norm(abs(np.array(dst_pos_init) - np.array(way[1])), ord=np.inf) > np.linalg.norm(
                    abs(np.array(way[0]) - np.array(way[1])), ord=np.inf):
                way = [dst_pos_init] + way
            else:
                way[0] = dst_pos_init
            if np.linalg.norm(abs(np.array(cur_pos_init) - np.array(way[-2])), ord=np.inf) > np.linalg.norm(
                    abs(np.array(way[-1]) - np.array(way[-2])), ord=np.inf):
                way.append(cur_pos_init)
            else:
                way[-1] = cur_pos_init
            way.reverse()
        except IndexError:
            pass
        # print way

        next_pos = cur_pos_init
        for pos in way:
            np_pos = np.array(pos)
            if np.linalg.norm(np_pos - cur_pos_init, ord=np.inf) > step:
                break
            next_pos = pos
        return np.array(np.array(next_pos) - cur_pos_init)

    # Algorithm dictionary
    algorithms = {'naive': naive, 'a-star': a_star}

    def run(self, cur_pos, step, weight_map, strategy='naive'):
        """
        Run the path planning once
        :param cur_pos: input current position
        :param step: input step length
        :param weight_map: input weight map for path planning
        :param strategy: input destination position
        :return: a movement in two directions
        """
        self.cur_pos = cur_pos

        act = self.algorithms[strategy](self, step, weight_map)
        act = np.array(np.round(act), dtype=int)

        ret_act = np.zeros(len(act_dict) * 2 + 1, dtype=int)
        for i, a in enumerate(act):
            idx = np.argmin(abs(np.array(act_dict) - a))
            act[i] = act_dict[idx]
            ret_act[idx + i * len(act_dict)] = 1

        return ret_act

    # TODO: Try maybe Dijkstra


if __name__ == '__main__':
    cur_pos = np.array([30, 30])
    dst_pos = np.array([470, 640])
    pp = PathPlanning(cur_pos, dst_pos)

    weight_map = 255 * np.ones((500, 800, 3), np.int16)
    for i in range(70):
        weight_map[60, i] = (0, 0, 0)

    while np.linalg.norm(abs(cur_pos - dst_pos), ord=np.inf) > 14:
        # while (cur_pos != dst_pos).any():
        # dp = pp.run(cur_pos, 3, weight_map, strategy='a-star')
        # dp = np.argmax(dp[0:7]) - 3, np.argmax(dp[7:14]) - 3
        t = time.time()
        dp = pp.run(cur_pos, 14, weight_map, strategy='a-star')
        dp = 4 * (np.argmax(dp[0:7]) - 3), 4 * (np.argmax(dp[7:14]) - 3)
        cur_pos += dp
        print('cur pos', cur_pos, 'to', dst_pos, 'time', time.time() - t)
