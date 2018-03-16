import numpy as np


class PathPlanning:
    """
    Path Planning Method class
    """
    def __init__(self, cur_pos, dst_pos):
        self.cur_pos = cur_pos
        self.dst_pos = dst_pos

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

    def naive(self, step):
        """
        Naive path planning method. It takes a straight line towards the destination.
        :param step: step length
        :return: a movement in two directions
        """
        return (self.dst_pos - self.cur_pos) / np.linalg.norm(self.dst_pos - self.cur_pos) * step

    # Algorithm dictionary
    algorithms = {'naive': naive}

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
        act = self.algorithms[strategy](self, step)
        act = np.array(np.round(act), dtype=int)
        for i, a in enumerate(act):
            if a < -3:
                act[i] = -3
            elif a > 3:
                act[i] = 3
        ret_act = np.zeros(15, dtype=int)
        ret_act[act[0] + 3] = 1
        ret_act[act[1] + 3 + 7] = 1

        return ret_act

    # TODO: Add A-star/Dijkstra


if __name__ == '__main__':
    cur_pos = np.array([0, 0])
    dst_pos = np.array([120, 50])
    pp = PathPlanning(cur_pos, dst_pos)

    while (cur_pos != dst_pos).all():
        dp = pp.run(cur_pos, 3)
        dp = np.array(np.ceil(dp), dtype=int)
        cur_pos += dp
        print(cur_pos)
