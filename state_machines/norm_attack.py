from state_machines.path_planning import PathPlanning, robot_size
import numpy as np

# Arrive at ROI threshold
ROI_THRESH = 20
# Single step length
STEP_LEN = 14
# Distance when close to target enough
CLOSE_THRESH = 80
# When closer than this distance, will shot when pursuit
PURSUIT_SHOT_DIST = 200


class NormAttack:
    """
    Norm Attack Finite State Machine
    """
    # Element = None if unknown
    def __init__(self, self_id, self_info, enemy_info, init_state='start'):
        # ID of current robot
        self.self_id = self_id
        # States
        self.last_state = None
        self.cur_state = init_state
        # Self and enemy information
        self.last_self_info = None
        self.self_info = np.array(self_info)
        self.last_enemy_info = None
        self.enemy_info = np.array(enemy_info)
        # Path planning instance
        self.path_planner = None
        # Target enemy robot ID
        self.target_id = 0

    def get_weight_map(self, obstacle_weight=(0, 0, 0), enemy_weight=(0, 255, 0), friend_weight=(255, 0, 0)):
        """
        Generate a weight map representing weights in the whole map.
        :param obstacle_weight: weight for obstacles
        :param enemy_weight: weight for enemy vehicle
        :param friend_weight: weight for friend vehicle
        :return: numpy array of (800 * 500)
        """
        wmap = 255 * np.ones((500, 800, 3), dtype=np.uint8)

        # Draw environment obstacles
        wmap[100:130, 120:200] = obstacle_weight
        wmap[370:400, 600:680] = obstacle_weight
        wmap[250:280, 0:80] = obstacle_weight
        wmap[220:250, 720:800] = obstacle_weight
        wmap[230:350, 180:210] = obstacle_weight
        wmap[150:270, 590:620] = obstacle_weight
        wmap[300:500, 310:340] = obstacle_weight
        wmap[0:200, 460:490] = obstacle_weight

        wmap = wmap.transpose((1, 0, 2))

        dx, dy = (_ // 2 for _ in robot_size)

        # Draw enemy robots' position
        if self.enemy_sighted:
            if self.enemy_position[1 - self.target_id, 0:2].any():
                enemy_x, enemy_y= self.enemy_position[1 - self.target_id, 0], self.enemy_position[1 - self.target_id, 1]
                wmap[enemy_x - dx: enemy_x + dx, enemy_y - dy: enemy_y + dy, :] = enemy_weight

        # Draw friend robot's position
        wmap[self.self_position[1 - self.self_id, 0] - dx: self.self_position[1 - self.self_id, 0] + dx,
             self.self_position[1 - self.self_id, 1] - dy: self.self_position[1 - self.self_id, 1] + dy,
             :] = friend_weight
        return wmap

    def run(self, self_info, enemy_info):
        """
        Run the norm attack state machine once.
        :param self_info: numpy array, input information of self
        :param enemy_info: numpy array, input information of enemy, None if element unknown
        :return: action array of this robot
        """
        # Update info
        self.last_self_info = self.self_info
        self.last_enemy_info = self.enemy_info
        self.self_info = np.array(self_info)
        self.enemy_info = np.array(enemy_info)

        # Run state machine
        act = self.state_methods[self.cur_state](self)
        return act

    def run_start(self):
        """
        Run start state once.
        :return: action array of this robot
        """
        # State transition
        if not self.enemy_sighted:
            next_state = 'patrol'
        elif not self.target_is_moving and self.close_to_target_enough:
            next_state = 'dps'
        else:
            next_state = 'pursuit'

        # Action
        if next_state == 'patrol':
            act = self.patrol_action(init=True)
        elif next_state == 'dps':
            act = self.dps_action()
        else:
            act = self.pursuit_action(init=True)

        # Perform state transition
        self.last_state = self.cur_state
        self.cur_state = next_state
        return list(act)

    def run_patrol(self):
        """
        Run patrol state once.
        :return: action array of this robot
        """
        # State transition
        if not self.enemy_sighted:
            next_state = 'patrol'
        elif not self.target_is_moving and self.close_to_target_enough:
            next_state = 'dps'
        else:
            next_state = 'pursuit'

        # Action
        if next_state == 'patrol':
            act = self.patrol_action()
        elif next_state == 'dps':
            act = self.dps_action()
        else:
            act = self.pursuit_action()

        # Perform state transition
        self.last_state = self.cur_state
        self.cur_state = next_state
        return list(act)

    def run_pursuit(self):
        """
        Run pursuit state once.
        :return: action array of this robot
        """
        # State transition
        if not self.enemy_sighted:
            next_state = 'patrol'
        elif not self.target_is_moving and self.close_to_target_enough:
            next_state = 'dps'
        else:
            next_state = 'pursuit'

        # Action
        if next_state == 'patrol':
            act = self.patrol_action()
        elif next_state == 'dps':
            act = self.dps_action()
        else:
            act = self.pursuit_action()

        # Perform state transition
        self.last_state = self.cur_state
        self.cur_state = next_state
        return list(act)

    def run_dps(self):
        """
        Run DPS state once.
        :return: action array of this robot
        """
        # State transition
        if not self.enemy_sighted:
            next_state = 'patrol'
        elif self.close_to_target_enough:
            next_state = 'dps'
        else:
            next_state = 'pursuit'

        # Action
        if next_state == 'patrol':
            act = self.patrol_action()
        elif next_state == 'dps':
            act = self.dps_action()
        else:
            act = self.pursuit_action()

        # Perform state transition
        self.last_state = self.cur_state
        self.cur_state = next_state
        return list(act)

    def patrol_action(self, init=False):
        """
        Plan and command patrol action for robot.
        :param init: whether to initiate path planner.
        :return: action array of this robot
        """
        if init:
            f = lambda x, y: (np.ceil((x - y / 2) / y) - 0.5) * 2
            dst = np.array([f(self.self_position[self.self_id][0], 800) * -300 + 400,
                            f(self.self_position[self.self_id][1], 500) * -200 + 250])
            self.path_planner = PathPlanning(self.self_position[self.self_id], dst)
        elif np.linalg.norm(self.self_position[self.self_id] - self.path_planner.get_dst) < ROI_THRESH:
            f = lambda x, y: (np.ceil((x - y / 2) / y) - 0.5) * 2
            dst = np.array([f(self.self_position[self.self_id][0], 800) * -300 + 400,
                            f(self.self_position[self.self_id][1], 500) * -200 + 250])
            self.path_planner.update_dst(dst)

        weight_map = self.get_weight_map()
        act = self.path_planner.run(self.self_position[self.self_id], STEP_LEN, weight_map, strategy='a-star')
        return act

    def pursuit_action(self, init=False):
        """
        Plan and command pursuit action for robot.
        :param init: whether to initiate the path planner
        :return: action array of this robot
        """
        # Update target
        try:
            dist0 = np.linalg.norm(self.enemy_position[0] - self.self_position[self.self_id])
            dist1 = np.linalg.norm(self.enemy_position[1] - self.self_position[self.self_id])
            self.target_id = np.argmin([dist0, dist1])
        except TypeError:
            if self.enemy_position[0] is None:
                self.target_id = 1
            else:
                self.target_id = 0

        # Pursuit target
        if init:
            self.path_planner = PathPlanning(self.self_position[self.self_id], self.enemy_position[self.target_id])
        self.path_planner.update_dst(self.enemy_position[self.target_id])
        weight_map = self.get_weight_map()
        act = self.path_planner.run(self.self_position[self.self_id], STEP_LEN, weight_map, strategy='a-star')
        if np.fmin(dist0, dist1) < PURSUIT_SHOT_DIST:
            act[-1] = 1
        return act

    def dps_action(self):
        """
        Command dps action for robot.
        :return: action array of this robot
        """
        ret_array = np.zeros(11 * 2 + 1)
        ret_array[-1] = 1
        ret_array[11 // 2] = 1
        ret_array[11 // 2 + 11] = 1
        return ret_array

    @property
    def self_position(self):
        return self.self_info[0:2, 0:2]

    @property
    def enemy_position(self):
        return self.enemy_info[0:2, 0:2]

    @property
    def last_enemy_position(self):
        if self.last_enemy_info is not None:
            return self.last_enemy_info[0:2, 0:2]
        else:
            return None

    @property
    def enemy_sighted(self):
        return self.enemy_info[0:2, 0:2].any()

    @property
    def target_is_moving(self):
        if self.last_enemy_position is None:
            return False
        else:
            return (self.last_enemy_position[self.target_id] - self.enemy_position[self.target_id]).any()

    @property
    def close_to_target_enough(self):
        # TODO: use graph-based distance
        return np.linalg.norm(self.enemy_position[self.target_id] - self.self_position[self.self_id]) < CLOSE_THRESH

    state_methods = {'patrol': run_patrol, 'pursuit': run_pursuit, 'dps': run_dps, 'start': run_start}
