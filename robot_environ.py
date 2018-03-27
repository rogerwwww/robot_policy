import tensorflow as tf
import numpy as np
import random
import pygame
import pdb
scale = 0.1
robot_size = [int(500*scale), int(500*scale)]
act_dict = [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12]
d_size = [_/2 for _ in robot_size]

def _sign(x):
    if x>=0:
        return 1
    return -1

def create_raw_map_img():
    img = 255 * np.ones([int(8000*scale), int(5000*scale), 3], dtype=np.float32)
    bars = []
    # bar_1
    img[int(scale*1200):int(scale*2000), int(scale*1000):int(scale*1300), :] = 0
    img[int(scale*6000):int(scale*6800), int(scale*3700):int(scale*4000), :] = 0
    bars.append([int(scale*1200),int(scale*1000), int(scale*(2000-1200)), int(scale*(1300-1000))])
    bars.append([int(scale*6000),int(scale*3700), int(scale*(2000-1200)), int(scale*(1300-1000))])

    # bar_2
    img[int(scale*0):int(scale*800), int(scale*2500):int(scale*2800), :] = 0
    img[int(scale*7200):int(scale*8000), int(scale*2200):int(scale*2500), :] = 0
    bars.append([int(scale*0),int(scale*2500), int(scale*(800-0)), int(scale*(2800-2500))])
    bars.append([int(scale*7200),int(scale*2200), int(scale*(800-0)), int(scale*(2800-2500))])

    # bar_3
    img[int(scale*1800):int(scale*2100), int(scale*2300):int(scale*3500), :] = 0
    img[int(scale*5900):int(scale*6200), int(scale*1500):int(scale*2700), :] = 0
    bars.append([int(scale*1800),int(scale*2300), int(scale*(2100-1800)), int(scale*(3500-2300))])
    bars.append([int(scale*5900),int(scale*1500), int(scale*(2100-1800)), int(scale*(3500-2300))])

    # bar_4
    img[int(scale*3100):int(scale*3400), int(scale*3000):int(scale*5000), :] = 0
    img[int(scale*4600):int(scale*4900), int(scale*0):int(scale*2000), :] = 0
    bars.append([int(scale*3100),int(scale*3000), int(scale*(3400-3100)), int(scale*(5000-3000))])
    bars.append([int(scale*4600),int(scale*0), int(scale*(3400-3100)), int(scale*(5000-3000))])

    return img, bars

def draw_pos_tool(pos, color, map_img):
    for i in range(int(pos[0]-d_size[0]), int(pos[0]+d_size[0])):
        for j in range(int(pos[1]-d_size[1]), int(pos[1]+d_size[1])):
            map_img[i,j] = color
    return map_img

def draw_pos(pos_1, pos_2, raw_map_img):
    map_img = raw_map_img.copy()
    # red
    map_img = draw_pos_tool(pos_1[0][:2], np.array([255, 0, 0],dtype=np.float32), map_img)
    map_img = draw_pos_tool(pos_1[1][2:], np.array([255, 0, 0],dtype=np.float32), map_img)
    # blue
    map_img = draw_pos_tool(pos_2[0][:2], np.array([0, 255, 0],dtype=np.float32), map_img)
    map_img = draw_pos_tool(pos_2[1][2:], np.array([0, 255, 0],dtype=np.float32), map_img)
    return map_img

def _move_tool(pos_pre, dir, map_img):
    flag = False
    pos_new = [pos_pre[0]+dir[0], pos_pre[1]+dir[1]]
    corners = [[pos_new[0]+d_size[0], pos_new[1]+d_size[1]], [pos_new[0]+d_size[0], pos_new[1]-d_size[1]], [pos_new[0]-d_size[0], pos_new[1]+d_size[1]], [pos_new[0]-d_size[0], pos_new[1]-d_size[1]]]
    for corner in corners:
        if sum(map_img[int(corner[0]), int(corner[1])])<765:
            return flag
    return True

def move(pos_pre, dir, map_img):
    dir_new = dir

    # out
    while (pos_pre[0]+_sign(dir[0])*(abs(dir[0])+d_size[0])) < 0 or (pos_pre[0]+_sign(dir[0])*(abs(dir[0])+d_size[0]))>=int(scale*8000):
        if dir[0]==0:
            break
        dir[0] = _sign(dir[0]) * (abs(dir[0]) -1)
    while (pos_pre[1]+_sign(dir[1])*(abs(dir[1])+d_size[1])) < 0 or (pos_pre[1]+_sign(dir[1])*(abs(dir[1])+d_size[1]))>=int(scale*5000):
        if dir[1]==0:
            break
        dir[1] = _sign(dir[1]) * (abs(dir[1]) -1)

    # bar
    print('move to:', [pos_pre[0]+_sign(dir[0])*(abs(dir[0])+d_size[0]), pos_pre[1]+_sign(dir[1])*(abs(dir[1])+d_size[1])],)

    print(sum(map_img[int(pos_pre[0]+_sign(dir[0])*(abs(dir[0])+d_size[0])), int(pos_pre[1])]),
		sum(map_img[int(pos_pre[0]), int(pos_pre[1]+_sign(dir[1])*(abs(dir[1])+d_size[1]))]))
    print('dir',dir,)
    while not _move_tool(pos_pre, [dir[0], 0], map_img):
        if dir[0]==0:
            break
        dir[0] = _sign(dir[0]) * (abs(dir[0]) -1)
    while not _move_tool(pos_pre, [0, dir[1]], map_img):
        #pdb.set_trace()
        if dir[1]==0:
            break
        dir[1] = _sign(dir[1]) * (abs(dir[1]) -1)
    '''
    while sum(map_img[pos_pre[0]+_sign(dir[0])*(abs(dir[0])+d_size[0]), pos_pre[1]])<765:
        pdb.set_trace()
        if dir[0]==0:
            break
        dir[0] = _sign(dir[0]) * (abs(dir[0]) -1)
    while sum(map_img[pos_pre[0], pos_pre[1]+_sign(dir[1])*(abs(dir[1])+d_size[1])])<765:
        pdb.set_trace()
        if dir[1]==0:
            break
        dir[1] = _sign(dir[1]) * (abs(dir[1]) -1)
    '''
    print('dir_new',dir)
    return dir

def shoot(pos, target, state_1, state_2, map_img):
    flag = False
    dx   = int(robot_size[0]/10)
    dy   = int(robot_size[0]/10)
    targets = [[target[0]+dx, target[1]+dy], [target[0]+dx, target[1]-dy], [target[0]-dx, target[1]+dy], [target[0]-dx, target[1]-dy]]
    for tar in targets:
        for tmp_x in range(pos[0], tar[0]):
            tmp_y = pos[1] + int((tar[1]-pos[1])/(tar[0]-pos[0])) * (tmp_x-pos[0])
            if sum(map_img[int(tmp_x), int(tmp_y)]) < 2 * 255:
                return False, state_1, state_2
    state_1['hit'] = True
    state_2['shooted'] = True
    return True, state_1, state_2

def get_action(act, policy='MAX'):
    if policy=='MAX':
        return [act_dict[act[0]], act_dict[act[1]], act[-1]]
    #return [act_dict[random.randint(0,6)], act_dict[random.randint(0,6)], act_dict[random.randint(0,1)]]
    return [act_dict[0], act_dict[0], act_dict[random.randint(0,1)]]


def get_state(info_1, info_2, act_1, act_2, raw_map_img, policy='MAX'):
    # get state[x,y,dx,dy,shoot,hit,shooted]
    state_1 = []
    state_2 = []
    act_1 = [get_action(act_1[0], policy=policy), get_action(act_1[1], policy=policy)]
    act_2 = [get_action(act_2[0], policy=policy), get_action(act_2[1], policy=policy)]


    # move:[dx,dy]
    dir_tmp = move(info_1[0][:2], act_1[0][:2], raw_map_img)
    state_1.append({'x':info_1[0][0], 'y':info_1[0][1], 'dx':dir_tmp[0], 'dy':dir_tmp[1]})
    dir_tmp = move(info_1[1][:2], act_1[1][:2], raw_map_img)
    state_1.append({'x':info_1[1][0], 'y':info_1[1][1], 'dx':dir_tmp[0], 'dy':dir_tmp[1]})

    dir_tmp = move(info_2[0][:2], act_2[0][:2], raw_map_img)
    state_2.append({'x':info_2[0][0], 'y':info_2[0][1], 'dx':dir_tmp[0], 'dy':dir_tmp[1]})
    dir_tmp = move(info_2[1][:2], act_2[1][:2], raw_map_img)
    state_2.append({'x':info_2[1][0], 'y':info_2[1][1], 'dx':dir_tmp[0], 'dy':dir_tmp[1]})


    # shoot init
    if act_1[0][-1]:
        state_1[0]['shoot']= True
    else:
        state_1[0]['shoot']= False
    state_1[0]['hit']      = False
    state_1[0]['shooted']  = False
    if act_1[1][-1]:
        state_1[1]['shoot']= True
    else:
        state_1[1]['shoot']= False
    state_1[1]['hit']      = False
    state_1[1]['shooted']  = False

    if act_2[0][-1]:
        state_2[0]['shoot']= True
    else:
        state_2[0]['shoot']= False
    state_2[0]['hit']      = False
    state_2[0]['shooted']  = False
    if act_2[1][-1]:
        state_2[1]['shoot']= True
    else:
        state_2[1]['shoot']= False
    state_2[1]['hit']      = False
    state_2[1]['shooted']  = False
    

    # hit or shooted
    if state_1[0]['shoot']:
        shoot(info_1[0][:2], info_2[0][:2], state_1[0], state_2[0], raw_map_img)
        shoot(info_1[0][:2], info_2[1][:2], state_1[0], state_2[1], raw_map_img)
    if state_1[1]['shoot']:
        shoot(info_1[1][:2], info_2[0][:2], state_1[1], state_2[0], raw_map_img)
        shoot(info_1[1][:2], info_2[1][:2], state_1[1], state_2[1], raw_map_img)

    if state_2[0]['shoot']:
        shoot(info_2[0][:2], info_1[0][:2], state_2[0], state_1[0], raw_map_img)
        shoot(info_2[0][:2], info_1[1][:2], state_2[0], state_1[1], raw_map_img)
    if state_2[1]['shoot']:
        shoot(info_2[1][:2], info_1[0][:2], state_2[1], state_1[0], raw_map_img)
        shoot(info_2[1][:2], info_1[1][:2], state_2[1], state_1[1], raw_map_img)
    

    return state_1, state_2

def _get_reward(state):
    # move
    state['reward'] = 3 + np.sqrt((state['x']+state['dx']-4000*scale)**2 + (state['y']+state['dy']-2500*scale)**2) - np.sqrt((state['x']-4000*scale)**2 + (state['y']-2500*scale)**2)

    # shoot, hit or shooted
    if state['hit']:
        state['reward'] += 10.
    elif state['shoot'] and not state['hit']:
        state['reward'] += -1.

    if state['shooted']:
        state['reward'] += -10.
    return state

def get_reward(flag, info_1, info_2, act_1, act_2, raw_map_img, policy='MAX'):
    # init
    state_1, state_2 = get_state(info_1, info_2, act_1, act_2, raw_map_img, policy=policy)
    reward_1 = np.zeros([2,2 * flag.mov_num + 1])
    reward_2 = np.zeros([2,2 * flag.mov_num + 1])

    # reward
    state_1 = [_get_reward(state_1[0]), _get_reward(state_1[1])]
    state_2 = [_get_reward(state_2[0]), _get_reward(state_2[1])]

    reward_1[0,act_1[0][0]] = state_1[0]['reward']
    reward_1[0,act_1[0][1]+flag.mov_num] = state_1[0]['reward']
    if state_1[0]['hit']:
        reward_1[0,-1] = state_1[0]['reward']
    reward_1[1,act_1[0][0]] = state_1[1]['reward']
    reward_1[1,act_1[0][1]+flag.mov_num] = state_1[1]['reward']
    if state_1[1]['hit']:
        reward_1[1,-1] = state_1[1]['reward']

    reward_2[0,act_2[0][0]] = state_2[0]['reward']
    reward_2[0,act_2[0][1]+flag.mov_num] = state_2[0]['reward']
    if state_2[0]['hit']:
        reward_2[0,-1] = state_2[0]['reward']
    reward_2[1,act_2[0][0]] = state_2[1]['reward']
    reward_2[1,act_2[0][1]+flag.mov_num] = state_2[1]['reward']
    if state_2[1]['hit']:
        reward_2[1,-1] = state_2[1]['reward']

    return state_1, state_2, reward_1, reward_2

def get_new_info(info_pre, state):
    info_pre[0] += state['dx']
    info_pre[1] += state['dy']
    if state['shooted']:
        info_pre[2] -= 1
    return info_pre

def get_init():
    info_1 = [[30, 30, 100, 0],  [30, 470, 100, 0]]
    info_2 = [[770, 30, 100,0 ], [770, 470, 100, 0]]
    raw_map_img, _ = create_raw_map_img()

    map_img_new = draw_pos(info_1, info_2, raw_map_img)
    return info_1, info_2, map_img_new

def environ(flag, info_1, info_2, act_1_p, act_2_p, raw_map_img, policy='MAX'):
    # info[2,4]: [x,y,blood,buff]
    # act[2,15]:  [3,2,1,0,-1,-2,-3,3,2,1,0,-1,-2,-3,shoot]

    print('info:')
    print(info_1)
    print(info_2)

    act_1 = []
    act_1.append([np.argmax(act_1_p[0,:flag.mov_num]), np.argmax(act_1_p[0,flag.mov_num:2*flag.mov_num]), int(act_1_p[0,-1]>0.5)])
    act_1.append([np.argmax(act_1_p[1,:flag.mov_num]), np.argmax(act_1_p[1,flag.mov_num:2*flag.mov_num]), int(act_1_p[1,-1]>0.5)])

    act_2 = []
    act_2.append([np.argmax(act_2_p[0,:flag.mov_num]), np.argmax(act_2_p[0,flag.mov_num:2*flag.mov_num]), int(act_2_p[0,-1]>0.5)])
    act_2.append([np.argmax(act_2_p[1,:flag.mov_num]), np.argmax(act_2_p[1,flag.mov_num:2*flag.mov_num]), int(act_2_p[1,-1]>0.5)])
    '''
    print '\nact:'
    print act_1
    print act_2
    '''

    # pos+pre
    map_img = draw_pos(info_1, info_2, raw_map_img)

    # state and reward
    state_1, state_2, reward_1, reward_2 = get_reward(flag, info_1, info_2, act_1, act_2, raw_map_img, policy=policy)
    #pdb.set_trace()

    print('\nstate:')
    print(state_1)
    print(state_2)
    '''

    print '\nreward:'
    print reward_1
    print reward_2
    '''

    # new info and map
    info_1_new = [get_new_info(info_1[0], state_1[0]), get_new_info(info_1[1], state_1[1])]
    info_2_new = [get_new_info(info_2[0], state_2[0]), get_new_info(info_2[1], state_2[1])]
    '''
    print 'info_1_new', info_1_new
    print 'info_2_new', info_2_new
    '''
    map_img_new = draw_pos(info_1_new, info_2_new, raw_map_img)


    return info_1_new, info_2_new, reward_1, reward_2, map_img_new