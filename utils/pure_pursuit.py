"""
Path tracking simulation with pure pursuit steering control and PID speed control.
author: Atsushi Sakai (@Atsushi_twi)
"""
import numpy as np
import math
import matplotlib.pyplot as plt

class PurePursuit:

    def __init__(self, k,Lfc, kp, L):
        self.k = k # look forward gain
        self.Lfc = Lfc # look-ahead distance
        self.kp = kp # speed proportional gain
        self.L = L #Vehicle wheelbase [m]

    def update(target_speed, coordinates, x,y,v, yaw){
        
        rear_x = x - ((L / 2) * math.cos(yaw))
        rear_y = y - ((L / 2) * math.sin(yaw))

        lastIndex = len(cx) - 1

        if(lastIndex > target_ind):
            ai = PIDControl(target_speed, v)
            di, target_ind = pure_pursuit_control(rear_x, rear_y, cx, cy, target_ind)
            return convert(ai, di)
            
        else:
            return (0, 0)

        return 
    }


    def PIDControl(self, target, current):
        a = self.Kp * (target - current)

        return a


    def pure_pursuit_control(v, yaw, rear_x,rear_y, cx,cy, pind):

        ind = calc_target_index(rear_x,rear_y, cx, cy)

        if pind >= ind:
            ind = pind

        if ind < len(cx):
            tx = cx[ind]
            ty = cy[ind]
        else:
            tx = cx[-1]
            ty = cy[-1]
            ind = len(cx) - 1

        alpha = math.atan2(ty - rear_y, tx - rear_x) - yaw

        self..Lf = self.k * v + self.Lfc

        delta = math.atan2(2.0 * L * math.sin(alpha) / self.Lf, 1.0)

        return delta, ind

    def calc_distance(rear_x, rear_y, point_x, point_y):

        dx = rear_x - point_x
        dy = rear_y - point_y
        return math.sqrt(dx ** 2 + dy ** 2)


    def calc_target_index(rear_x,rear_y, cx, cy):

        # search nearest point index
        dx = [rear_x - icx for icx in cx]
        dy = [rear_y - icy for icy in cy]
        d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
        ind = d.index(min(d))
        old_nearest_point_index = ind
      
        self.L = 0.0

        self.Lf = k * state.v + Lfc

        # search look ahead target point index
        while self.Lf > self.L and (ind + 1) < len(cx):
            dx = cx[ind] - rear_x
            dy = cy[ind] - rear_y
            self.L = math.sqrt(dx ** 2 + dy ** 2)
            ind += 1

        return ind