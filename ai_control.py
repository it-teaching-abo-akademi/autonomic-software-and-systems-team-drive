#!/usr/bin/env python

import glob
import os
import sys
from collections import deque
import math
import numpy as np

try:
    sys.path.append(glob.glob('**/**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import ai_knowledge as data
from ai_knowledge import Status

# Executor is responsible for moving the vehicle around
# In this implementation it only needs to match the steering and speed so that we arrive at provided waypoints
# BONUS TODO: implement different speed limits so that planner would also provide speed target speed in addition to direction
class Executor(object):
  def __init__(self, knowledge, vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
    self.target_pos = knowledge.get_location()
    
  #Update the executor at some intervals to steer the car in desired direction
  def update(self, time_elapsed):
    status = self.knowledge.get_status()
    #TODO: this needs to be able to handle
    if status == Status.DRIVING:
      #dest = self.knowledge.get_current_destination()

      target_speed = self.knowledge.get_target_speed()
      look_ahead = self.knowledge.get_look_ahead()

      self.update_control(None, {"target_speed": target_speed, "look_ahead": look_ahead }, time_elapsed)

  # TODO: steer in the direction of destination and throttle or brake depending on how close we are to destination
  # TODO: Take into account that exiting the crash site could also be done in reverse, so there might need to be additional data passed between planner and executor, or there needs to be some way to tell this that it is ok to drive in reverse during HEALING and CRASHED states. An example is additional_vars, that could be a list with parameters that can tell us which things we can do (for example going in reverse)
  def update_control(self, destination, additional_vars, delta_time):

    look_ahead = additional_vars["look_ahead"]
    target_speed = additional_vars["target_speed"]
    
    forward = self.knowledge.get_rotation().get_forward_vector()
    velocity = self.knowledge.get_velocity()
    current_speed = np.linalg.norm(np.dot(np.array([velocity.x, velocity.y]),np.array([forward.x, forward.y])))

    if look_ahead is not None:
      # calculate throttle and steer
      throttle = self.calculate_throttle(target_speed, current_speed, delta_time)
      steer = self.calculate_steer(look_ahead,current_speed)
      
      control = carla.VehicleControl()
      control.throttle = throttle
      control.steer = steer
      control.brake = 0.0 #np.fmax(np.fmin(0.0, 1.0), 0.0)
      control.hand_brake = False
      self.vehicle.apply_control(control)

  def calculate_throttle(self, target_speed, speed, delta_time):
      delta_time = delta_time/1000
      # Pid throttle control
      kp = 1
      ki = 1
      kd = 0.01

      # error term
      delta_v = target_speed - speed

      # I
      integral = self.knowledge.get_integral() + delta_v * delta_time

      # D
      derivate = (delta_v - self.knowledge.get_previous_error()) / delta_time

      result = kp * delta_v + ki * integral + kd * derivate
   
      if result > 0:
        throttle_output = np.tanh(result)
     
        throttle_output = max(0.0, min(1.0, throttle_output))
        if throttle_output - self.knowledge.get_throttle_previous() > 0.1:
            throttle_output = self.knowledge.get_throttle_previous() + 0.1
    
      else:
          throttle_output = 0

      self.knowledge.update_data("integral", integral)
      self.knowledge.update_data("previous_error", delta_v)
      self.knowledge.update_data("throttle_previous", throttle_output)

      throttle = np.fmax(np.fmin(throttle_output, 1.0), 0.0)

      return throttle

  def calculate_steer(self, waypoints, speed):
 
    # Stanley controller for lateral control
    k_e = 0.3
    k_v = 10

    rad_to_steer = 180.0 / 70.0 / np.pi

    # Calculate heading error
    yaw_path = np.arctan2(waypoints[-1][1]-waypoints[0][1], waypoints[-1][0]-waypoints[0][0])
    yaw_diff = yaw_path - math.radians(self.knowledge.get_rotation().yaw)

    if yaw_diff > np.pi:
        yaw_diff -= 2 * np.pi
    if yaw_diff < - np.pi:
        yaw_diff += 2 * np.pi
  
    heading = self.knowledge.get_rotation().get_forward_vector()
    x = self.knowledge.get_location().x + self.knowledge.get_bounding_box().extent.x * heading.x
    y = self.knowledge.get_location().y + self.knowledge.get_bounding_box().extent.y * heading.y

    if self.knowledge.DEBUG:
      self.knowledge.get_world().debug.draw_point(carla.Location(x,y,4),
              color=carla.Color(r=0, g=255, b=255), life_time=1.0) 

    current_xy = np.array([x, y])
    crosstrack_error = np.min(np.sum((current_xy - np.array(waypoints)[:, :2])**2, axis=1))

    yaw_cross_track = np.arctan2(y-waypoints[0][1], x-waypoints[0][0])
    yaw_path2ct = yaw_path - yaw_cross_track
    if yaw_path2ct > np.pi:
        yaw_path2ct -= 2 * np.pi
    if yaw_path2ct < - np.pi:
        yaw_path2ct += 2 * np.pi
    if yaw_path2ct > 0:
        crosstrack_error = abs(crosstrack_error)
    else:
        crosstrack_error = - abs(crosstrack_error)

    yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + speed))
    
    # print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)

    # Control low
    steer_expect = yaw_diff + yaw_diff_crosstrack
    if steer_expect > np.pi:
        steer_expect -= 2 * np.pi
    if steer_expect < - np.pi:
        steer_expect += 2 * np.pi
    steer_expect = min(1.22, steer_expect)
    steer_expect = max(-1.22, steer_expect)

    input_steer = rad_to_steer * steer_expect
    steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)

    return steer


# Planner is responsible for creating a plan for moving around
# In our case it creates a list of waypoints to follow so that vehicle arrives at destination
# Alternatively this can also provide a list of waypoints to try avoid crashing or 'uncrash' itself
class Planner(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge
    self.path = deque([])
    self.wait = 100

  # Create a map of waypoints to follow to the destination and save it
  def make_plan(self, source, destination):
    self.path = self.build_path(source,destination)
    self.update_plan()
    #self.knowledge.update_destination(self.get_current_destination())
  
  # Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):

      self.update_plan()
      self.knowledge.update_destination(self.get_current_destination())
  
  #Update internal state to make sure that there are waypoints to follow and that we have not arrived yet
  def update_plan(self):
    prev = self.knowledge.get_location()
    while len(self.path) > 0 and self.knowledge.arrived_at(self.path[0]):

      if self.knowledge.DEBUG:
        self.knowledge.get_world().debug.draw_point(self.path[0], color=carla.Color(r=0, g=0, b=255), life_time=20.0)

      prev = self.path.popleft()

    if len(self.path) > 0:
      look_ahead = [(prev.x,prev.y, 20),(self.path[0].x,self.path[0].y, 20)] # Waypoint speed limit
      self.knowledge.update_data("look_ahead", look_ahead)
    else:
      self.knowledge.update_data("look_ahead", None)

  #get current destination 
  def get_current_destination(self):
    status = self.knowledge.get_status()
    #if we are driving, then the current destination is next waypoint
    if status == Status.DRIVING:
      #TODO: Take into account traffic lights and other cars
      if self.path is not None and len(self.path) > 0:
        if self.knowledge.get_at_traffic_light() and self.knowledge.traffic_light_state() == carla.TrafficLightState.Red:
          self.knowledge.update_data("target_speed", 0)
          print("At red light")
        else:
          self.knowledge.update_data("target_speed", 3)
      else:
        self.knowledge.set_status(Status.ARRIVED)
        print("Arrived at Destination")
      return self.knowledge.get_location()
    if status == Status.ARRIVED:
      return self.knowledge.get_location()
    if status == Status.HEALING:
      #TODO: Implement crash handling. Probably needs to be done by following waypoint list to exit the crash site.
      #Afterwards needs to remake the path.
      ##
      return self.knowledge.get_location()
    if status == Status.CRASHED:
      #TODO: implement function for crash handling, should provide map of wayoints to move towards to for exiting crash state. 
      #You should use separate waypoint list for that, to not mess with the original path. 
      return self.knowledge.get_location()
    #otherwise destination is same as current position
    return self.knowledge.get_location()

  #TODO: Implementation
  def build_path(self, source, destination):
    #TODO: create path of waypoints from source to

    graph = self.knowledge.get_road_graph()
    
    source_waypoint = self.knowledge.get_map().get_waypoint(source.location, project_to_road=True)
    destination_waypoint = self.knowledge.get_map().get_waypoint(destination, project_to_road=True)

    # Generate road order from topology 

    source_node = str(source_waypoint.road_id) + "-" + str(source_waypoint.lane_id)
    destination_node = str(destination_waypoint.road_id) + "-" + str(destination_waypoint.lane_id)

    waypoints =  graph.dijsktra(source_node, destination_node)
    print("Road waypoints", len(waypoints))
    
    # Generate detailed path using waypoint.next()
    points = []
    
    edge_index = 0
    dist = 0
    change = 1 # Waypoint distance
    
    while(edge_index < len(waypoints)):
      
      current_waypoint = waypoints[edge_index][0]
      next_waypoint = waypoints[edge_index][1]
      if current_waypoint.road_id == next_waypoint.road_id:
        edge_index +=1
        dist = 0
      else:
        loc = list(current_waypoint.next(dist*change))
    
        if loc[0] == None:
          waypoints = []
          print("No path available")
          break
        else:
          max_idx = 0
          max_value = 0

          for i in range(len(loc)):
            value = self.knowledge.distance(current_waypoint.transform.location, loc[i].transform.location)
            if value >= max_value:
              max_value = value
              max_idx = i

          if self.knowledge.distance(loc[i].transform.location, next_waypoint.transform.location) < 2*change:
            points.append(loc[max_idx].transform.location)
            edge_index +=1
            dist = 0
          else:
            points.append(loc[max_idx].transform.location)
            dist += 1

    if len(waypoints) > 0:
      # Interpolate last distance to destination

      xs = [waypoints[-1][1].transform.location.x, destination.x]
      ys = [waypoints[-1][1].transform.location.y, destination.y]

      xx = np.linspace(xs[0], xs[1], num=10)
      yy = np.interp(xx, xs, ys)

      pe = [carla.Location(xx[i],yy[i],1) for i in range(len(xx))]
      points.extend(pe)

      # Interpolate start location to source

      xs = [source.location.x, waypoints[0][0].transform.location.x]
      ys = [source.location.y, waypoints[0][0].transform.location.y]

      xx = np.linspace(xs[0], xs[1], num=10)
      yy = np.interp(xx, xs, ys)

      pe = [carla.Location(xx[i],yy[i],1) for i in range(len(xx))]
      point = pe.extend(points)
      
      self.knowledge.get_world().debug.draw_point(destination, color=carla.Color(r=0, g=0, b=0), life_time=20.0)
      
      for w1, w2 in waypoints:
        
        self.knowledge.get_world().debug.draw_line(w1.transform.location, w2.transform.location,
        color=carla.Color(r=0, g=255, b=255), life_time=120.0)
        self.knowledge.get_world().debug.draw_point(destination_waypoint.transform.location, color=carla.Color(r=255, g=255, b=255), life_time=20.0)

      if self.knowledge.DEBUG:
        for p in points:
          self.knowledge.get_world().debug.draw_point(p, color=carla.Color(r=255, g=0, b=0), life_time=20.0)

    return deque(points)


