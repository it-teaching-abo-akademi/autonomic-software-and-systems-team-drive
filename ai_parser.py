#!/usr/bin/env python

import glob
import os
import sys

import numpy as np
import math

from utils import PathGraph

try:
    sys.path.append(glob.glob('**/**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import weakref
import carla
import ai_knowledge as data


# Monitor is responsible for reading the data from the sensors and telling it to the knowledge
# TODO: Implement other sensors (lidar and depth sensors mainly)
# TODO: Use carla API to read whether car is at traffic lights and their status, update it into knowledge
class Monitor(object):
  def __init__(self, knowledge,vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
    weak_self = weakref.ref(self)

    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)

    # Debugging access
    self.knowledge.update_data('world', self.vehicle.get_world())

    # Get map
    self.knowledge.update_data('map', self.vehicle.get_world().get_map())
    self.knowledge.update_data('road_graph', self.parse_map())

    # Get vehicle information
    self.knowledge.update_data('bounding_box', self.vehicle.bounding_box)

    # Init params
    self.knowledge.update_data("integral", 0)
    self.knowledge.update_data("previous_error", 0)
    self.knowledge.update_data("throttle_previous", 0)
    
    self.knowledge.update_data("velocity", 0)
    self.knowledge.update_data("look_ahead", None)
    self.knowledge.update_data("target_speed", None)
    self.knowledge.update_data('lidar',None)

    # Set up lidar
    zd = self.vehicle.bounding_box.extent.z*2.1
    xd = self.vehicle.bounding_box.extent.x+1.5

    angle =  math.degrees(math.tanh(zd/xd))
    dist = math.sqrt(xd**2+zd**2)*100
  
    lidar = self.vehicle.get_world().get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar.set_attribute('points_per_second', str(200))
    lidar.set_attribute('channels', str(2))
    lidar.set_attribute('range', str(dist))
    lidar.set_attribute('upper_fov', str(angle))
    lidar.set_attribute('lower_fov', str(-angle))
    lidar.set_attribute('sensor_tick', '0.5')
    lidar_location =carla.Location(z=self.vehicle.bounding_box.extent.z*2.1) # Car height can change
    self.knowledge.update_data('lidar_location',lidar_location)
    self.sensor = self.vehicle.get_world().spawn_actor(lidar,carla.Transform(lidar_location), attach_to=self.vehicle)
    
    self.knowledge.update_data('lidar',None)
    self.sensor.listen(lambda event: Monitor._lidar_update(weak_self, event))


  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):

    # Update the position of vehicle into knowledge
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)
    
    # Update velocity
    self.knowledge.update_data("velocity", self.vehicle.get_velocity())

    # Update traffic light data
    self.knowledge.update_data('at_traffic_light', self.vehicle.is_at_traffic_light())
    self.knowledge.update_data('traffic_light_state', self.vehicle.get_traffic_light_state())

  def parse_map(self):
    print("Parsing map")
    graph = PathGraph()

    for start_waypoint, end_waypoint in self.knowledge.get_map().get_topology():
      
      # Node names are made up of road_id - lane_id
      start_name = str(start_waypoint.road_id) + "-" + str(start_waypoint.lane_id)
      end_name = str(end_waypoint.road_id) + "-" + str(end_waypoint.lane_id)

      start_location = start_waypoint.transform.location
      end_location = end_waypoint.transform.location

      graph.add_edge(start_name,end_name ,abs(start_location.distance(end_location)),start_waypoint, end_waypoint)

      for waypoint in [start_waypoint, end_waypoint]:
        waypoint_name = str(waypoint.road_id) + "-" + str(waypoint.lane_id)
        # Assumption: Max one lane change
        if waypoint.lane_change.name is not "None":
          lanes = []
          if waypoint.lane_change == carla.LaneChange.Left:
            lane = waypoint.get_left_lane()
            lanes.append(lane)
          elif waypoint.lane_change == carla.LaneChange.Right:
            # Causes segementation fault
            #lane = waypoint.get_right_lane()
            #lanes.append(lane)
            pass
          elif waypoint.lane_change == carla.LaneChange.Both:
            left = waypoint.get_left_lane()
            right = waypoint.get_right_lane()
            lanes.extend([left, right])
 
          for lane in lanes:
            if lane is not None and lane.lane_type == "driving":
              lane_location = lane.transform.location
              lane_name = str(lane.road_id) + "-" + str(lane.lane_id)
              graph.add_edge(waypoint_name,lane_name ,abs(waypoint.transform.location.distance(lane_location)),waypoint, lane)
              graph.add_edge(lane_name, waypoint_name ,abs(waypoint.transform.location.distance(lane_location)),lane,waypoint)
           
              
              
              if self.knowledge.DEBUG:
                pass
                #self.knowledge.get_world().debug.draw_line(waypoint.transform.location,lane_location,color=carla.Color(r=0, g=255, b=0), life_time=120.0)  
       

      if self.knowledge.DEBUG:
        pass
        # For debugging
        #text = "road id = %d, lane id = %d, transform = %s"
        #print(text % (start_waypoint.road_id, start_waypoint.lane_id, start_waypoint.transform))
        #print(text % (end_waypoint.road_id, end_waypoint.lane_id, end_waypoint.transform))
        """
        self.knowledge.get_world().debug.draw_line(start_location,end_location,
        color=carla.Color(r=255, g=0, b=0), life_time=120.0)  

        self.knowledge.get_world().debug.draw_point(start_location,
        color=carla.Color(r=0, g=255, b=255), life_time=120.0)

        self.knowledge.get_world().debug.draw_point(end_location,
        color=carla.Color(r=255, g=255, b=0), life_time=120.0)
        """
    return graph





  @staticmethod
  def _on_invasion(weak_self, event):
    self = weak_self()
    if not self:
      return
    self.knowledge.update_data('lane_invasion',event.crossed_lane_markings)

  @staticmethod
  def _lidar_update(weak_self, event):
    self = weak_self()
    if not self:
      return
    self.knowledge.update_data('lidar',event)

# Analyser is responsible for parsing all the data that the knowledge has received from Monitor and turning it into something usable
# TODO: During the update step parse the data inside knowledge into information that could be used by planner to plan the route
class Analyser(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge

  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    
    self.parse_lidar()

    return

  def parse_lidar(self):

    lidar = self.knowledge.get_lidar()
  
    if lidar is not None:
      bounding_box = self.knowledge.get_bounding_box()
      location = self.knowledge.get_location()
      lidar_location = self.knowledge.get_lidar_location()
      
      sensor_position = carla.Location(location.x+lidar_location.x, location.y+lidar_location.y,lidar_location.z+location.z)
      
      if self.knowledge.DEBUG:
        self.knowledge.get_world().debug.draw_point(sensor_position,
        color=carla.Color(r=255, g=0, b=0), life_time=1.0) 
      


      detections = []

      if not lidar == None:     
        for point in lidar:          
          relative_location = carla.Location(location.x+point.x, location.y+point.y, 1.5+bounding_box.extent.z+location.z+point.z) # 3 => Lidar height

          valid_x = relative_location.x > bounding_box.extent.x  and relative_location.x < -bounding_box.extent.x
          valid_y = relative_location.y > bounding_box.extent.y  and relative_location.y < -bounding_box.extent.y
          valid_z = relative_location.z > 0.5
         
          # Add only valid points
          if valid_x and valid_y and valid_z and self.knowledge.distance(relative_location, location) < 3: # 3 =>  Max distance 
            detections.append(point)
            
          if self.knowledge.DEBUG:
            # Draw where the lidar points are close
            self.knowledge.get_world().debug.draw_line(relative_location,sensor_position,
            color=carla.Color(r=0, g=255, b=0), life_time=1.0) 
      
      self.knowledge.update_data('lidar_close', detections)

  