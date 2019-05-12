#!/usr/bin/env python

import glob
import os
import sys

import numpy as np

from utils.dijsktra import Graph

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
    self.knowledge.update_data("velocity", 0)
    self.knowledge.update_data("waypoints", None)

    # Debugging access
    self.knowledge.update_data('world', self.vehicle.get_world())

    # Get map
    self.knowledge.update_data('map', self.vehicle.get_world().get_map())
    self.knowledge.update_data('road_graph', self.parse_map())

    # Get vehicle information
    self.knowledge.update_data('bounding_box', self.vehicle.bounding_box)

    # Init control parms
    self.knowledge.update_data("integral", 0)
    self.knowledge.update_data("previous_error", 0)
    self.knowledge.update_data("throttle_previous", 0)

    """
    # Set up lidar
    lidar = self.vehicle.get_world().get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar.set_attribute('range', '500')
    lidar.set_attribute('points_per_second', '10000')
    lidar.set_attribute('channels', '8')
    lidar.set_attribute('upper_fov', '-10')
    lidar.set_attribute('lower_fov', '-55')
    lidar.set_attribute('sensor_tick', '30')
    transform = carla.Transform(carla.Location(z=3))
    #self.sensor = self.vehicle.get_world().spawn_actor(lidar, transform, attach_to=self.vehicle)
    """
    self.knowledge.update_data('lidar',None)
   # self.sensor.listen(lambda event: Monitor._lidar_update(weak_self, event))

  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):


    # Update the position of vehicle into knowledge
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)
    
    # Update velocity
    velocity = self.vehicle.get_velocity()
    forward = self.vehicle.get_transform().get_forward_vector()
    speed = np.linalg.norm(np.dot(np.array([velocity.x, velocity.y, velocity.z]),np.array([forward.x, forward.y, forward.z])))
    self.knowledge.update_data('velocity', speed)

    # Update traffic light data
    self.knowledge.update_data('at_traffic_light', self.vehicle.is_at_traffic_light())
    self.knowledge.update_data('traffic_light_state', self.vehicle.get_traffic_light_state())

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

  def parse_map(self):

    graph = Graph()

    for start_waypoint, end_waypoint in self.knowledge.get_map().get_topology():

      start_location = start_waypoint.transform.location
      end_location = end_waypoint.transform.location

      start_name = str(start_waypoint.road_id) + "-" + str(start_waypoint.lane_id)
      end_name = str(end_waypoint.road_id) + "-" + str(end_waypoint.lane_id)

      graph.add_edge(start_name,end_name ,abs(start_location.distance(end_location)),start_waypoint, end_waypoint)
      
      # For debugging

      text = "road id = %d, lane id = %d, transform = %s"
      #print(text % (start_waypoint.road_id, start_waypoint.lane_id, start_waypoint.transform))
      #print(text % (end_waypoint.road_id, end_waypoint.lane_id, end_waypoint.transform))
 
      self.knowledge.get_world().debug.draw_line(start_location,end_location,
      color=carla.Color(r=255, g=0, b=0), life_time=120.0)  

      self.knowledge.get_world().debug.draw_point(start_location,
      color=carla.Color(r=0, g=255, b=255), life_time=120.0)

      self.knowledge.get_world().debug.draw_point(end_location,
      color=carla.Color(r=255, g=255, b=0), life_time=120.0)
 
    return graph


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

      location = self.knowledge.get_location()
      bounding_box = self.knowledge.get_bounding_box()

      detections = []

      if not lidar == 0:     
        for point in lidar:          
          relative_location = carla.Location(location.x+point.y, location.y+point.x*-1, 3+location.z+point.z*-1)

          # Remove ground and car boundary from detections
          if relative_location.z > 0.5 and point.x > bounding_box.extent.x and self.knowledge.distance(relative_location, location) < 3:
            detections.append(point)
            
            # Draw where the lidar points are close
            self.knowledge.get_world().debug.draw_point(relative_location,
            color=carla.Color(r=0, g=255, b=0), life_time=1.0) 
      
      self.knowledge.update_data('lidar_close', detections)



