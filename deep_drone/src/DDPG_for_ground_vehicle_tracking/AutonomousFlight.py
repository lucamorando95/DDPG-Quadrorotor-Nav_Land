#!/usr/bin/env python
import rospy 
import time
from geometry_msgs.msg import Twist,PoseWithCovariance
from std_msgs.msg import String 
from std_msgs.msg import Empty 
from nav_msgs.msg import Odometry

COMMAND_PERIOD = 1000

class AutonomousFlight():
    def __init__(self):
        self.status = ""
        rospy.init_node('forward', anonymous=False)
        self.rate = rospy.Rate(10)
        self.pubTakeoff = rospy.Publisher("ardrone/takeoff",Empty, queue_size=10)
        self.pubLand = rospy.Publisher("ardrone/land",Empty, queue_size=10)
        self.pubCommand = rospy.Publisher('cmd_vel',Twist, queue_size=10)
        self.command = Twist()
        #self.commandTimer = rospy.Timer(rospy.Duration(COMMAND_PERIOD/1000.0),self.SendCommand)
        self.state_change_time = rospy.Time.now()  
        self.flag = False
        rospy.on_shutdown(self.SendLand)

    def SendTakeOff(self, TakeOff, uav):
        take_off_z = 1.5
        self.pubTakeoff.publish(Empty())
         
        try:
            poseData = rospy.wait_for_message(
                         '/ground_truth/state', Odometry, timeout=5)
            
                    
        except:
            rospy.loginfo(
                         "Current drone pose not ready yet, retrying to get robot pose")
        try:
               altitudeVelDrone = rospy.wait_for_message('/drone/cmd_vel', Twist, timeout=5)
            
                    
        except:
               rospy.loginfo(
                         "Current drone pose not ready yet, retrying to get robot pose")

        if poseData.pose.pose.position.z < take_off_z :
            print("altitude",poseData.pose.pose.position.z)
            print("yaw",altitudeVelDrone.angular.z)
            uav.SetCommand(0,0,1,0,0,altitudeVelDrone.angular.z)
            TakeOff = False
                
        else:
            uav.SetCommand(0,0,0,0,0,0)
            TakeOff = True
            self.start = time.time()
            print('######Waiting before starting the task ############')
            while self.flag == False:
                end = time.time()
                time_passed = end - self.start 
                if time_passed > 4:
                   self.flag = True
                   
        self.rate.sleep()
        
        
        self.flag = False
        return TakeOff
        
  
            
    def SendLand(self):
        self.pubLand.publish(Empty())
    
        
    def SetCommand(self, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z):
        
        self.command.linear.x = linear_x
        self.command.linear.y = linear_y
        self.command.linear.z = linear_z
        self.command.angular.x = angular_x
        self.command.angular.y = angular_y
        self.command.angular.z = angular_z
        self.pubCommand.publish(self.command)
        self.rate.sleep()

