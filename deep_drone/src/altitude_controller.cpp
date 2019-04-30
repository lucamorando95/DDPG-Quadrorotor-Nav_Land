#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <stdio.h>
#include <iostream>
#include <cmath>


using namespace std;
double x_ground = 0.0;
double y_ground = 0.0;
double z_ground = 0.0;

double drone_x = 0.0;
double drone_y = 0.0;
double drone_z = 0.0;
double drone_ang_vel_z = 0.0;

double drone_Yaw = 0.0;
double drone_lin_vel_x = 0.0;
double drone_lin_vel_y = 0.0;
double  drone_lin_vel_z = 0.0;
double Mobile_orientation_theta = 0.0; 

bool flagMobileOdom = false;
bool flagDroneOdom = false;
bool flagDroneImu = false;
bool flagDroneFix_vel = false;

nav_msgs::Odometry odom;
nav_msgs::Odometry drone_odom;
sensor_msgs::Imu drone_imu;
geometry_msgs::Vector3Stamped drone_fix_vel;

void Box_odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
    odom = *msg;
    x_ground = odom.pose.pose.position.x;
    y_ground = odom.pose.pose.position.y;
    z_ground = odom.pose.pose.position.z;
    
// quaternion to RPY conversion
    tf::Quaternion q(
        odom.pose.pose.orientation.x,
        odom.pose.pose.orientation.y,
        odom.pose.pose.orientation.z,
        odom.pose.pose.orientation.w);
    tf::Matrix3x3 m(q);
    
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    
   // angular position
    Mobile_orientation_theta = yaw;
 
    flagMobileOdom = true;
}

void drone_odom_callback(const nav_msgs::Odometry::ConstPtr& msg){
     drone_odom = *msg;
     drone_x = drone_odom.pose.pose.position.x;
     drone_y = drone_odom.pose.pose.position.y;
     drone_z = drone_odom.pose.pose.position.z;
     
     // quaternion to RPY conversion
    tf::Quaternion q(
        drone_odom.pose.pose.orientation.x,
        drone_odom.pose.pose.orientation.y,
        drone_odom.pose.pose.orientation.z,
        drone_odom.pose.pose.orientation.w);
    tf::Matrix3x3 m(q);
    
    double drone_roll, drone_pitch, drone_yaw;
    m.getRPY(drone_roll, drone_pitch, drone_yaw);
    drone_Yaw = drone_yaw;
   // angular position
    flagDroneOdom = true;
    
}

void drone_Imu_callback(const sensor_msgs::Imu::ConstPtr& msg){
     drone_imu = *msg;
     drone_ang_vel_z = drone_imu.angular_velocity.z;
     flagDroneImu = true;
     
}

void drone_fix_Vel_callback(const geometry_msgs::Vector3Stamped::ConstPtr& msg){
     drone_fix_vel = *msg;
     drone_lin_vel_x = drone_fix_vel.vector.x;
     drone_lin_vel_y = drone_fix_vel.vector.y;
     drone_lin_vel_z = drone_fix_vel.vector.z;
     flagDroneFix_vel = true;
      
}



int main(int argc, char **argv) {
ros::init(argc, argv, "Altitude_controller");

ros::NodeHandle nh;

//Publish and Subscribers Topic
ros::Publisher vel = nh.advertise<geometry_msgs::Twist>("/drone/cmd_vel", 1);
ros::Subscriber odom_mobile_sub = nh.subscribe("odom",5,Box_odom_callback);
ros::Subscriber odom_drone_sub = nh.subscribe("/ground_truth/state",5,drone_odom_callback);
ros::Subscriber imu_drone_sub = nh.subscribe("/ardrone/imu",5,drone_Imu_callback);
ros::Subscriber vel_drone_sub = nh.subscribe("/fix_velocity",1,drone_fix_Vel_callback);

double distance_error = 0.0;
double velocity_error = 0.0;
double x_ground_old, y_ground_old, yaw_ground_old = 0.0;
double yaw_ground_dot = 0.0;
double vel_ground_x = 0.0;
double vel_ground_y = 0.0;
double Kp_z = 1.5;
double Kd_z = 0.3;
double Kp_yaw = 1.5;
double Kd_yaw = 0.3;
double K_prop_distance = 0.0;
double dt = 0.05;
double distance_treshold = 1.5;
double velocity_treshold = 0.3;
double yaw_error = 0.0;
double desired_nav_z = 1.8;
ros::Rate r(5);
while (nh.ok()) {
  
  geometry_msgs::Twist drone_vel_msg;
  
//Evaluate velocity of the ground target
  vel_ground_x = (x_ground - x_ground_old)/dt;
  vel_ground_y = (y_ground - y_ground_old)/dt;
  yaw_ground_dot = (Mobile_orientation_theta - yaw_ground_old)/dt;
  
//Evaluate planar distance error between drone frame and ground frame 
  distance_error = sqrt(pow(x_ground - drone_x,2) + pow(y_ground - drone_y,2));
  velocity_error = sqrt(pow(vel_ground_x - drone_lin_vel_x,2) + pow(vel_ground_y - drone_lin_vel_y,2));
  
  if (distance_error < distance_treshold && velocity_error < velocity_treshold){
      K_prop_distance =distance_treshold/distance_error;
     //Landing can be executed
      drone_vel_msg.linear.z = 0.3*K_prop_distance*(Kp_z*(0-drone_z) + Kd_z * (0 - drone_lin_vel_z));
      //Reducing Yaw error respect mobile platform
      drone_vel_msg.angular.z = Kp_yaw * (Mobile_orientation_theta - drone_Yaw) + Kd_yaw*(yaw_ground_dot - drone_ang_vel_z);
      cout << "Landing" << endl;
   }
  /* else if ( drone_z > 1.8)
   {
      drone_vel_msg.linear.z = -0.2; 
      drone_vel_msg.angular.z = 0.0;
      cout << "Drone Altitude higher than 1.8m" << endl;
   }*/
   else
   {
      cout << "Increasing altitude at 1.8 m for a safety navigation" << endl;
      drone_vel_msg.linear.z = Kp_z*(desired_nav_z-drone_z) + Kd_z * (0 - drone_lin_vel_z);
      drone_vel_msg.angular.z = Kp_yaw * (Mobile_orientation_theta - drone_Yaw) + Kd_yaw*(yaw_ground_dot - drone_ang_vel_z);
      
    }

// publish the message
    vel.publish(drone_vel_msg);
    flagMobileOdom = false; 
    flagDroneOdom = false;
    flagDroneImu = false;
    flagDroneFix_vel = false;

  x_ground_old = x_ground;
  y_ground_old = y_ground;
  yaw_ground_old = Mobile_orientation_theta;

    ros::spinOnce();
    r.sleep();
  }
}




