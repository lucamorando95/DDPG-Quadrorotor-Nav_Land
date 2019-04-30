#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace std;

  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  
  double x_old = 0.0;
  double y_old = 0.0;
  double orientation_theta = 0.0;
  double orientation_theta_old = 0.0;
  bool flagOdom = false;
  
  nav_msgs::Odometry odom;

void Box_odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
    odom = *msg;
    x = odom.pose.pose.position.x;
    y = odom.pose.pose.position.y;
    z = odom.pose.pose.position.z;
    
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
    orientation_theta = yaw;
    
    
    
    flagOdom = true;
}


int main(int argc, char **argv) {
  ros::init(argc, argv, "velocity_publisher");

  ros::NodeHandle nh;

  ros::Publisher vel = nh.advertise<geometry_msgs::Twist>("/mobile/cmd_vel", 1);
  ros::Subscriber odom_sub = nh.subscribe("odom",5,Box_odom_callback);
  
  double des_x = 2.0;
  double des_y = 3.0;
  double distance = 0.0;
 
  double dt = 0.05;
  double theta_des = 0.0;
  double theta_dot = 0.0;
  double Kp_theta = 2.5;
  double Kd_theta = 0.3;
  double theta_err = 0.0;
  double Kp_x = 0.9;
  double Kd_x = 0.3;
  double Kp_y = 0.9;
  double Kd_y = 0.3;
  double x_err = 0.0;
  double x_dot = 0.0;
  double y_dot = 0.0;
  double y_err = 0.0;
  double vx = 0.3;
  double vy = 0.0;
  double vz = 0.0;
  
  bool eval_direction = false;
 
  
  ros::Time current_time, last_time;
  
  
  
  ros::Rate r(10);
  while (nh.ok()) {
  current_time = ros::Time::now();
  geometry_msgs::Twist msg;

  /*cout<<"des_x - x: " << des_x - x << endl;
  cout<<"des_y - y: " << des_y - y << endl;
  cout<<"theta_des - orientation_theta: " << theta_des - orientation_theta << endl; */
   
 if (eval_direction == false){
      distance = sqrt(pow(des_x - x, 2) + pow(des_y - y, 2));
      theta_des = ((des_x - x)/distance) * acos(orientation_theta);
      eval_direction = true;  
      cout << "theta_des" << theta_des << endl;
}
    if (des_x - x > 0.1 && des_y - y > 0.1){
        if (theta_des - orientation_theta > 0.01 ) {

           //eval_direction = true; 
        
         
           //PID control theta orientation 
           theta_err = theta_des - orientation_theta;
           theta_dot = (orientation_theta - orientation_theta_old)/dt;
           msg.angular.z = Kp_theta * theta_err + Kd_theta * theta_dot;
         }
        else
         {
            cout << "Desired Theta Orientation reached " << endl;
             //PID control Position 
            x_err = des_x - x;
            x_dot = (x - x_old)/dt;
            msg.linear.x = Kp_x * x_err + Kd_x*x_dot;
            
            y_err = des_y - y;
            y_dot = (y - y_old)/dt;
            msg.linear.y = Kp_y * y_err + Kd_y*y_dot;
            cout << "x_err " << x_err << endl;
            cout << "y_err " << y_err << endl;
            if (x_err < 0.3 && y_err < 0.3){
               cout << "Desired Position reached " << endl;
            }

         }
    }
    else {
      
         msg.linear.x = 0.0;
         msg.linear.y = 0.0;
         msg.linear.z = 0.0;

         msg.angular.x = 0.0;
         msg.angular.y = 0.0;
         msg.angular.z = 0.0;

     }
     


    // next, we'll publish the cmd_vel message over ROS
   
    // publish the message
    vel.publish(msg);
    flagOdom = false;

    //Update variables 
    orientation_theta_old = orientation_theta;
    x_old = x;
    y_old = y;
    last_time = ros::Time::now();
    
    //dt = last_time - current_time;
    ros::spinOnce();
    r.sleep();
  }
}
