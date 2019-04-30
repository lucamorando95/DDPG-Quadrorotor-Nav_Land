
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <stdio.h>
#include <cmath>

//using namespace std;

  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  double x_old = 0.0;
  double y_old = 0.0;
  double orientation_theta = 0.0;
  double orientation_theta_old = 0.0;
  
  double drone_Yaw = 0.0;
  double drone_ang_vel_z = 0.0;

  bool flagOdom = false;
  bool flagDroneOdom = false;
  bool flagDroneImu = false;

  nav_msgs::Odometry odom;
  //Take Odometry drone --> andra nel futuro altitude script
  nav_msgs::Odometry drone_odom;
  sensor_msgs::Imu drone_imu;

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
    std::cout << " orientation_theta: " << orientation_theta  << "\n";
   
    flagOdom = true;
}

void drone_odom_callback(const nav_msgs::Odometry::ConstPtr& msg){
     drone_odom = *msg;
    /* drone_x = drone_odom.pose.pose.position.x;
     drone_y = drone_odom.pose.pose.position.y;
     drone_z = drone_odom.pose.pose.position.z;*/
     
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


int main(int argc, char **argv) {
  ros::init(argc, argv, "velocity_publisher");

  ros::NodeHandle nh;

  ros::Publisher vel = nh.advertise<geometry_msgs::Twist>("/mobile/cmd_vel",1);
  ros::Subscriber odom_sub = nh.subscribe("odom",1,Box_odom_callback);
  ros::Subscriber odom_drone_sub = nh.subscribe("/ground_truth/state",5,drone_odom_callback); //Drone callback --> alt script
   ros::Subscriber imu_drone_sub = nh.subscribe("/ardrone/imu",5,drone_Imu_callback);
   ros::Publisher drone_vel = nh.advertise<geometry_msgs::Twist>("/drone/cmd_vel", 1);


//Only to decide the correct orientation angle theta of the mobile
  double des_x = 2.0;
  double des_y = 3.0;
  double theta_des = 0.0;
  double distance = 0.0;  
 
  double vx = 0.3;
  double vy = 0.0;
  double vz = 0.0;
  
  double start_x = 0.0;
   double start_y = 0.0;
  double theta_dot = 0.0;
  double Kp_theta = 1.2;
  double Kd_theta = 0.1;
  double theta_err = 0.0;
  double dt = 0.05;
  double Kp_x = 0.8;
  double Kd_x = 0.2;
  double Kp_y = 0.8;
  double Kd_y = 0.2;
  double x_err = 1.0;
  double x_dot = 0.0;
  double y_dot = 0.0;
  double y_err = 1.0;
  double distance_done = 0.0;
 
  //Drone yaw Variable
  double yaw_ground_dot = 0.0;
  double Kp_yaw = 1.5;
  double Kd_yaw = 0.3;
  double yaw_error = 0.0;

  bool eval_direction = false;
  bool flag = false;
  
  ros::Time current_time, last_time;
  current_time = ros::Time::now();
  last_time = ros::Time::now();
  
  ros::Rate r(5.0);
  while (nh.ok()) {
   geometry_msgs::Twist msg;
   geometry_msgs::Twist drone_vel_msg;
  if (x < 0.01 && x > -0.01 && flag == false){   
      eval_direction = false;
      flag = true;
      std::cout << " sono qui " << "\n";
     }
  if (eval_direction == false){
      distance = sqrt(pow(des_x - x, 2) + pow(des_y - y, 2));
      theta_des = ((des_x - x)/distance) * acos(orientation_theta);
       std::cout << "theta_des: " << theta_des << "\n";
      //PID control theta orientation 
      theta_err = theta_des - orientation_theta;
      theta_dot = (orientation_theta - orientation_theta_old)/dt;
       std::cout << " theta_dot: " <<  theta_dot << "\n";
       
      msg.angular.z = Kp_theta * theta_err + Kd_theta * theta_dot;
       std::cout << " theta_dot: " <<  theta_dot << "\n";
       std::cout << "theta err: " << theta_err << "\n";
       std::cout << "msg.angular.z: " << msg.angular.z << "\n";

        //Yaw drone Control ---> Da mettere poi nello script che controlla altitudine una volta fatto il training per i test
       drone_vel_msg.angular.z = Kp_yaw * (orientation_theta - drone_Yaw) + Kd_yaw*(yaw_ground_dot - drone_ang_vel_z);
      if (theta_des - orientation_theta < 0.05 ) {
          eval_direction = true; 
          start_x = x;
          start_y = y;
         
      }
     
   
}
else{  

    //Yaw drone Control ---> Da mettere poi nello script che controlla altitudine una volta fatto il training per i test
    drone_vel_msg.angular.z = Kp_yaw * (orientation_theta - drone_Yaw) + Kd_yaw*(yaw_ground_dot - drone_ang_vel_z);
//if the vehicle is moving more than 1.5 meters in any directions, start moving a costant velocity
    distance_done = sqrt(pow(start_x + x, 2) + pow(start_y  + y, 2));
    std::cout << " ditstance_done: " <<distance_done << "\n"; 
    if ( distance_done <= 1.5 ){ 

       x_err = 1.5- distance_done;
       std::cout << " start_x + x: " <<start_x + x << "\n";
        std::cout << " x: " << x << "\n";
       x_dot = (x - x_old)/dt;
       msg.linear.x = Kp_x * x_err + Kd_x*x_dot;
       
       std::cout << "x_err" << x_err << "\n";  
       std::cout << " msg.linear.x" << msg.linear.x << "\n";   
       y_err = 1.5- distance_done;
       y_dot = (y - y_old)/dt;
       msg.linear.y = Kp_y * y_err + Kd_y*y_dot;
           
       msg.linear.z = 0.0;

       msg.angular.x = 0.0;
       msg.angular.y = 0.0;
       msg.angular.z = 0.0;
       flag = false;
    }
    else {
       msg.linear.x = vx;
       msg.linear.y = vy;
       msg.linear.z = 0.0;

       msg.angular.x = 0.0;
       msg.angular.y = 0.0;
       msg.angular.z = 0.0;
       flag = false;
    
    }
}
    // next, we'll publish the cmd_vel message over ROS
   
    // publish the message
   
    vel.publish(msg);
    drone_vel.publish(drone_vel_msg);

    orientation_theta_old = orientation_theta;
    x_old = x;
    y_old = y;

    flagOdom = false;
    flagDroneOdom = false;
    flagDroneImu = false;
    last_time = current_time;
    ros::spinOnce();
    r.sleep();
  }
}
