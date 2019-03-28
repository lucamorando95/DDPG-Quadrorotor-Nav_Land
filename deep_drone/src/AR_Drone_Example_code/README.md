# AR_Drone_Example_code

--To run example code, make sure your code become node--

- Make Workspace of your code (http://wiki.ros.org/ROS/Tutorials/CreatingPackage)
		- go to your simulator directory, cd ~/"yourdirectory_workspace"/src
		- catkin_create_pkg (nama package) std_msgs rospy roscpp
		- cd ..
		- catkin_make
		- source devel/setup.bash
		- rospack depends (nama package)
	- Coding your python/C++ program
	- Make your code bocome node python
		- cd ~/"yourdirectory_workspace"/src/"your directory"
		- chmod +x (file name)
    - to make sure your code is node 
      $ls
      and if  your code colour is green, run code
		- open new command line
		- cd ~/"yourdirectory_workspace"/
		- source devel/setup.bash
		- rosrun script (file name)
