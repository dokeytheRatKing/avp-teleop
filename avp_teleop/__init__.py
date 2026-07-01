"""AVP -> MuJoCo teleoperation pipeline for the Astribot S1.

Modular teleoperation link:

    Apple Vision Pro  --(UDP)-->  retargeting  -->  MuJoCo sim (or ROS robot)

Sub-packages / modules:
    config            single source of truth (IP, joint names, ranges, gains)
    transport         zero-dependency UDP pub/sub + a fixed binary message schema
    avp_publisher     reads the AVP stream and publishes hand frames
    retarget.frames   coordinate conventions + relative-pose calibration
    retarget.arm_ik   damped least squares IK (wrist pose -> 7 arm joints)
    retarget.hand_retarget   21 keypoints -> per-finger curl -> finger joints
    robot_interface   abstraction layer (SimRobot now, ROSRobot later)
    sim_teleop        main loop: subscribe -> retarget -> drive MuJoCo
"""

__all__ = ["config"]
