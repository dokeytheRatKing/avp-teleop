"""AVP -> Astribot S1 **upper-body** teleoperation (head + torso + both arms).

This package extends the dual-arm teleop in :mod:`avp_teleop` by adding the
Apple Vision Pro head 6-DoF pose as a third tracked end-effector and solving the
whole upper body (4-DoF torso + 2-DoF neck + two 7-DoF arms) in a *single*
merged Pinocchio + Pink differential-IK problem. Because the head, left-hand and
right-hand frame tasks share one configuration, the arms automatically
compensate for torso motion -- no hand-written coordinate correction.

The existing :mod:`avp_teleop` package is reused (transport ``HandFrame``,
relative-pose calibration, finger retargeting, the ``SimRobot`` interface) and
left completely unchanged.
"""
