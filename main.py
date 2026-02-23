import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import json
from robot import Panda
from teleop import KeyboardController
from objects import objects


# function that returns the goal positions and rotation around z for each object
# from our lecture notes, these are theta_box, theta_banana, etc.
def get_object_goals():
    box_state = box.get_state()
    banana_state = banana.get_state()
    bottle_state = bottle.get_state()
    goals = {}
    goals["box_position"] = box_state["position"] + np.array([0, 0, 0.05])
    goals["box_rotz"] = box_state["euler"][2] + np.pi/2
    goals["banana_position"] = banana_state["position"] + np.array([0, 0, -0.01])
    goals["banana_rotz"] = banana_state["euler"][2] + np.pi/2
    goals["bottle_position"] = bottle_state["position"] + np.array([-0.01, 0, 0.05])
    goals["bottle_rotz"] = bottle_state["euler"][2] + 0.0
    return goals

# function that outputs the actions to reach potential target
def get_object_actions(robot_position, robot_euler, goals):
    actions = {}
    actions["box"] = action_to_goal(robot_position, robot_euler, goals["box_position"], goals["box_rotz"])
    actions["banana"] = action_to_goal(robot_position, robot_euler, goals["banana_position"], goals["banana_rotz"])
    actions["bottle"] = action_to_goal(robot_position, robot_euler, goals["bottle_position"], goals["bottle_rotz"])
    return actions

# function that outputs the next target position and target quaternion if we are 
# reaching for the goal_position and goal_rotz
def action_to_goal(robot_position, robot_euler, goal_position, goal_rotz):
    position_error = goal_position - robot_position
    rotz_error = goal_rotz - robot_euler[2]
    if np.linalg.norm(position_error) > 0.01:
        position_error = position_error / np.linalg.norm(position_error)

    if np.abs(rotz_error) > 0.01:
        rotz_error = rotz_error / np.abs(rotz_error)
    # the gains 0.001 and 0.005 match the default pos_step and rot_step in teleop
    target_position = robot_position + 0.001 * position_error
    target_euler = np.array([np.pi, 0, robot_euler[2] + 0.005 * rotz_error])
    return target_position, np.array(p.getQuaternionFromEuler(target_euler))

# parameters
control_dt = 1. / 240.

# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
# disable keyboard shortcuts so they do not interfere with keyboard control
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                cameraYaw=40.0,
                                cameraPitch=-30.0, 
                                cameraTargetPosition=[0.5, 0.0, 0.2])

# load the objects
urdfRootPath = pybullet_data.getDataPath()
plane = objects.PyBulletObject("plane.urdf", basePosition=[0, 0, -0.625])
table = objects.PyBulletObject("table/table.urdf", basePosition=[0.5, 0, -0.625])
box = objects.YCBObject("003_cracker_box.urdf", basePosition=[0.65, -0.3, 0.09], 
                                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
banana = objects.YCBObject("011_banana.urdf", basePosition=[0.75, 0.3, 0.025], 
                                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
bottle = objects.YCBObject("006_mustard_bottle.urdf", basePosition=[0.5, 0.00, 0.06], 
                                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

# load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)

# teleoperation interface
teleop = KeyboardController()

# run simulation
# you can teleoperate the robot using the keyboard;
# see "teleop.py" for the mapping between keys and motions
state = panda.get_state()
target_position = state["ee-position"]
target_quaternion = state['ee-quaternion']

start_position = np.array(state["ee-position"])
BETA = 10.0
assist_enabled = True
prev_toggle_pressed = False

while True:
    # update the target pose
    action = teleop.get_action()
    human_position = target_position + action[0:3]
    human_quaternion = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler(action[3:6]),
                                                [0, 0, 0], target_quaternion)[1]
    human_quaternion = np.array(human_quaternion)

    # activate manual control when you press '.' toggle button
    toggle_pressed = action[7] > 0
    if toggle_pressed and not prev_toggle_pressed:
        assist_enabled = not assist_enabled
        print(f"assistance {'enabled' if assist_enabled else 'disabled'}")
    prev_toggle_pressed = toggle_pressed

    # share autonomy between human and robot
    ### to implement: currently we just execute human action ###
    # what robot should do by understadning the human intentions?
    # predict human goal

    # predict the humans goal
    state = panda.get_state()
    curr_position = np.array(state["ee-position"])

    P = [0, 0, 0]
    # can use a loop here to iterate through whole code
    goals = get_object_goals()

    for idx, theta in enumerate(["box_position", "banana_position", "bottle_position"]):
        theta_position = goals[theta]
        start_to_goal = np.linalg.norm(start_position - theta_position)
        curr_to_goal = np.linalg.norm(curr_position - theta_position)
        start_to_curr = np.linalg.norm(curr_position - start_position)
        P[idx] = np.exp(BETA * start_to_goal) / (np.exp(BETA * (start_to_curr + curr_to_goal)))

    P = np.array(P)
    
    P = P/ np.sum(P)

    #print(P)

    # step 2 : find the robot action a_r

    actions = get_object_actions(curr_position, state["ee-euler"], goals)
    
    if assist_enabled:
        if P[0] > 0.6:
            print("i am helping for the: box")
            robot_position, robot_quaternion = actions["box"]
        elif P[1] > 0.6:
            print("i am helping for the : banana")
            robot_position, robot_quaternion = actions["banana"]
        elif P[2] > 0.6:
            print("i am helping for the: bottle")
            robot_position, robot_quaternion = actions["bottle"]
        else:
            print("i am confused")
            robot_position = np.copy(human_position)
            robot_quaternion = np.copy(human_quaternion)
    else:
        robot_position = np.copy(human_position)
        robot_quaternion = np.copy(human_quaternion)
    

    # step 3. blend the human action with robot action

    # we can control the manual mode or assitive mode using the toggle buttion '.'
    if not assist_enabled:
        ALPHA = 0.0
    elif np.linalg.norm(action[0:6]) < 1e-5:
        ALPHA = 1.0
    else:
        ALPHA = 0.1



    target_position = (1 - ALPHA) * human_position + ALPHA * robot_position
    target_quaternion = (1 - ALPHA) * human_quaternion + ALPHA * robot_quaternion

    # impose workspace limit
    if target_position[2] < 0.02:
        target_position[2] = 0.02

    # move to the target pose
    panda.move_to_pose(ee_position=target_position, ee_quaternion=target_quaternion, positionGain=1)

    # open or close the gripper
    if action[6] == +1:
        panda.open_gripper()
    elif action[6] == -1:
        panda.close_gripper()

    # step simulation
    p.stepSimulation()
    time.sleep(control_dt)
