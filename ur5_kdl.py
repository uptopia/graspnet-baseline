#!/usr/bin/env python3

# /workspace/orocos_kinematics_dynamics/python_orocos_kdl/PyKDL
import os,sys
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print("ROOT_DIR: " + ROOT_DIR)
# sys.path.append("/workspace/orocos_kinematics_dynamics/python_orocos_kdl/PyKDL")
# print(sys.path)
import PyKDL
import math
# sys.path.remove('/usr/lib/python3/dist-packages')

def create_ur5_chain():
    # DH parameters
    a = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
    alpha = [math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0]
    d = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
    theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    chain = PyKDL.Chain()

    for i in range(6):
        chain.addSegment(PyKDL.Segment(PyKDL.Joint(PyKDL.Joint.RotZ),\
                        PyKDL.Frame(PyKDL.Rotation.RotZ(theta[i])*PyKDL.Rotation.RotX(alpha[i]), \
                                       PyKDL.Vector(a[i], -d[i]*math.sin(alpha[i]), d[i]*math.cos(alpha[i])))))
    return chain

def compute_IK(chain, target_pose):
    '''
    Forward Kinematics
    '''
    fk = PyKDL.ChainFkSolverPos_recursive(chain)
    pos = PyKDL.Frame()
    q = PyKDL.JntArray(6)
    qq = [-0.9, -0.9, 43.0, 21.0, 16.0,-27.0]

    for i in range(6):
        q[i]= qq[i]
    fk_flag=fk.JntToCart(q, pos)
    print("fk_flag: ", fk_flag)
    print("pos: ", pos)

    '''
    Inverse Kinematics
    '''

    ikv = PyKDL.ChainIkSolverVel_pinv(chain)
    ik = PyKDL.ChainIkSolverPos_NR(chain, fk, ikv)

    target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(target_pose[3],target_pose[4], target_pose[5]),\
                                PyKDL.Vector(target_pose[0],target_pose[1], target_pose[2]))
    initial_joint_angles = PyKDL.JntArray(chain.getNrOfJoints())
    result = PyKDL.JntArray(chain.getNrOfJoints())
    ik.CartToJnt(initial_joint_angles, target_frame, result)
    print("result: ", result)
    return result

if __name__ == "__main__":
    chain = create_ur5_chain()
    target_pose = [0.5, 0.3, 0.4, 0.1, 0.0, 0.0]
    joint_angles = compute_IK(chain, target_pose)
    print("joint angles:", joint_angles)
