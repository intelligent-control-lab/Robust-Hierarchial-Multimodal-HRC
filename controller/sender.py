import rospy
from std_msgs.msg import String
import argparse
from pathlib import Path
FILE_DIR = Path(__file__).parent
import time
import pdb

def send_intention_to_ros(intention):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('intention', anonymous=True)
    rospy.loginfo(intention)
    pub.publish(intention)
    # time.sleep(0.1)
    # rospy.loginfo(intention)
    # pub.publish(intention)

def send_command_to_ros(command="stop",count=0):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    command = f"{command}_{count}"
    rospy.init_node('command', anonymous=True)
    rospy.loginfo(command)
    rate = rospy.Rate(10) # 10hz
    pub.publish(command)
    rate.sleep()
    rospy.loginfo(command)
    pub.publish(command)

def send_action_to_ros(action):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('action', anonymous=True)
    rospy.loginfo(action)
    pub.publish(action)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intention', type=str,
                    help="intention send to rospy")
    parser.add_argument('--command', type=str,
                    help="command send to rospy")
    parser.add_argument('--index', type=int,
                    help="command index")
    parser.add_argument('--action', type=str,
                    help="action send to rospy")
    args = parser.parse_args()
    # while True:
    if args.intention:
        send_intention_to_ros(args.intention)
    if args.command:
        send_command_to_ros(args.command,args.index)
    if args.action:
        send_action_to_ros(args.action)
    