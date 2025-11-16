import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class NodeController(Node):
    def __init__(self):
        super().__init__('node_controller')

    def publish_send(self):
        """send a message to the 'brainquery' topic"""
        publisher = self.create_publisher(String, 'brainquery', 10)
        msg = String()
        # input query string
        msg.data = "get hi node status"
        publisher.publish(msg)
        # wait for a moment to ensure the message is sent
        rclpy.spin_once(self, timeout_sec=1.0)
        # log the message
        self.get_logger().info(f"Published message: {msg.data}")

def main():
    rclpy.init()
    node = NodeController()
    node.publish_send()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()