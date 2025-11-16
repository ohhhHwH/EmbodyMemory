# Brain

Central runtime of the embodied system. Responsible for:

- Interpreting commands
- Task decomposition into Skills
- Planning under Cost constraints
- Reading/writing from Memory
- Calling LLM/VLM for reasoning

## 🧠 Includes:

- World model abstraction
- Model interfaces (e.g., via Ollama or DeepSeek)
- Prompt construction (JIT / dynamic)
- Skill orchestration logic

This is the decision-making and planning center of the system.

requirements.txt contains dependencies for the brain module.(as a example)

add .env file in /brain directory with the following content:

```plaintext
API_KEY=sk-xxx
```

Test CMD as follow :

```SHELL
python  path/to/brain-deepseek.py
```

add publisher and subscriber in nodetest.py

publisher code as follow:

```python
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
        msg.data = "Hello from NodeController"
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
```

TODO :

- [ ] the model's prompt error recognition as a command to be called
```
 I don't have a "quit" function, but I can close the hi or hello objects if you'd like. Would you like me to close one of them?

For example, I can close the hi object with:
{
    [FC]:api_close_hi();
}

Or the hello object with:
{
    [FC]:api_close_hello();
}

Let me know which one you'd like to close, or if you'd prefer something else. 
debug take time: 6.8265299797058105
debug tool_calls:
 [{'name': 'api_close_hi', 'args': {}}, {'name': 'api_close_hello', 'args': {}}]
debug tool call: api_close_hi with args {}
```

- [√] when use ros2 node to call the tool ,cant be executed, need to be fixed
```
run_ros2_node() the rclpy.spin(node) is a blocking synchronous call that occupies the main thread, preventing control from returning to asyncio. As a result, when MCP needs to continue running the coroutine at await result = self.tool_session_map[tool_name].call_tool(...), the asyncio event loop is blocked by rclpy.spin(), and the coroutine cannot be scheduled, making it appear as if "execution is stuck".

add def run_async(coro) function to run async function in a blocking way, so that MCPClient can call the tool function in a blocking way.
```


