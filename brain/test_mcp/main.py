import asyncio
import os
import time
import subprocess
import sys
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
import os
import time

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
print(root_dir)
from Robonix.manager.eaios_decorators import package_init, mcp_start,eaios



system_prompt_en = ""
with open("sys_prompt.txt", "r") as f:
    system_prompt_en = f.read().strip()

# judge whether the message contains a tool call
def judge_tool_call(content):
    content_split = content.split("\n")
    for i in content_split:
        if "[FC]" in i:
            return True
    return False
    

def tool_calls_format(tool_calls_str: str):
    '''
    {
        [FC]:get_alerts(state=CA);
    }
    to
    [    
        {
            "name": "get_alerts",
            "args": {
                "state": "CA"
            }
        }
    ]
    '''
    tool_calls = []
    tools_split = tool_calls_str.split("\n")
    for i in tools_split:
        if "[FC]" in i:
            # get func name  [Funcall]:map_create();
            funcName = i.split(":")[1].split("(")[0].strip()
            # get parameters
            args_str = i.split("(")[1].split(")")[0].strip()
            args_dict = {}
            if args_str:
                args_list = args_str.split(",")
                for arg in args_list:
                    key, value = arg.split("=")
                    args_dict[key.strip()] = value.strip().strip("'")
            tool_calls.append({
                "name": funcName,
                "args": args_dict
            })
    return tool_calls

# add tool func to asyncio event loop
def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []
        self.tool_session_map = {}
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key="your keys",
        )

    async def connect_to_server(self):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py)
        """
        available_tools = []
        tool_session_map: Dict[str, ClientSession] = {}
        for server_url in server_url_array:
            read, write = await self.exit_stack.enter_async_context(sse_client(url=server_url))
            # TODO : add connect error handler
            session: ClientSession = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        
            # List available tools
            response = await session.list_tools()
            tools = response.tools
            for tool in tools:
                if tool.name in tool_session_map:
                    print(f"Tool: {tool.name}, exist")
                else :
                    available_tools.append(tool)
                    tool_session_map[tool.name] = session
                    print(f"Tool: {tool.name}, Description: {tool.description}")
                    if tool.name == "test_nv":
                        result = await session.call_tool("test_nv", {})
        self.available_tools = available_tools
        self.tool_session_map = tool_session_map
        print("finished connecting to server")

    async def process_query(self, query: str) -> str:

        # get available tools from server
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in self.available_tools]

        
        # current query tools
        query_prompt = system_prompt_en
        print("in this query Available tools:")
        for tool in available_tools:
            print(f"tool: {tool['name']} - {tool['description']}")
            query_prompt += f"{tool['name']}: {tool['description']}\n"

        # print(f"debug query_prompt: {query_prompt}\n\n\n")
        
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "system",
                "content": query_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # Initial Claude API call - this demo use deepseek chat model
        start_time = time.time()
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            # tools=available_tools
        )
        end_time = time.time()
        
        content = response.choices[0].message.content
        print("debug response:\n", content, "\ndebug take time:", end_time - start_time)
        
        # Process response and handle tool calls
        tool_results = []
        final_text = []
        
        while judge_tool_call(content) == True:
            
            tool_calls = tool_calls_format(content[content.find("{"):content.rfind("}") + 1]) # str -> list
            print("debug tool_calls:\n", tool_calls)
            for tool in tool_calls:
                tool_name = tool["name"]
                tool_args = tool["args"]
                
                print(f"debug tool call: {tool_name} with args {tool_args}")

                # eg : result = await self.session.call_tool("get_alerts", {"state": "CA"})
                # eg : result = await self.session.call_tool("get_forecast", {"latitude": 37.7749, "longitude": -122.4194})
                result = await self.tool_session_map[tool_name].call_tool(tool_name, tool_args)
                
                print(f"debug tool call result: {result.content}")
                
                tool_results.append({
                    "call": tool_name,
                    "result": result.content
                })
                
                # add llm response to messages
                messages.append({
                    "role": "assistant",
                    "content": content
                })
                
                # add tool call result to messages
                messages.append({
                    "role": "user",
                    "content": f"Calling tool {tool_name} with args {tool_args} returned: {result.content}",
                })
                
                # Get next response from llm
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                )
                
                print("debug response:\n", response.choices[0].message.content)

                # loop through response content
                content = response.choices[0].message.content
        
        # out of loop, no more tool calls
        final_text.append(content)
        
        return "\n".join(final_text)

    async def chat_loop(self, init_query: str = None):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                # query = "告诉我你现在可以调用哪些函数"

                if query.lower() == 'quit':
                    break
                print("get query:", query)
                response = await self.process_query(query)
                print("\n" + response)
                
                # return # test
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

# ros2 topic to get information from 
class BrainNodeController(Node):
    def __init__(self, client: MCPClient):
        super().__init__("client_node_controller")
        self.get_logger().info("Client Node Controller initialized")
        
        # init server process
        self.client = client

        # init ROS2 Topic to get information for query
        self.topic_subscriber = self.create_subscription(
            String,
            'brainquery',
            self.call_service,
            10
        )
        
        self.get_logger().info("Client node initialized, waiting for 'brain-query' messages...")

    def shutdown_node(self):
        """Shutdown the node and clean up resources"""
        self.get_logger().info("Shutting down client node...")
        self.destroy_node()

    def call_service(self, msg):
        print(f"ROS2 Received message: {msg.data}")
        if msg.data.lower() == 'quit':
            self.shutdown_node()
            return "Node shutdown requested"
        response = run_async(self.client.process_query(msg.data))
        print(f"Response from server: {response}")
        return response

async def run_mcp_client():
    # Load environment variables from .env file
    load_dotenv()
    client = MCPClient(api_key=os.getenv("API_KEY"))
    # start the server process and get PID
    process = subprocess.Popen(
        ["python3", "capability/example_hello/api/cap_server.py"],
    )
    print(f"server PID: {process.pid}")

    time.sleep(2)  # wait for server to start

    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()
        # Terminate the server process
        process.terminate()
        process.wait()  # Wait for the process to terminate

def run_ros2_node():
    # Load environment variables from .env file
    load_dotenv()
    client = MCPClient(api_key=os.getenv("API_KEY"))

    process = subprocess.Popen(
        ["python3", "capability/example_hello/api/cap_server.py"],
    )
    print(f"server PID: {process.pid}")
    time.sleep(2)  # wait for server to start
    
    rclpy.init()
    node = BrainNodeController(client)
    try:
        run_async(node.client.connect_to_server())
        rclpy.spin(node)  # keep the node running to receive messages
    except KeyboardInterrupt:
        node.get_logger().info("Node shutdown requested...")
    finally:
        run_async(client.cleanup())
        # Terminate the server process
        process.terminate()
        process.wait()  # Wait for the process to terminate
        node.shutdown_node()
        rclpy.shutdown()

def main():
    asyncio.run(run_mcp_client())
    
def test():
    run_ros2_node()

if __name__ == "__main__":
    test()