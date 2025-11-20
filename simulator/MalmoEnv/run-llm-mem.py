# ------------------------------------------------------------------------------------------------
# Copyright (c) 2018 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------
import malmoenv
import argparse
from pathlib import Path
import time
from PIL import Image
import cv2
import numpy as np
import os
from openai import OpenAI
import requests
import base64
import requests
from openai import OpenAI
import base64
import json
import random
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import time
from openai import OpenAI
import sys
from pathlib import Path
# 加入 LLM



import xml.etree.ElementTree as ET
from pathlib import Path
import malmoenv
import argparse
from pathlib import Path
import time
from PIL import Image
import sys
import os
import asyncio
import sys
import os
import asyncio
import json
import json
import ast

from dotenv import load_dotenv


# 找到 brain 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str("/home/hyl/EmbodyMemory"))
from brain.brain_deepseek import *
import xml.etree.ElementTree as ET

# 加入 memory
from vlm_detect import test3_kimi, test5_kimiV2
from memory.memory_module import CurrentState

system_prompt_en_mc = '''
    as a player in minecraft, you will answer user queries and use tools to get information.
    you will use the tools provided by the server to get information.
    if you need to call tools, you should return the function call in the format without any explanation or other words:
    {
        `entity`:"`Action index`:`action`:`Action Type`"
        `entity`:"`Action index`:`action`:`Action Type`"
    }
    for a example, if you want to move forward once and backward continuous and you have 4 actions 
    [0:move 1 [DiscreteMovement], 1:move -1 [DiscreteMovement], 11:move 1 [ContinuousMovement], 12:move -1 [ContinuousMovement]]
    you should return:
    {
        "entity":"0:move 1:DiscreteMovement"
        "entity":"12:move -1:ContinuousMovement"
    }
    the action index should be correspond to the action
    you will not call the tools directly, but return the function call in the format above.
    when you get the tool call results, you will continue to answer the user query based on the tool call results.
    never make up tools or parameters that are not in.
'''
system_prompt_cn_mc = '''
    作为 minecraft 中的玩家，你将回答用户的查询并使用工具获取信息。
    你将使用服务器提供的工具来获取信息。
    如果你需要调用工具，你应该以不带任何解释或其他词语的格式返回函数调用：
    {
        `entity`:"`Action index`:`action`:`Action Type`"
        `entity`:"`Action index`:`action`:`Action Type`"
    }
    例如，如果你想前进一次并连续后退，并且你有4个动作
    [0:move 1 [DiscreteMovement], 1:move -1 [DiscreteMovement], 11:move 1 [ContinuousMovement], 12:move -1 [ContinuousMovement]]
    你应该返回：
    {
        "entity":"0:move 1:DiscreteMovement"
        "entity":"12:move -1:ContinuousMovement"
    }
    动作索引应与你的动作相对应。
    你不会直接调用工具，而是以上述格式返回函数调用。
    当你获得工具调用结果时，你将继续根据工具调用结果回答用户查询。
    永远不要编造不在其中的工具或参数。
'''

# 加入 memory


# 新增：从 mission xml 中解析 ObservationFromGrid 的 min/max
def get_observation_grid_range(source, grid_name=None):
    # 载入 xml 文本aaaaaaa
    if isinstance(source, (str, Path)) and Path(source).exists():
        xml_text = Path(source).read_text()
    else:
        xml_text = source

    ns = {'m': 'http://ProjectMalmo.microsoft.com'}
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None

    for obs in root.findall('.//m:ObservationFromGrid', ns):
        grid = obs.find('m:Grid', ns)
        if grid is None:
            continue
        name = grid.get('name')
        if grid_name is not None and name != grid_name:
            continue
        min_e = grid.find('m:min', ns)
        max_e = grid.find('m:max', ns)
        if min_e is None or max_e is None:
            continue
        try:
            xmin = int(min_e.get('x'))
            ymin = int(min_e.get('y'))
            zmin = int(min_e.get('z'))
            xmax = int(max_e.get('x'))
            ymax = int(max_e.get('y'))
            zmax = int(max_e.get('z'))
        except (TypeError, ValueError):
            continue
        return {'name': name, 'min': (xmin, ymin, zmin), 'max': (xmax, ymax, zmax)}
    return None

def info_observation_grid_range(around, grid_range):
    # 根据 grid_range 对 around 进行处理，将 y 轴上下翻转
    
    if around is None or grid_range is None:
        return None
    
    x_min, y_min, z_min = grid_range['min']
    x_max, y_max, z_max = grid_range['max']
    
    # 将 around 划分成 y 轴的多个层
    y_layers = y_max - y_min + 1
    x_size = x_max - x_min + 1
    z_size = z_max - z_min + 1
    layer_size = x_size * z_size
    
    around_layers = []
    for y in range(y_layers):
        start_idx = y * layer_size
        end_idx = start_idx + layer_size
        layer = around[start_idx:end_idx]
        
        
        # 将 layer 转换 floor3x3: ['lava', 'obsidian', 'obsidian', 'lava', 'obsidian', 'obsidian', 'lava', 'obsidian', 'obsidian']
        '''
        increasing z
        |       0 1 2
        |       3 4 5
        |       6 7 8 
        |-----> increasing x
        '''
        # 切分 x z 层 为二维列表
        layer_2d = []
        for x in range(x_size):
            row = layer[x * z_size:(x + 1) * z_size]
            # 将 row 翻转
            # row = row[::-1]
            layer_2d.append(row)
        # 打印 layer_2d
        # print("layer_2d:", layer_2d)
        
        # 在前面插入
        around_layers.append(layer_2d)
        
    return around_layers

def info_observation_grid_range_reserve(around, grid_range):
        # 根据 grid_range 对 around 进行处理，将 y 轴上下翻转
    
    if around is None or grid_range is None:
        return None
    
    x_min, y_min, z_min = grid_range['min']
    x_max, y_max, z_max = grid_range['max']
    
    # 将 around 划分成 y 轴的多个层
    y_layers = y_max - y_min + 1
    x_size = x_max - x_min + 1
    z_size = z_max - z_min + 1
    layer_size = x_size * z_size
    
    around_layers = []
    for y in range(y_layers):
        start_idx = y * layer_size
        end_idx = start_idx + layer_size
        layer = around[start_idx:end_idx]
        
        
        # 将 layer 转换 floor3x3: ['lava', 'obsidian', 'obsidian', 'lava', 'obsidian', 'obsidian', 'lava', 'obsidian', 'obsidian']
        '''
        increasing z
        |       0 1 2
        |       3 4 5
        |       6 7 8 
        |-----> increasing x
        '''
        # 切分 x z 层 为二维列表
        layer_2d = []
        for x in range(x_size):
            row = layer[x * z_size:(x + 1) * z_size]
            # 将 row 翻转
            row = row[::-1]
            layer_2d.insert(0, row)
        # 打印 layer_2d
        # print("layer_2d:", layer_2d)
        
        # 在前面插入
        around_layers.insert(0, layer_2d)
        
    return around_layers

def save_img(obs, env):
    # 如果有 depth 信息 depth 为 4
    h, w, d = env.observation_space.shape
    # obs = obs.reshape((960, 1440, 3))   # 高、宽、通道
    obs = obs.reshape((h, w, d))
    print("obs reshaped:", obs.shape, "obs size:", obs.size)
    # # 上下翻转图像
    obs = cv2.flip(obs, 0)
    print("obs.shape:", obs.shape, "obs.size:", obs.size)
    frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("malmo_obs.png", frame)
    
    # --- 分离通道 ---
    if d == 4:
        rgb = obs[:, :, :3]        # RGB 通道
        depth = obs[:, :, 3]       # 深度通道（通常是 float 或 uint8）
    else:
        rgb = obs
        depth = None
    
    # --- 保存 RGB 图像 ---
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("malmo_rgb.png", rgb_bgr)
    # print("已保存 RGB 图像: malmo_rgb.png")

    # --- 保存深度图像 ---
    if depth is not None:
        # 深度通常是原始数值，范围可大可小
        # 为了保存可视化效果，将其归一化到 0~255
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        cv2.imwrite("malmo_depth.png", depth_uint8)
        # print("已保存 Depth 图像: malmo_depth.png")
        # 根据图像的深度信息加 mask ，超过阈值的部分设为白色
        depth_threshold = 200  # 根据需要调整阈值
        mask = depth_uint8 < depth_threshold
        # rgb_masked = np.zeros_like(rgb)
        rgb_masked = np.ones_like(rgb) * 255  # 白色背景
        rgb_masked[mask] = rgb[mask]
        rgb_masked_bgr = cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR)
        cv2.imwrite("malmo_obs.png", rgb_masked_bgr)
    else:
        print("当前观测中没有深度通道")


# 加入 memory
# 根据info来更新 memory
# 将所有可用技能转换成 json 格式 == scene_info
def mc_cap2scene_info(actions, actions_type, grid_info=None):
    skills = []
    skill_specs = {}

    # 遍历动作，生成 capability 名称
    for i, (act, act_type) in enumerate(zip(actions, actions_type, )):
        if act is None:
            act = f"action_{i}"
        act_clean = str(act).strip()
        base = act_clean.split()[0] if len(act_clean.split()) > 0 else f"action{i}"
        # 规范化名称："`Action index`:`action`:`Action Type`"
        
        # cap_name = f"{base}:{i}:{act_type}".replace(" ", "_").replace("-", "neg").replace(".", "_").lower()
        cap_name = f"{base}:{i}:{act_type}".lower()
        
        # 保证唯一
        if cap_name in skills:
            suffix = 1
            while f"{cap_name}_{suffix}" in skills:
                suffix += 1
            cap_name = f"{cap_name}_{suffix}"
        skills.append(cap_name)

        # 生成简单的 skill_spec
        skill_specs[cap_name] = {
            "description": f"action '{act_clean}' and {act_type}",
            "type": "capability",
            "input": None,
            "output": None,
            "dependencies": []
        }

    # 构造 entity_graph（简化版，与 scene_data.json 风格一致）
    entity_graph = {
        "entities": {
            "/": {
                "name": "/",
                "parent": "",
                "children": ["/entity", "/temp"]
            },
            "/temp": {
                "name": "temp",
                "parent": "/",
                "children": []
            },
            "/entity": {
                "name": "entity",
                "parent": "/",
                "children": ["/entity/camera"]
            },
            "/entity/camera": {
                "name": "camera",
                "parent": "/entity",
                "children": []
            },
            # "/entity/radar": {
            #     "name": "radar",
            #     "parent": "/entity",
            #     "children": []
            # }, # grid_info
        },
        "skills": {
            "/": [],
            "/entity": skills
        },
        "graph_structure": {
            "name": "/",
            "path": "/",
            "skills": [],
            "children": {
                "robot": {
                    "name": "robot",
                    "path": "/robot",
                    "skills": skills,
                    "children": {}
                }
            }
        }


    }

    scene_info = {
        "entity_graph": entity_graph,
        "skill_specs": {"skill_specs": skill_specs}
    }
    
    print(scene_info)
    # 写入 scene_info.json
    with open("scene_info.json", "w") as f:
        json.dump(scene_info, f, indent=4)
    

    return scene_info

# 记录临时记忆-空间 更新 entity_graph
def record_short_space_memory(scene_info, obj_list, entity):
    # 更新 entity_graph
    entity_graph = scene_info.get("entity_graph", {})
    if not entity_graph:
        return
    # 根据 entities 获取当前实体位置
    if not entity:
        return scene_info
    

    # 将物体作为 entities/temp 的 孩子 添加到 /temp 下
    for obj in obj_list:
        obj_name = obj.get('name', 'unknown')
        obj_x = obj.get('x', 0)
        obj_y = obj.get('y', 0)
        obj_z = obj.get('z', 0) 
        # 生成唯一路径
        obj_entity_path = f"/temp/{obj_name}_{int(obj_x)}_{int(obj_y)}_{int(obj_z)}"
        entity_graph["entities"][obj_entity_path] = {
            "name": obj_name,
            "parent": "/temp",
            "children": []
        }
        # 将该物体添加到 /temp 的 children 中
        entity_graph["entities"]["/temp"]["children"].append(obj_entity_path)
    
    return scene_info
    
def short2long_space_memory(entity, around, scene_info):
    # 根据 当前 x y z 判定当前位置，并根据around信息更新精确坐标,从scene_info中获取 entity_graph中的 /temp 下的物体 根据坐标和 around 信息更新物体的精确位置
    if not entity or around is None or scene_info is None:
        return scene_info
    entity_graph = scene_info.get("entity_graph", {})
    if not entity_graph:
        return scene_info
    # 取第一个 entity 作为参考
    ref_entity = entity
    ref_x = ref_entity.get('x', 0)
    ref_y = ref_entity.get('y', 0)
    ref_z = ref_entity.get('z', 0) 
    # 遍历 /temp 下的物体
    temp_children = entity_graph["entities"].get("/temp", {}).get("children", [])
    for temp_entity_path in temp_children:
        temp_entity = entity_graph["entities"].get(temp_entity_path, {})
        temp_name = temp_entity.get("name", "unknown")
        # 假设物体名中包含相对位置 如 tree_1_0_2 表示相对于参考实体偏移 (1,0,2)
        parts = temp_name.split('_')
        if len(parts) >= 4:
            try:
                offset_x = int(parts[-3])
                offset_y = int(parts[-2])
                offset_z = int(parts[-1])
                # 计算精确位置
                precise_x = ref_x + offset_x
                precise_y = ref_y + offset_y
                precise_z = ref_z + offset_z
                # 更新实体信息
                temp_entity['precise_position'] = (precise_x, precise_y, precise_z)
            except ValueError:
                continue
            
'''
depth = 0 对应 距离对应 0格
depth = 0 对应 距离对应 1格
depth = 107 对应 距离对应 2格
depth = 149 对应 距离对应 3格
depth = 174 对应 距离对应 4格
depth = 214 对应 距离对应 5格
depth = 221 对应 距离对应 6格
depth = 237 对应 距离对应 7格
'''
def depth_to_blocks(depth_value):
    if depth_value <= 0:
        return 0
    elif depth_value <= 107:
        return 1
    elif depth_value <= 149:
        return 2
    elif depth_value <= 174:
        return 3
    elif depth_value <= 214:
        return 4
    elif depth_value <= 221:
        return 5
    elif depth_value <= 237:
        return 6
    else:
        return 7

'''
        # 修改识别的物体的位置-这里是模糊绝对位置 # 根据视角和深度计算相对位置
        x y z 轴单位为 格
        yaw 0 视角方向为 y 轴 正方向
        yaw 270 视角方向为 x 轴 正方向
        yaw 180 视角方向为 y 轴 负方向
        yaw 90 视角方向为 x 轴 负方向
        
        depth 0-255 距离对应为 0 - 正无穷
        depth = 0 对应 距离对应 0格
        depth = 0 对应 距离对应 1格
        depth = 107 对应 距离对应 2格
        depth = 149 对应 距离对应 3格
        depth = 174 对应 距离对应 4格
        depth = 214 对应 距离对应 5格
        depth = 221 对应 距离对应 6格
        depth = 237 对应 距离对应 7格
    
    env.view_angle 当前视角角度 -2 到 2 - 暂时用不上 -look 向上向下看
'''
def entity_pos2obj_pos(entity, obj):
    # 获取 entity 的位置和朝向
    e_x = entity.get('x', 0)
    e_y = entity.get('y', 0)
    e_z = entity.get('z', 0)
    e_yaw = entity.get('yaw', 0)
    
    # 获取 obj 的深度
    o_x = obj.get('x', 0)
    o_y = obj.get('y', 0)
    o_depth = obj.get('depth', 0)
    
    
    # 根据当前视角和物体在图像中的相对位置计算物体的绝对位置
    # 计算物体相对于视角中心的偏移角度
    # 这里假设图像宽度为 1440，高度为 960
    img_width = 1440
    img_height = 960

    fov_horizontal = 180
    fov_vertical = 90
    
    angle_offset_x = (o_x - img_width / 2) / (img_width / 2) * (fov_horizontal / 2)
    angle_offset_y = (o_y - img_height / 2) / (img_height / 2) * (fov_vertical / 2)
    # 计算物体的实际距离（格数）
    distance_blocks = depth_to_blocks(o_depth)
    # 计算物体的绝对位置
    total_yaw = e_yaw + angle_offset_x
    rad = np.deg2rad(total_yaw)
    o_x = e_x + distance_blocks * (-np.sin(rad))
    o_z = e_z + distance_blocks * np.cos(rad)
    o_y = e_y  # 暂时不考虑高度变化
    return o_x, o_y, o_z
  

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='malmovnv test')
    parser.add_argument('--mission', type=str, default='simulator/MalmoEnv/missions/defaultworld.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--port2', type=int, default=None, help="(Multi-agent) role N's mission port. Defaults to server port.")
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=1, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=0, help='the agent role - defaults to 0')
    parser.add_argument('--episodemaxsteps', type=int, default=0, help='max number of steps per episode')
    parser.add_argument('--saveimagesteps', type=int, default=0, help='save an image every N steps')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync every N resets'
                                                              ' - default is 0 meaning never.')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()
    env = malmoenv.make()
    
    # action_filter = {"move", "turn", "use", "attack", "look", "jump"}
    action_filter = {"move", "turn", "jump", "jumpmove"}
    
    # 获取 xml 中ObservationFromGrid的 around 范围
    around_range = get_observation_grid_range(args.mission, grid_name='around')
    print(f"Around range from XML: {around_range}")
    
    
    
    # 将xml中内容传入 并 解析
    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode,
             action_filter=action_filter,
             resync=args.resync)
    
    # 创建当前场景记忆
    cs = CurrentState()
    scene_info = mc_cap2scene_info(env.actions, env.actions_type, around_range)
    cs.init_Scene(scene_info)
    
    # 在当前目录下创建log文件夹，并获取当前时间作为log文件名
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'action.log'
    
    
    # 清空action.log写入实验信息
    with open(log_file, 'a') as f:
        f.write('======================\n')
        f.write('======================\n')
        f.write('xml ' + xml + '\n')
        f.write('======================\n')
        f.write('======================\n')
        
        
        f.write('mission ' + args.mission + '\n')
        f.write('role ' + str(args.role) + '\n')
        f.write('exp_uid ' + args.experimentUniqueId + '\n')
        f.write('episodes ' + str(args.episodes) + '\n')
        f.write('episode ' + str(args.episode) + '\n')
        f.write('resync ' + str(args.resync) + '\n')
        f.write('======================\n')

        f.write('env.commands ' + str(env.commands) + '\n')
        f.write('env.actions ' + str(env.actions) + '\n')
        # 写入多行回车
        f.write('\n\n\n\n')
    
    load_dotenv()
    api_key = os.getenv("DS_API_KEY")
    client = MCPClient(api_key=api_key)

    for i in range(args.episodes):
        print("reset " + str(i))
        obs = env.reset()
        
        # 打开action.log将写入 episode i
        with open(log_file, 'a') as f:
            f.write('\n\n\n\nepisode ' + str(i) + '\n')
            f.write('======================\n')
            
            
        # 获取用户指令
        user_request = 'find flower and move to it'
        # user_request = input("Press queey or type 'exit': ")
        if user_request.lower() == 'exit':
            print("Exiting the experiment.")
            break
        
        # 初始化 steps, done
        steps = 0
        done = False
        
        # 调试 根据用户输入数字进行相应的操作 # 打印索引 + actions_type
        print("debug Available actions:")
        for i, (act, act_type) in enumerate(zip(env.actions, env.actions_type)):
            print(f"{i}: {act} [{act_type}]", end='\t')
            if (i + 1) % 5 == 0:
                print()
                
        actions_prompt = "the available actions:\n" + ", ".join(f"{i}:{act} [{act_type}]" for i, (act, act_type) in enumerate(zip(env.actions, env.actions_type)))
        
        prompt = system_prompt_en_mc + actions_prompt
        
        # prompt += f'\n 玩家周围的观测网格信息是 {around_range}, 观测为三维，第一维为y轴（第一层为y轴大的），第二维为z轴（第一层为z轴大的），第三层为x轴（第一层为x轴大的）\n'
        prompt += f'\n the observation grid info around player is: {around_range}, the observation is 3D, the first dimension is y axis (the first layer is the largest y axis), the second dimension is z axis (the first layer is the largest z axis), the third dimension is x axis (the first layer is the largest x axis).\n'
        
        # 通过 llm 生成一系列动作
        action_sequence, messages = client.query_request(query=user_request,
                                                          info=None,
                                                          safe_rule=None,
                                                          prompt=prompt)

        user_input = ""
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):

            # add 根据当前环境和用户指令生成一系列动作
            # ['"entity":0:move 1', '"entity":1:move -1', '"entity":2:turn 1', '"entity":3:turn -1', '"entity":4:jump 1', '"entity":5:look 1', '"entity":6:look -1', '"entity":7:attack 1', '"entity":8:use 1']
            action = 0
            print("debug Generated action sequence:", action_sequence)
            
            if action_sequence is None or len(action_sequence) == 0:
                print("No action sequence generated, exiting the episode.")
                break
            
            # 遍历 action_sequence
            cur_act_msg = ""
            # 由于
            for act in action_sequence:
                # "entity": "0:move 1:DiscreteMovement"
                entity, action, act_str, act_type = act.replace('"', '').split(':')

                action = int(action)
                print(f"Entity: {entity}, Index: {action}, Action: {act_str}, Type: {act_type}")
                print(action, str(type(action)))
                
                # 判断 action, act_str, act_type  与 env 中 actions是否匹配
                if action < 0 or action >= len(env.actions):
                    print(f"Action index {action} out of range, skipping this action.")
                    cur_act_msg += f"Skipped invalid action {act}\n"
                    continue
                expected_act_str = env.actions[action]
                # expected_act_type = env.actions_type[action]
                # if act_str != expected_act_str or act_type != expected_act_type:
                if act_str != expected_act_str:
                    print(f"Action string mismatch for index {action}: expected '{expected_act_str}', got '{act_str}'. Skipping this action.")
                    cur_act_msg += f"Skipped invalid action {act}\n"
                    continue
                
                # user_input = input("q")
                # if user_input.lower() == 'q':
                #     break
                print("\n" * 5)
                
                env.render()
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                steps += 1
            
                # 将以上信息写入action.log
                with open(log_file, 'a') as f:
                    f.write("action: " + str(action) + '\n')
                    f.write('reward: ' + str(reward) + '\n')
                    f.write('done: ' + str(done) + '\n')
                    f.write('obs: ' + str(obs) + '\n')
                    f.write('info: ' + info + '\n')
                    f.write('-------------------------\n')
                    
                print("action: " + str(act_str))
                print("reward: " + str(reward))
                print("done: " + str(done))
                print("obs: " + str(obs))
                print("info" + info)
                

                # 将 info 字符串 转成 info 字典
                info = eval(info)
                # 打印出 info 字典的 around 信息
                around = info.get('around', None)
                around = info_observation_grid_range_reserve(around, around_range)
                print("info around: " )
                for layer in around:
                    print(layer, "len:", len(layer))
                    
                    
                entities = info.get('entities', [])

                # 取第一个 entity 作为参考
                entity = entities[0] if entities else {}
                for key, value in entity.items():
                    print(f"  {key}: {value}")
                    
                save_img(obs, env)
                
                
                print("-------------------")
                
                test5_kimiV2()
                # 读取json文件打印识别到的物体和深度信息
                json_output_path = "detection_output_kimi.json"
                obj_list = []
                with open(json_output_path, 'r', encoding='utf-8') as json_file:
                    detection_data = json.load(json_file) # detection_data 是一个列表
                    print("Detected objects and their depth information:")
                    # for obj in detection_data.get('objects', []): 
                    for obj in detection_data:
                        name = obj.get('label', 'unknown')
                        depth = obj.get('depth', 'unknown')
                        print(f"Object: {name}, Depth: {depth}")
                        # 根据当前xy值和识别到的物体深度计算物体的绝对位置-粗略的-后续根据“雷达”信息精确定位
                        obj_list.append(
                            {
                                'name': name,
                                'depth': depth,
                                'x': entity.get('x'),
                                'y': entity.get('y'),
                                'z': entity.get('z')
                            }
                    )
                
                cur_act_msg += f"action {act_str}, detect object is {obj_list}, around info is {around}\n"
                

                time.sleep(1)
                
                
            
            messages.append({"role": "user", "content": cur_act_msg})
            action_sequence, messages = client.query_request(query=user_request, messages=messages)
        
        if user_input.lower() == 'q':
            break
        # 打印messages最后一个 content 
        print(messages[-1]['content'] if messages else "No messages.")

    env.close()


