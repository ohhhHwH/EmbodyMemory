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
system_prompt_en_mc_v2 = '''
    as a player in minecraft, you will answer user queries and use tools to get information.
    you will use the tools provided by the server to get information.
    if you need to call tools, you should return the function call in the format without any explanation or other words:
    {
        `Action index`:`action`
        `Action index`:`action`
    }
    for a example, if you want to move forward once, then turn degree add 90, and you have 5 actions
    [0:move 1 , 1:move -1 , 2:turn 1 , 3:turn -1, 4:craft [item]]
    you should return:
    {
        0:move 1
        2:turn 1
        4:craft planks
    }
    the action index should be correspond to the action.
    you will not call the tools directly, but return the function call in the format above.
    when you get the tool call results, you will continue to answer the user query based on the tool call results.
    never make up tools or parameters that are not in.
'''

system_prompt_cn_mc_v2 = '''
    作为 minecraft 中的玩家，你将回答用户的查询并使用工具获取信息。
    你将使用服务器提供的工具来获取信息。
    如果你需要调用工具，你应该以不带任何解释或其他词语的格式返回函数调用：
    {
        `Action index`:`action`
        `Action index`:`action`
    }
    例如，如果你想前进一次,然后转向,最后制造木板，并且你有5个动作
    [0:move 1 , 1:move -1 , 2:turn 1 , 3:turn -1, 4:craft [item]]
    你应该返回：
    {
        0:move 1
        2:turn 1
        4:craft planks
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
        yaw 0 视角方向为 z 轴 正方向
        yaw 270 视角方向为 x 轴 正方向
        yaw 180 视角方向为 z 轴 负方向
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
  
def inventory_parse(info):
    inventory = {}
    for i in range(0,40):
        key = 'InventorySlot_'+str(i)+'_item' # 物品名称
        size_key = 'InventorySlot_'+str(i)+'_size' # 物品数量
        var_key = 'InventorySlot_'+str(i)+'_variant' # 材质
        col_key = 'InventorySlot_'+str(i)+'_colour' # 颜色
        item = {}
        
        if key in info and info[key] != 'air':
            item['id'] = i
            item['item'] = info[key]
            item['size'] = int(info[size_key])             
            if var_key in info:
                item['variant'] = int(info[var_key])
            if col_key in info:
                item['colour'] = int(info[col_key])
            inventory[i] = item
        
            
    return inventory


# 对每个 动作 进行技能描述

act_info_en = {
    "move 1": "Move forward",
    "move -1": "Move backward",
    "turn 1": "yaw degrees add 90",
    "turn -1": "yaw degrees minus 90",
    "look 1": "Look down, look degree minus 1, max is -2",
    "look -1": "Look up, look degree add 1, min is 2",
    "jumpmove": "Jump while moving forward",
    "attack": "Attack the target in front",
    "use": "Use item",
    "jumpuse": "Jump and use item",
    "discardCurrentItem": "Discard the currently selected item",
    
    "hotbar.[int]": "Select hotbar slot [int]",
    "swapInventoryItems [i j]": "Swap inventory slots i and j items",
    "combineInventoryItems [i j]": "Combine inventory slots i and j items",
    
    "craft [item_name]": "Craft [item_name] when enough materials in inventory",
    "nearbyCraft [item_name]": "Craft [item_name] using nearby crafting table when enough materials in inventory",
    "nearbySmelt [item_name]": "Smelt [item_name] using nearby furnace when enough materials in inventory",

}

viewinfo_en = """yaw 0 is z axis positive direction,yaw 180 is z axis negative direction
    yaw 270 is x axis positive direction.yaw 90 is x axis negative direction"""

craftitem_en = """The name of an item to be crafted or smelted.

    Supported 2*2 crafting recipes (usable with 'craft [item]'):
        'log' -> 'planks' (4)
        'planks' -> 'stick' (4)
        'planks' -> 'crafting_table' (1)
        'coal + stick' -> 'torch' (4)

    Supported nearbyCraft 3*3 crafting recipes (require a nearby crafting table):
        'planks + stick' -> 'wooden_pickaxe'
        'planks + stick' -> 'wooden_sword'
        'planks + stick' -> 'wooden_axe'
        'planks + stick' -> 'wooden_shovel'
        'cobblestone + stick' -> 'stone_pickaxe'
        'cobblestone + stick' -> 'stone_sword'
        'cobblestone + stick' -> 'stone_axe'
        'cobblestone + stick' -> 'stone_shovel'
        'iron_ingot + stick' -> 'iron_pickaxe'
        'iron_ingot + stick' -> 'iron_sword'
        'iron_ingot + stick' -> 'iron_axe'
        'iron_ingot + stick' -> 'iron_shovel'
        'diamond + stick' -> 'diamond_pickaxe'
        'diamond + stick' -> 'diamond_sword'
        'diamond + stick' -> 'diamond_axe'
        'diamond + stick' -> 'diamond_shovel'
        'planks + planks' -> 'chest'
        'planks + cobblestone' -> 'furnace'
        'iron_ingot' -> 'bucket'
        'iron_ingot' -> 'iron_helmet / chestplate / leggings / boots'

    Supported nearbySmelt smelting recipes (require a furnace):
        'iron_ore' -> 'iron_ingot'
        'gold_ore' -> 'gold_ingot'
        'raw_porkchop' -> 'cooked_porkchop'
        'raw_beef' -> 'cooked_beef'
        'log' -> 'charcoal'
        'sand' -> 'glass'

    Notes:
        - 'craft' works only for 2*2 recipes.
        - 'nearbyCraft' works only if a crafting table is within reach.
        - 'nearbySmelt' works only if a furnace is within reach.
        - All recipes require enough materials in inventory.
    """

act_info_cn = {
    "move 1": "前进一步",
    "move -1": "后退一步",
    "turn 1": "向右转90度",
    "turn -1": "向左转90度",
    "look 1": "向下看，视角角度减1，最大值为-2",
    "look -1": "向上看，视角角度加1，最小值为2",
    "jumpmove": "跳跃并前进",
    "attack": "攻击前方目标",
    "use": "使用物品",
    "jumpuse": "跳跃并使用物品",
    "hotbar.[int]": "选择快捷栏槽位[int]",
    "swapInventoryItems [i j]": "交换库存槽位i和j的物品",
    "combineInventoryItems [i j]": "将库存槽位i和j的物品合并",
    "discardCurrentItem": "丢弃当前选中的物品",
    "craft [item_name]": "当库存中有足够材料时，制作[item_name]",
    "nearbyCraft [item_name]": "当库存中有足够材料时，使用附近的工作台制作[item_name]",
    "nearbySmelt [item_name]": "当库存中有足够材料时，使用附近的熔炉熔炼[item_name]",
}

viewinfo_cn = """yaw 0 是 z 轴正方向，yaw 180 是 z 轴负方向，
    yaw 270 是 x 轴正方向，yaw 90 是 x 轴负方向"""

craftitem_cn = """要制作或熔炼的物品名称。

    支持的2*2制作配方（可与'craft [item]'一起使用）：
        'log' -> 'planks' (4)
        'planks' -> 'stick' (4)
        'planks' -> 'crafting_table' (1)
        'coal + stick' -> 'torch' (4)
    支持的nearbyCraft 3*3制作配方（需要附近有工作台）：
        'planks + stick' -> 'wooden_pickaxe'
        'planks + stick' -> 'wooden_sword'
        'planks + stick' -> 'wooden_axe'
        'planks + stick' -> 'wooden_shovel'
        'cobblestone + stick' -> 'stone_pickaxe'
        'cobblestone + stick' -> 'stone_sword'
        'cobblestone + stick' -> 'stone_axe'
        'cobblestone + stick' -> 'stone_shovel'
        'iron_ingot + stick' -> 'iron_pickaxe'
        'iron_ingot + stick' -> 'iron_sword'
        'iron_ingot + stick' -> 'iron_axe'
        'iron_ingot + stick' -> 'iron_shovel'
        'diamond + stick' -> 'diamond_pickaxe'
        'diamond + stick' -> 'diamond_sword'
        'diamond + stick' -> 'diamond_axe'
        'diamond + stick' -> 'diamond_shovel'
        'planks + planks' -> 'chest'
        'planks + cobblestone' -> 'furnace'
        'iron_ingot' -> 'bucket'
        'iron_ingot' -> 'iron_helmet / chestplate / leggings / boots'

    支持的nearbySmelt熔炼配方（需要附近有熔炉）：
        'iron_ore' -> 'iron_ingot'
        'gold_ore' -> 'gold_ingot'
        'raw_porkchop' -> 'cooked_porkchop'
        'raw_beef' -> 'cooked_beef'
        'log' -> 'charcoal'
        'sand' -> 'glass'

    注意：
        - 'craft'仅适用于2*2配方。
        - 'nearbyCraft'仅在工作台在可达范围内时有效。
        - 'nearbySmelt'仅在熔炉在可达范围内时有效。
        - 所有配方都需要库存中有足够的材料。
    """


def parse_action_string(action_str):    # 解析动作字符串，返回动作和动作类型
    try:
        parts = action_str.split(':')
        if len(parts) != 2:
            return None, None
        action_id = parts[0]
        action_str = parts[1]
        # 通过 act_info_en 检查 id 与 str 是否匹配
        if "hotbar." in action_str:
            # 提取数字 id
            hotbar_num = action_str.split('.')[-1]
            action_str = f"hotbar.{hotbar_num}"
        elif "craft" in action_str.lower():
            # 提取制作的物品
            item_name = action_str.split(' ', 1)[1] if ' ' in action_str else ''
            # TODO 检查物品是否充足
            pass
        elif "InventoryItems" in action_str:
            # 提取 i j 
            i, j = action_str.split(' ')[1:3]
        elif "Smelt" in action_str:
            # 提取制作的物品
            item_name = action_str.split(' ', 1)[1] if ' ' in action_str else ''
            # TODO 检查物品是否充足
            pass
        elif action_str not in act_info_en:
            return None, None
        return action_id, action_str
    except Exception:
        return None, None

def action_prompt_generate(actions, actions_type, action_filter, action_diy):
    # action_diy 是 需要作为字符串输入 的动作集合 并使用 step_diy 来执行

    # actions 描述
    
    prompt = "You can use the following actions:\n"
    i = 0
    for act in act_info_en :
        prompt += f"{i}: {act}: {act_info_en[act]}\n"
        i += 1
    
    return prompt

if __name__ == '__main__':
    
    # 解析命令行参数
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

    # 载入 mission xml
    xml = Path(args.mission).read_text()
    env = malmoenv.make()
    
    # 定义 action_filter 和 action_diy
    action_filter = {"move", "turn", "use", "attack","jumpmove",
                     "craft", "nearbyCraft",
                     "swapInventoryItems", "combineInventoryItems", "discardCurrentItem",
                     "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9"}
    action_diy = {
        "hotbar": ["hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5", "hotbar.6",
                   "hotbar.7", "hotbar.8", "hotbar.9"],
        "craft": ["craft", "nearbyCraft"],
        "swapInventoryItems": ["swapInventoryItems", ],
        "combineInventoryItems": ["combineInventoryItems"],
        "discardCurrentItem": ["discardCurrentItem"],
    }

    
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
    log_file = log_dir / f'action_{time.strftime("%Y%m%d")}.log'
    
    
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
    
    # 初始化 MCPClient
    load_dotenv()
    api_key = os.getenv("DS_API_KEY")
    client = MCPClient(api_key=api_key)
    
    # 构建 actions_prompt
    system_prompt = system_prompt_en_mc_v2
    actions_prompt = action_prompt_generate(env.actions, env.actions_type, action_filter, action_diy)
    rule_prompt = viewinfo_en + craftitem_en
    
    # obs_prompt = f'\n 玩家周围的观测网格信息是 {around_range}, 观测为三维，第一维为y轴（第一层为y轴大的），第二维为z轴（第一层为z轴大的），第三层为x轴（第一层为x轴大的）\n'
    obs_prompt = f'\n the observation grid info around player is: {around_range}, the observation is 3D, the first dimension is y axis (the first layer is the largest y axis), the second dimension is z axis (the first layer is the largest z axis), the third dimension is x axis (the first layer is the largest x axis).\n'
    
    for i in range(args.episodes):
        print("reset " + str(i))
        obs = env.reset()
        
        # 打开action.log将写入 episode i
        with open(log_file, 'a') as f:
            f.write('\n\n\n\nepisode ' + str(i) + '\n')
            f.write('======================\n')

        # 初始化 steps, done
        steps = 0
        done = False
             
        # prompt生成
        prompt = system_prompt + actions_prompt + rule_prompt + obs_prompt
        
        # 获取用户指令
        # user_request = input("Press queey or type 'exit': ")
        user_request = 'Make a wooden axe'
        if user_request.lower() == 'exit':
            print("Exiting the experiment.")
            break
        
        # TODO 根据 记忆 检索相关信息加入 prompt
        # relative_info = cs.retrieval_Request(user_request)
        # prompt += f"\nRelevant information from memory:\n{relative_info}\n"
        
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

            for act in action_sequence:
                # 解析动作字符串
                act_idx, act_str = parse_action_string(act)
                if act_idx is None:
                    print(f"Invalid action format: {act}, skipping.")
                    with open(log_file, 'a') as f:
                        f.write(f"Invalid action format: {act}, skipping.\n")
                    continue
                action = act_str
                
                # 调试：用户决定是否执行
                user_input = input("input 'q' to quit:")
                if user_input.lower() == 'q':
                    break
                else:
                    pass
                print("\n" * 5)
                
                # 执行动作
                with open(log_file, 'a') as f:
                    f.write("diy action: " + action + '\n')
                print("diy action: " + action)
                env.render()
                
                obs, reward, done, info = env.step_diy(action)
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
                
                # 打印当前库存信息
                inventories = inventory_parse(info)
                
                # 打印出 info 字典的 around 信息
                around = info.get('around', None)
                around = info_observation_grid_range_reserve(around, around_range)
                print("info around: " )
                for layer in around:
                    print(layer, "len:", len(layer))
                    
                # 取第一个 entity 作为参考
                entities = info.get('entities', [])
                entity = entities[0] if entities else {}
                for key, value in entity.items():
                    print(f"  {key}: {value}")
                    
                # 保存图像
                save_img(obs, env)
                
                print("---------detect info---------")
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
                
                # LLM 做法
                cur_act_msg += f"action {act_str}, entity {entity}, detect object is {obj_list}, around info is {around}\n"
                
                # TODO LLM+MEM 做法
                # 更新短期空间记忆 + 短期-》长期 + 更新当前场景记忆 + 检索相关信息
                # scene_info = record_short_space_memory(scene_info, obj_list, entity)
                # scene_info = short2long_space_memory(entity, around, scene_info)
                # cs.update_Scene(scene_info)
                # cur_act_msg += cs.retrieval_Request(user_request)
                
                time.sleep(1)
                
                
            # 更新 llm messages
            messages.append({"role": "user", "content": cur_act_msg})
            action_sequence, messages = client.query_request(query=user_request, messages=messages)
        
        if user_input.lower() == 'q':
            break
        # 打印messages最后一个 content 
        print(messages[-1]['content'] if messages else "No messages.")

    env.close()


