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
import re
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
    作为 minecraft 中的玩家,你将回答用户的查询并使用工具获取信息。
    你将使用服务器提供的工具来获取信息。
    如果你需要调用工具,你应该以不带任何解释或其他词语的格式返回函数调用：
    {
        `entity`:"`Action index`:`action`:`Action Type`"
        `entity`:"`Action index`:`action`:`Action Type`"
    }
    例如,如果你想前进一次并连续后退,并且你有4个动作
    [0:move 1 [DiscreteMovement], 1:move -1 [DiscreteMovement], 11:move 1 [ContinuousMovement], 12:move -1 [ContinuousMovement]]
    你应该返回：
    {
        "entity":"0:move 1:DiscreteMovement"
        "entity":"12:move -1:ContinuousMovement"
    }
    动作索引应与你的动作相对应。
    你不会直接调用工具,而是以上述格式返回函数调用。
    当你获得工具调用结果时,你将继续根据工具调用结果回答用户查询。
    永远不要编造不在其中的工具或参数。
'''
system_prompt_en_mc_v2 = '''as a player in minecraft, you will answer user queries and use tools to get information.
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
    作为 minecraft 中的玩家,你将回答用户的查询并使用工具获取信息。
    你将使用服务器提供的工具来获取信息。
    如果你需要调用工具,你应该以不带任何解释或其他词语的格式返回函数调用：
    {
        `Action index`:`action`
        `Action index`:`action`
    }
    例如,如果你想前进一次,然后转向,最后制造木板,并且你有5个动作
    [0:move 1 , 1:move -1 , 2:turn 1 , 3:turn -1, 4:craft [item]]
    你应该返回：
    {
        0:move 1
        2:turn 1
        4:craft planks
    }
    动作索引应与你的动作相对应。
    你不会直接调用工具,而是以上述格式返回函数调用。
    当你获得工具调用结果时,你将继续根据工具调用结果回答用户查询。
    永远不要编造不在其中的工具或参数。
'''
sub_mission_prompt_en = '''
    as a player in minecraft, you will decompose the user's task into several sub-tasks, each sub-task should be as short and easy to complete as possible.
    the types of sub-tasks are divided into the following categories:
    1. find tasks: such as "find wood", "find water", etc.
    2. gather tasks: such as "gather wood", "gather stone", etc.
    3. craft tasks: such as "craft planks", "craft tools", etc.
    4. explore tasks: such as "explore cave", "explore map", etc.
    please decompose the user's task description into several sub-tasks in order, the format of the sub-tasks is
    {
        1:"find wood",
        2:"gather wood",
        3:"craft planks",
        4:"explore cave"
    }
'''
sub_mission_prompt_cn = '''
    作为 minecraft 中的玩家,你将用户的任务拆解成多个子任务,每个子任务应尽可能简短且易于完成。
    子任务的类型分为以下几种：
    1. 寻找类任务：如“寻找 木头”、寻找 水”等。
    2. 采集类任务：如“采集 木头”、“采集 石头”等。
    3. 制作类任务：如“制作 木板”、“制作 工具”等。
    4. 探索类任务：如“探索 洞穴”、“探索 地图”等。
    请根据用户的任务描述,将任务拆解成多个子任务,并按顺序列出,其子任务格式为
    {
        1:"找到 木头",
        2:"采集 木头",
        3:"制作 木板",
        4:"探索 洞穴"
    }
    请确保子任务简洁明了,并且每个子任务都可以通过游戏内的动作来完成。
'''

# 对每个 动作 进行技能描述

act_info_enV2 = {
    "move 1": "Move forward",
    "move -1": "Move backward",
    "turn 1": "yaw degrees add 90, when yaw degree is 0, next turn 1 will be 90",
    "turn -1": "yaw degrees minus 90, when yaw degree is 0, next turn -1 will be 270",
    "look 1": "Look down, view degree minus 90, max is -180",
    "look -1": "Look up, view degree add 90, min is 180, when view degree is 180, next look 1 will not change",
    "jumpmove": "Jump while moving forward, when there is an obstacle in front, can jump over the obstacle",
    "attack": "Attack the target in front 1-2 blocks, collect items. When you need to collect items at REL y=0, first use look 1 to adjust the view to level 0, then attack, when collecting REL y=1 items, no need to adjust the view, when collecting REL y=2 items, need to use look -1 to adjust the view to level 2",
    "use": "Use item, use the currently selected hotbar item on the target in front 1-2 blocks, must have a target to use the item",
    "jumpuse": "Jump and use item",
    "discardCurrentItem": "Discard the currently selected item",
    
    "hotbar.[int]": "Select hotbar slot [int]",
    "swapInventoryItems [i j]": "Swap inventory slots i and j items",
    "combineInventoryItems [i j]": "Combine inventory slots i and j items",
    
    "craft [item_name]": "Craft [item_name] when enough materials in inventory",
    "nearbyCraft [item_name]": "Need to place a crafting table in the surrounding environment first, then craft [item_name] using nearby crafting table when enough materials in inventory",
    "nearbySmelt [item_name]": "Smelt [item_name] using nearby furnace when enough materials in inventory",

}

act_info_en = {
    "move 1": "Move forward",
    "move -1": "Move backward",
    "turn 1": "yaw degrees add 90, when yaw degree is 0, next turn 1 will be 90",
    "turn -1": "yaw degrees minus 90, when yaw degree is 0, next turn -1 will be 270",
    "look 1": "Look down, view degree minus 90, max is -180",
    "look -1": "Look up, view degree add 90, min is 180, when view degree is 180, next look 1 will not change",
    "jumpmove": "Jump while moving forward, when there is an obstacle in front, can jump over the obstacle",
    "attack": "Attack the target in front 1-2 blocks, collect items. When you need to collect items at REL y=0, first use look 1 to adjust the view to level 0, then attack, when collecting REL y=1 items, no need to adjust the view, when collecting REL y=2 items, need to use look -1 to adjust the view to level 2",
    "use": "Use item, use the currently selected hotbar item on the target in front 1-2 blocks, must have a target to use the item",
    "jumpuse": "Jump and use item",
    "discardCurrentItem": "Discard the currently selected item",
    
    "hotbar.[int]": "Select hotbar slot [int]",
    "swapInventoryItems [i j]": "Swap inventory slots i and j items",
    "combineInventoryItems [i j]": "Combine inventory slots i and j items",
    
    "craft [item_name]": "Craft [item_name] when enough materials in inventory",
    "nearbyCraft [item_name]": "Need to place a crafting table in the surrounding environment first, then craft [item_name] using nearby crafting table when enough materials in inventory",
    "nearbySmelt [item_name]": "Smelt [item_name] using nearby furnace when enough materials in inventory",

}

act_info_cnV2 = {
    "move 1": "前进一步",
    "move -1": "后退一步",
    "turn 1": "向右转90度",
    "turn -1": "向左转90度",
    "look 1": "向下看,视角角度减90,最大值为-180,当视角角度为-180时,下一次向下看将不再变化",
    "look -1": "向上看,视角角度加90,最小值为180,当视角角度为180时,下一次向上看将不再变化",
    "jumpmove": "跳跃并前进,当前方有障碍物时,可以跳跃前进越过障碍物",
    "attack": "攻击前方1-2格的目标,采集物品,当需要采集REL y=0的物品时,先 look 1 向下调整视角到第0层,再使用attack进行采集, 采集 REL y=1 的物品时,不需要调整视角, 采集 REL y=2 的物品时,需要 look -1 向上调整视角到第2层",
    "use": "使用物品, 将当前快捷栏选中的物品使用在前方1-2格的目标上,必须有目标才能使用物品",
    "jumpuse": "跳跃并使用物品",
    "hotbar.[int]": "选择快捷栏槽位[int]",
    "swapInventoryItems [i j]": "交换库存槽位i和j的物品",
    "combineInventoryItems [i j]": "将库存槽位i和j的物品合并",
    "discardCurrentItem": "丢弃当前选中的物品",
    "craft [item_name]": "当库存中有足够材料时,制作[item_name]",
    "nearbyCraft [item_name]": "需要先将crafr table放置在周围的环境中,当库存中有足够材料时,使用附近的工作台制作[item_name]",
    "nearbySmelt [item_name]": "当库存中有足够材料时,使用附近的熔炉熔炼[item_name]",
}

act_info_cn = {
    "move 1": "前进一步",
    "move -1": "后退一步",
    "turn 1": "向右转90度",
    "turn -1": "向左转90度",
    "look 1": "向下看,视角角度减90,最大值为-180,当视角角度为-180时,下一次向下看将不再变化",
    "look -1": "向上看,视角角度加90,最小值为180,当视角角度为180时,下一次向上看将不再变化",
    "jumpmove": "跳跃并前进,当前方有障碍物时,可以跳跃前进越过障碍物",
    "attack": "攻击前方1-2格的目标,采集物品,当需要采集REL y=0的物品时,先 look 1 向下调整视角到第0层,再使用attack进行采集, 采集 REL y=1 的物品时,不需要调整视角, 采集 REL y=2 的物品时,需要 look -1 向上调整视角到第2层",
    "use": "使用物品, 将当前快捷栏选中的物品使用在前方1-2格的目标上,必须有目标才能使用物品",
    "jumpuse": "跳跃并使用物品",
    "hotbar.[int]": "选择快捷栏槽位[int]",
    "swapInventoryItems [i j]": "交换库存槽位i和j的物品",
    "combineInventoryItems [i j]": "将库存槽位i和j的物品合并",
    "discardCurrentItem": "丢弃当前选中的物品",
    "craft [item_name]": "当库存中有足够材料时,制作[item_name]",
}


viewinfo_en = """yaw 0 is z axis positive direction,yaw 180 is z axis negative direction
    yaw 270 is x axis positive direction.yaw 90 is x axis negative direction.
    player is located at the center of around, y axis level 0 is the player's level, the center grid of level 0 is the player's grid,
    y axis level 1 is the player's view level, to observe the next level of the grid in front, you need to adjust the view with look 1.
    when you need to collect objects on the y axis level 0 in front of the player, you need to adjust the view down to level 0 with look 1.
    """
    
viewinfo_cn = """yaw 0 是 z 轴正方向,yaw 180 是 z 轴负方向,
    yaw 270 是 x 轴正方向,yaw 90 是 x 轴负方向。
    玩家位于 around 的中心位置,y轴第0层为玩家所在层级,第0层中心格为玩家所在格,
    y轴第1层为玩家视角层级,观察前面一格的下一级需要 look 1 调整视角。
    当需要搜集玩家面前y轴第0层的物体,需要look 1向下调整视界到第0层。
    """

around_info_en = """the observation is 3D, the first dimension is y axis,
the second dimension is z axis,
the third dimension is x axis.
"""

around_info_cn = """观察是3D的,第一个维度是y轴,
第二个维度是z轴,
第三个维度是x轴。
"""

craftitem_enV2 = """
    Notes:
        - 'craft' works only for 2*2 recipes.
        - 'nearbyCraft' works only if a crafting table is within reach.
        - 'nearbySmelt' works only if a furnace is within reach.
        - All recipes require enough materials in inventory.
    """

craftitem_cnV2 = """
    注意：
        - 'craft'仅适用于2*2配方,包括planks stick torch。
        - 'nearbyCraft'仅在工作台在可达范围内时有效。
        - 'nearbySmelt'仅在熔炉在可达范围内时有效。
        - 所有配方都需要库存中有足够的材料。
    """

craftitem_en = """
    Notes:
        - All recipes require enough materials in inventory.
    """

craftitem_cn = """
    注意：
        - 所有配方都需要库存中有足够的材料。
    """

# 实际运行中 都用 craft 且不需要 区分 2x2 和 3x3 和 建立工作台和熔炉, 这里只是仿真模拟
craft_requirements = {
    
            "planks": {"log": 1},                 # 1 木头 → 4 木板（RL 中通常简化为 1:1）
            "stick": {"planks": 2},               # 2 木板 → 4 木棍（简化为 2:1）
            # 火把
            "torch": {"stick": 1, "coal": 1},     

            "crafting_table": {"planks": 4},      # 4 木板 → 1 工作台
            # 基础工具
            "wooden_pickaxe": {"planks": 3, "stick": 2},
            "wooden_axe": {"planks": 3, "stick": 2},
            "wooden_shovel": {"planks": 1, "stick": 2},
            "wooden_sword": {"planks": 2, "stick": 1},

            # 石制工具
            "stone_pickaxe": {"cobblestone": 3, "stick": 2},
            "stone_axe": {"cobblestone": 3, "stick": 2},
            "stone_shovel": {"cobblestone": 1, "stick": 2},
            "stone_sword": {"cobblestone": 2, "stick": 1},

            # 熔炉与基本方块
            "crafting_table": {"planks": 4},
            "furnace": {"cobblestone": 8},
            
            # 基础工具
            "wooden_pickaxe": {"planks": 3, "stick": 2},
            "wooden_axe": {"planks": 3, "stick": 2},
            "wooden_shovel": {"planks": 1, "stick": 2},
            "wooden_sword": {"planks": 2, "stick": 1},
            
            # 石制工具
            "stone_pickaxe": {"cobblestone": 3, "stick": 2},
            "stone_axe": {"cobblestone": 3, "stick": 2},
            "stone_shovel": {"cobblestone": 1, "stick": 2},
            "stone_sword": {"cobblestone": 2, "stick": 1},

            # 熔炉与基本方块
            "crafting_table": {"planks": 4},
            "furnace": {"cobblestone": 8},
            
            # 铁制工具
            "iron_pickaxe": {"iron_ingot": 3, "stick": 2},
            "iron_axe": {"iron_ingot": 3, "stick": 2},
            "iron_shovel": {"iron_ingot": 1, "stick": 2},
            "iron_sword": {"iron_ingot": 2, "stick": 1},

            # 铁制护甲
            "iron_helmet": {"iron_ingot": 5},
            "iron_chestplate": {"iron_ingot": 8},
            "iron_leggings": {"iron_ingot": 7},
            "iron_boots": {"iron_ingot": 4},
    
            # 熔炼
            "iron_ingot": {"iron_ore": 1, "coal": 1},  # 简化：需要1煤作燃料
            "gold_ingot": {"gold_ore": 1, "coal": 1},
            "glass": {"sand": 1, "coal": 1},

            # 食物加工
            "cooked_beef": {"raw_beef": 1, "coal": 1},
            "cooked_salmon": {"raw_salmon": 1, "coal": 1},

            # 建筑材料
            "stone": {"cobblestone": 1, "coal": 1},     # 平滑石头（烧制）
            "stone_bricks": {"stone": 4},
    

}

craft_requirementsV2 = {
    "craft" :  {
            "planks": {"log": 1},                 # 1 木头 → 4 木板（RL 中通常简化为 1:1）
            "stick": {"planks": 2},               # 2 木板 → 4 木棍（简化为 2:1）
            # 火把
            "torch": {"stick": 1, "coal": 1},     
    },
    
    "nearbyCraft" : {
            "crafting_table": {"planks": 4},      # 4 木板 → 1 工作台
            # 基础工具
            "wooden_pickaxe": {"planks": 3, "stick": 2},
            "wooden_axe": {"planks": 3, "stick": 2},
            "wooden_shovel": {"planks": 1, "stick": 2},
            "wooden_sword": {"planks": 2, "stick": 1},

            # 石制工具
            "stone_pickaxe": {"cobblestone": 3, "stick": 2},
            "stone_axe": {"cobblestone": 3, "stick": 2},
            "stone_shovel": {"cobblestone": 1, "stick": 2},
            "stone_sword": {"cobblestone": 2, "stick": 1},

            # 熔炉与基本方块
            "crafting_table": {"planks": 4},
            "furnace": {"cobblestone": 8},
            
            # 基础工具
            "wooden_pickaxe": {"planks": 3, "stick": 2},
            "wooden_axe": {"planks": 3, "stick": 2},
            "wooden_shovel": {"planks": 1, "stick": 2},
            "wooden_sword": {"planks": 2, "stick": 1},
            
            # 石制工具
            "stone_pickaxe": {"cobblestone": 3, "stick": 2},
            "stone_axe": {"cobblestone": 3, "stick": 2},
            "stone_shovel": {"cobblestone": 1, "stick": 2},
            "stone_sword": {"cobblestone": 2, "stick": 1},

            # 熔炉与基本方块
            "crafting_table": {"planks": 4},
            "furnace": {"cobblestone": 8},
            
            # 铁制工具
            "iron_pickaxe": {"iron_ingot": 3, "stick": 2},
            "iron_axe": {"iron_ingot": 3, "stick": 2},
            "iron_shovel": {"iron_ingot": 1, "stick": 2},
            "iron_sword": {"iron_ingot": 2, "stick": 1},

            # 铁制护甲
            "iron_helmet": {"iron_ingot": 5},
            "iron_chestplate": {"iron_ingot": 8},
            "iron_leggings": {"iron_ingot": 7},
            "iron_boots": {"iron_ingot": 4},
    },
    
    "nearbySmelt" : {
            # 熔炼
            "iron_ingot": {"iron_ore": 1, "coal": 1},  # 简化：需要1煤作燃料
            "gold_ingot": {"gold_ore": 1, "coal": 1},
            "glass": {"sand": 1, "coal": 1},

            # 食物加工
            "cooked_beef": {"raw_beef": 1, "coal": 1},
            "cooked_salmon": {"raw_salmon": 1, "coal": 1},

            # 建筑材料
            "stone": {"cobblestone": 1, "coal": 1},     # 平滑石头（烧制）
            "stone_bricks": {"stone": 4},
    }

}



# 从 mission xml 中解析 ObservationFromGrid 的 min/max
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
    # 根据 grid_range 对 around 进行处理,将 y 轴上下翻转
    
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

def info_observation_grid_range_reserve(around, grid_range)->dict:
        # 根据 grid_range 对 around 进行处理,将 y 轴上下翻转
    
    if around is None or grid_range is None:
        return None
    
    x_min, y_min, z_min = grid_range['min']
    x_max, y_max, z_max = grid_range['max']
    
    # 将 around 划分成 y 轴的多个层
    y_layers = y_max - y_min + 1
    x_size = x_max - x_min + 1
    z_size = z_max - z_min + 1
    layer_size = x_size * z_size
    
    around_layers = {}
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
        layer_2d = {}
        for z in range(z_size):
            row = layer[z * x_size:(z + 1) * x_size]
            # 将 row 翻转
            # row = row[::-1]
            # layer_2d.insert(0, row)
            layer_x = {}
            for x in range(x_size):
                layer_x[x + x_min] = row[x]
            layer_2d[z+z_min] = layer_x
        # 打印 layer_2d
        # print("layer_2d:", layer_2d)
        
        # 在前面插入
        # around_layers.insert(0, layer_2d)
        around_layers[y+y_min] = layer_2d

    
    return around_layers

def info_process(env, info):
    # 将 info 字符串 转成 info 字典
    info = eval(info)
    
    # 打印当前库存信息
    inventories = inventory_parse(info)
    print("Current Inventory:")
    slot = 0
    for item in inventories:
        print(f" Slot {slot}: {item}")
        slot += 1
    
    # 打印出 info 字典的 around 信息
    around = info.get('around', None)
    around = info_observation_grid_range_reserve(around, around_range)
    # print("info around: " )
    # for y, layer in around.items():
    #     print(f" REL y={y}:")
    #     for z, row in layer.items():
    #         print(f" REL z={z}: {row}")
    #     print("\n")
        
    # 取第一个 entity 作为参考
    entities = info.get('entities', [])
    # entity = entities[0] if entities else {}
    # 找到 entity['name'] == 'Agent0' 的实体
    entity = next((e for e in entities if e.get('name') == 'Agent0'), {})
    entity_processed = {}
    for key, value in entity.items():
        print(f"  {key}: {value}")
        if key in ['x', 'y', 'z']:
            entity_processed[key] = value
        elif key in ['yaw', 'pitch']:
            entity_processed[key] = value
        elif key in ['life', 'name']:
            entity_processed[key] = value
        else:
            pass
    entity_processed['view'] = env.view_angle * 90
    
    
    
    return inventories, around, entity_processed

def around_msg(around)->str:
    print("info around: " )
    msg = "info around: \n"
    for y, layer in around.items():
        print(f" REL y={y}:")
        msg += f" REL y={y}:\n"
        for z, row in layer.items():
            print(f" REL z={z}: {row}")
            msg += f" REL z={z}: {row}\n"
        msg += "\n" 
    return msg

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
        # 深度通常是原始数值,范围可大可小
        # 为了保存可视化效果,将其归一化到 0~255
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        cv2.imwrite("malmo_depth.png", depth_uint8)
        # print("已保存 Depth 图像: malmo_depth.png")
        # 根据图像的深度信息加 mask ,超过阈值的部分设为白色
        depth_threshold = 200  # 根据需要调整阈值
        mask = depth_uint8 < depth_threshold
        # rgb_masked = np.zeros_like(rgb)
        rgb_masked = np.ones_like(rgb) * 255  # 白色背景
        rgb_masked[mask] = rgb[mask]
        rgb_masked_bgr = cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR)
        cv2.imwrite("malmo_obs.png", rgb_masked_bgr)
    else:
        print("当前观测中没有深度通道")

# 将所有可用技能转换成 json 格式 == scene_info # 加入 memory # 根据info来更新 memory
def mc_cap2scene_info(actions, actions_type, act_info : dict, grid_info=None):
    entity_skills_name = []
    entity_skill_specs = {}
    
    # 遍历act_info_en,生成 capability 名称
    for i, (act, desc) in enumerate(act_info.items()):
        cap_name = f"{i}:{act}".lower()
        entity_skills_name.append(cap_name)
        entity_skill_specs[cap_name] = {
            "name": cap_name,
            "description": desc,
            # "input": None,
            # "output": None,
            # "dependencies": []
            # "children": []
        }
    
    # 构造 entity_graph（简化版,与 scene_data.json 风格一致）
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
        },
        "skills": { # skills 是 混合 情景节点的节点
            "/": [],
            "/temp": [],
            "/entity": entity_skills_name
        },
        "graph_structure": {
            "name": "/",
            "path": "/",
            "skills": [],
            "children": {
                "entity": {
                    "name": "entity",
                    "path": "/entity",
                    "skills": entity_skills_name,
                    "children": {}
                },
                "temp": {
                    "name": "temp",
                    "path": "/temp",
                    "skills": [],
                    "children": {}
                }
            }
        }
    }

    scene_info = {
        "entity_graph": entity_graph,
        "skill_specs": entity_skill_specs
    }

    return scene_info

def diff_obj_list(obj_list_bef, obj_list)->{list, list}:
    # 计算 obj_list 的差异
    input_diff = []
    output_diff = []
    
    # 合并 obj_list_bef 和 obj_list 为hash 元素为 key
    diff = {}
    for obj in obj_list_bef:
        if obj['ACC'] == True:
            key = (obj['name'], obj['RELx'], obj['RELy'], obj['RELz'], obj['size'])
            diff[key] = obj
    for obj in obj_list:
        if obj['ACC'] == True:
            key = (obj['name'], obj['RELx'], obj['RELy'], obj['RELz'], obj['size'])
            if key in diff:
                # 已存在，表示未变化，删除该键
                del diff[key]
            else:
                # 不存在，表示新增，加入 diff
                output_diff.append(obj)
    
    
    # 遍历所有物体，计算数量差异
    for key, obj in diff.items():
        input_diff.append(obj)

    return input_diff, output_diff

# 生成记忆点
def mem_generation(action, inventories_bef, aimed_object_bef, obj_list_bef, inventories, obj_list, entity, env)->list:
    msg = {}
    # inventory 包含的物体
    input_items, output_items = craft_diff_get(inventories_bef, inventories)
    # aimed 物体
    # aimed_obj = 
    # obj_list 
    obj_input_diff, obj_output_diff = diff_obj_list(obj_list_bef, obj_list)
    # around 包含的物体
    # get_around_objects_precise_pos(entity, around, around_range)
    # entity 的状态
    yaw = entity['yaw']
    degree = env.view_angle
    hotbar_id = env.hotbarid
    hotbar_item = get_hotbar_item(inventories, hotbar_id)
    
    # msg 根据action类型的不同记录不同的信息
    msg['name'] = action
    input = {}
    output = {}
    # dependencies = {}
    if "craft" in action or "nearbyCraft" in action or "nearbySmelt" in action:
        if output_items != {}:
            input["input_items"] = input_items
            output["output_items"] = output_items
    elif action == "attack":
        input['yaw'] = yaw
        input["obj_list"] = obj_input_diff
        input['aimed_obj'] = aimed_object_bef
        output["output_items"] = output_items
        output["obj_list"] = obj_output_diff
    elif action == "use" or action == "jumpuse" or "hotbar." in action:
        input['hotbar_item'] = hotbar_item
    elif "Item" in action:
        pass
    elif "look" in action :
        input['degree'] = degree
    else:
        # 移动的指令 ， 连续移动不做记录，但最后一次移动需要记录， 仅保留上一次移动的场景 即寻找最后的场景信息
        input["obj_list"] = obj_list
        input['aimed_obj'] = aimed_object_bef
        # output["obj_list"] = obj_output_diff
        
    msg['input'] = input
    msg['output'] = output
    # msg['dependencies'] = dependencies
    return msg

# 将 动作序列转换为 固定记忆 格式 TODO CS的场景mem 对记忆信息进行处理 删除无用的 移动信息， 只保留移动后的场景信息
def skill2FIXED_mem(task_describe, record_actions, scene_info):
    
    
    
    task_name = task_describe.replace(" ", "_").lower()
    task_describe = f"{task_describe} - composed of {len(record_actions)} steps."
    
    task_skills_spec = []
    task_skills_name = []
    # 遍历act_info_en,生成 capability 名称
    for i, mem_msg in enumerate(record_actions):
        action_name = mem_msg.get('name', '')
        # 改名 唯一
        cap_name = f"{i}:{action_name}".lower()
        mem_msg['action'] = cap_name
        mem_msg["description"] = f"step {i} action {action_name}"
        # 
        task_skills_name.append(cap_name)
        task_skills_spec.append(mem_msg)
        
    # 将 task_memory 添加到 temp_skills 中
    temp_skills = scene_info.get("entity_graph", {}).get("skills", {}).get("/temp", [])
    temp_skills.append(task_name)
    
    temp_skills_gs = scene_info.get("entity_graph", {}).get("graph_structure", {}).get("children", {}).get("temp", {}).get("skills", [])
    temp_skills_gs.append(task_name)
    
    # entity_skill_specs 中 加入 task_spec 说明
    task_spec = {
        "name": task_name,
        "description": task_describe,
        "input": record_actions[0].get("input", None), # 获取第一步的 input 作为 task_memory 的 input
        "output": record_actions[-1].get("output", None), # 获取最后一步的 output 作为 task_memory 的 output
        "actions": task_skills_spec,
    }
    temp_skill_specs = scene_info.get("skill_specs", {})
    temp_skill_specs[task_name] = task_spec

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
    
    # TODO 增量更新
    
    # 清空 /temp 下的 children
    entity_graph["entities"]["/temp"]["children"] = []
    entity_graph["graph_structure"]["children"]["temp"]["children"] = {}
    for child in list(entity_graph["entities"].keys()):
        if child.startswith("/temp/"):
            del entity_graph["entities"][child]
    
    obj_name_list = []
    # 将物体作为 entities /temp 的 孩子 添加到 /temp 下
    for obj in obj_list:
        obj_name = obj.get('name', 'unknown')
        obj_x = obj.get('RELx', 0)
        obj_y = obj.get('RELy', 0)
        obj_z = obj.get('RELz', 0) 
        # 生成唯一路径
        obj_name = f"{obj_name}_{int(obj_x)}_{int(obj_y)}_{int(obj_z)}"
        obj_entity_path = f"/temp/{obj_name}"
        obj_name_list.append(obj_entity_path)
        entity_graph["entities"][obj_entity_path] = {
            "name": obj_name,
            "parent": "/temp",
            "RElx": obj_x,
            "REly": obj_y,
            "RElz": obj_z,
            "size": obj.get('size', 1),
        }
        # scene_info 中  entity_graph 的 graph_structure 也要更新
        entity_graph["graph_structure"]["children"]["temp"]["children"][obj_name] = {
            "name": obj_name,
            "path": obj_entity_path,
        }
        # 将该物体添加到 /temp 的 children 中
        entity_graph["entities"]["/temp"]["children"].append(obj_entity_path)
    return scene_info
    
def short2long_space_memory(entity, around, scene_info):
    # 根据 当前 x y z 判定当前位置,并根据around信息更新精确坐标,从scene_info中获取 entity_graph中的 /temp 下的物体 根据坐标和 around 信息更新物体的精确位置
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
    # 这里假设图像宽度为 1440,高度为 960
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
    inventory = []
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
                item['variant'] = str(info[var_key])
            if col_key in info:
                item['colour'] = str(info[col_key])
            inventory.append(item)
        
            
    return inventory

def parse_action_sequence(action_sequence):    # 解析动作序列字符串,返回动作列表和动作类型列表
    # 提取 { }可能有多个 { }
    ret = []
    
    action_lines = re.findall(r'\{([^}]*)\}', action_sequence, re.DOTALL)
    
    for action_line in action_lines:
        ret.append(action_line)
    return ret

def parse_action_string(action_str):    # 解析动作字符串,返回动作和动作类型
    try:
        # 去掉所有的 "  
        action_str = action_str.replace('"', '')
        parts = action_str.split(':')
        if len(parts) != 2:
            return None, None
        action_id = parts[0]
        action_str = parts[1]
        # action_str 开头如果是空格则删去开头空格
        action_str = action_str.strip()
        
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

def get_around_list(around):
    listy = []
    for y, layer in around.items():
        listz = []
        for z, row in layer.items():
            listx = []
            for x, val in row.items():
                listx.insert(0, val)
            listz.insert(0, listx)
        listy.insert(0, listz)
    return listy

def get_around_objects_precise_pos(entity, around, around_range):
    # TODO 根据 around 信息 获取物体的精确位置列表
    obj_list = []
    
    
    around = get_around_list(around)


    # 遍历 around 各层, 对 x y z 相邻且相同的物体进行聚类,并给出中心点位置
    if around is None or around_range is None:
        return obj_list

    y_layers = len(around)
    if y_layers == 0:
        return obj_list
    x_size = len(around[0])
    z_size = len(around[0][0])

    # visited 三维标记
    visited = [[[False for _ in range(z_size)] for _ in range(x_size)] for _ in range(y_layers)]

    # grid min offsets（网格坐标的最小值,通常为负数,表示相对于实体的偏移起点）
    x_min, y_min, z_min = around_range['min']

    from collections import deque

    for y in range(y_layers):
        for x in range(x_size):
            for z in range(z_size):
                if visited[y][x][z]:
                    continue
                block = around[y][x][z]
                if not block or block == 'air':
                    visited[y][x][z] = True
                    continue

                # BFS 聚类同类相邻方块（6 邻域）
                q = deque()
                q.append((y, x, z))
                visited[y][x][z] = True
                coords = []
                label = block

                while q:
                    cy, cx, cz = q.popleft()
                    coords.append((cx, cy, cz))  # store as (x_idx,y_idx,z_idx)

                    # neighbors (6 directions)
                    neighs = [ (cy-1, cx, cz), (cy+1, cx, cz), (cy, cx-1, cz), (cy, cx+1, cz), (cy, cx, cz-1), (cy, cx, cz+1) ]
                    for ny, nx, nz in neighs:
                        if 0 <= ny < y_layers and 0 <= nx < x_size and 0 <= nz < z_size and not visited[ny][nx][nz]:
                            nblock = around[ny][nx][nz]
                            if nblock == label:
                                visited[ny][nx][nz] = True
                                q.append((ny, nx, nz))
                            else:
                                visited[ny][nx][nz] = False

                # 计算簇中心（索引平均）,然后映射到 grid 坐标（相对于实体的偏移）
                if coords:
                    sum_x = sum(c[0] for c in coords)
                    sum_y = sum(c[1] for c in coords)
                    sum_z = sum(c[2] for c in coords)
                    n = len(coords)
                    mean_x_idx = sum_x / n
                    mean_y_idx = sum_y / n
                    mean_z_idx = sum_z / n

                    # 将索引映射为 grid 坐标：grid_x = x_min + mean_x_idx
                    grid_x = x_min + mean_x_idx
                    grid_y = y_min + mean_y_idx
                    grid_z = z_min + mean_z_idx

                    # 四舍五入为整数格位置
                    grid_x_i = int(round(grid_x))
                    grid_y_i = int(round(grid_y))
                    grid_z_i = int(round(grid_z))

                    obj = {
                        'name': label,
                        'ACC': True,
                        'RELx': -grid_z_i,
                        'RELy': -grid_y_i,
                        'RELz': -grid_x_i,
                        'size': n,
                        'coords': coords,
                    }
                    obj_list.append(obj)
    print("Precise detected objects from around info:")
    for obj in obj_list:
        print(f"Object: {obj['name']}, Position: (y:{obj['RELy']}, z:{obj['RELz']}, x:{obj['RELx']}), Size: {obj['size']}")
    return obj_list

def process_detect_from_json(entity, json_path="detection_output_kimi.json"):
    obj_list = []
    with open(json_path, 'r', encoding='utf-8') as json_file:
        detection_data = json.load(json_file) # detection_data 是一个列表
        print("Detected objects and their depth information:")
        # for obj in detection_data.get('objects', []): 
        for obj in detection_data:
            name = obj.get('label', 'unknown')
            
            o_x, o_y, o_z = entity_pos2obj_pos(entity, obj)
            
            print(f"Object: {name}, Position: ({o_x}, {o_y}, {o_z})")
            # 根据当前xy值和识别到的物体深度计算物体的绝对位置-粗略的-后续根据“雷达”信息精确定位
            obj_list.append(
                {
                    'name': name,
                    'ACC': False,
                    'x': o_x,
                    'y': o_y,
                    'z': o_z
                }
        )
    return obj_list

def craft_check(action, inventories_bef, inventories):
    # 检查 action 中需要制作的 item,并将 inventories_bef  与 inventories 进行对比,判断是否制作成功
    # 提取 item
    item_name = action.split(' ', 1)[1] if ' ' in action else ''
    if item_name == '':
        return False, "No item specified in action"
    # 获取制作前后该物品的数量
    count_bef = 0
    count_aft = 0
    for item in inventories_bef:
        if item['item'] == item_name:
            count_bef += item['size']
    for item in inventories:
        if item['item'] == item_name:
            count_aft += item['size']
    if count_aft > count_bef:
        # 如果成功返回 item 制作成功 
        return True, f"Crafted {item_name} successfully"
    else:
        # TODO 否则返回错误原因
        
        return False, f"Failed to craft {item_name}, possibly due to insufficient materials"

def craft_diff_get(inventories_bef, inventories)->{dict, dict}:
    # 对比 inventories_bef 和 inventories, 返回制作的物品和数量
    item_counts_bef = {}
    item_counts_aft = {}
    
    input_items = {}
    output_items = {}
    
    for item in inventories_bef:
        item_name = item['item']
        item_size = item['size']
        item_counts_bef[item_name] = item_size
    
    for item in inventories:
        item_name = item['item']
        item_size = item['size']
        item_counts_aft[item_name] = item_size
    
    # 计算差异
    for item_name, aft_size in item_counts_aft.items():
        bef_size = item_counts_bef.get(item_name, 0)
        if aft_size > bef_size:
            output_items[item_name] = aft_size - bef_size
            
    for item_name, bef_size in item_counts_bef.items():
        aft_size = item_counts_aft.get(item_name, 0)
        if aft_size < bef_size:
            input_items[item_name] = bef_size - aft_size
            
    
    return input_items, output_items

# 获取 中心点 的瞄准的物体 TODO 对准物体进行采集操作 同时不破坏物体 使用指令 chat /give @p minecraft:stick 64
def get_aimed_object(yaw, around, view_angle)->str:
    msg = "the current aimed object is "
    
    # y=0,x=0,z=0 为玩家所在格
    # y=1,x=0,z=0 为玩家视角所在格
    # 根据 entity 的 yaw 和 view_angle 计算玩家视角方向
    
    # yaw 0 是 z 轴正方向,yaw 180 是 z 轴负方向,
    # yaw 270 是 x 轴正方向,yaw 90 是 x 轴负方向。
    
    a_y = 1  # 视角所在层
    a_x = 0
    a_z = 0
    
    # 如果 a_y = 0
    item = "none."
    if view_angle == 0:
        # 根据 yaw 计算 x z 方向
        for i in range(1, 3):
            if yaw == 0:
                a_z += 1
            elif yaw == 90:
                a_x -= 1
            elif yaw == 180:
                a_z -= 1
            elif yaw == 270:
                a_x += 1
            # 查看 around 中该位置的物体
            item = around[a_y][a_z][a_x]
            if item != 'air':
                break
    elif view_angle == 2: # 向下看
        item = around[a_y][a_z][a_x]
    elif view_angle ==  -2: # 向上看
        for i in range(1, 3):
            a_y += 1
            # 查看 around 中该位置的物体
            item = around[a_y][a_z][a_x]
            if item != 'air':
                break
    elif view_angle == 1: # 斜向下看
        if yaw == 0:
            a_z = 1
            a_y = 1
            item = around[a_y][a_z][a_x]
            if item == 'air':
                a_y = 0
                item = around[a_y][a_z][a_x]
                if item == 'air':
                    a_z = 2
                    item = around[a_y][a_z][a_x]
                    if item == 'air':
                        a_y = -1
                        item = around[a_y][a_z][a_x]
        elif yaw == 90:
            a_x = -1
            a_y = 1
            item = around[a_y][a_z][a_x]
            if item == 'air':
                a_y = 0
                item = around[a_y][a_z][a_x]
                if item == 'air':
                    a_x = -2
                    item = around[a_y][a_z][a_x]
                    if item == 'air':
                        a_y = -1
                        item = around[a_y][a_z][a_x]
        elif yaw == 180:
            a_z = -1
            a_y = 1
            item = around[a_y][a_z][a_x]
            if item == 'air':
                a_y = 0
                item = around[a_y][a_z][a_x]
                if item == 'air':
                    a_z = -2
                    item = around[a_y][a_z][a_x]
                    if item == 'air':
                        a_y = -1
                        item = around[a_y][a_z][a_x]
        elif yaw == 270:
            a_x = 1
            a_y = 1
            item = around[a_y][a_z][a_x]
            if item == 'air':
                a_y = 0
                item = around[a_y][a_z][a_x]
                if item == 'air':
                    a_x = 2
                    item = around[a_y][a_z][a_x]
                    if item == 'air':
                        a_y = -1
                        item = around[a_y][a_z][a_x]
    
    elif view_angle == -1: # 斜向上看
        if yaw == 0:
            a_z = 1
            a_y = 1
            item = around[a_y][a_z][a_x]
            if item == 'air':
                a_y = 2
                item = around[a_y][a_z][a_x]
                if item == 'air':
                    a_z = 2
                    item = around[a_y][a_z][a_x]
                    
        elif yaw == 90:
            a_x = -1
            a_y = 1
            item = around[a_y][a_z][a_x]
            if item == 'air':
                a_y = 2
                item = around[a_y][a_z][a_x]
                if item == 'air':
                    a_x = -2
                    item = around[a_y][a_z][a_x]
        elif yaw == 180:
            a_z = -1
            a_y = 1
            item = around[a_y][a_z][a_x]
            if item == 'air':
                a_y = 2
                item = around[a_y][a_z][a_x]
                if item == 'air':
                    a_z = -2
                    item = around[a_y][a_z][a_x]
        elif yaw == 270:
            a_x = 1
            a_y = 1
            item = around[a_y][a_z][a_x]
            if item == 'air':
                a_y = 2
                item = around[a_y][a_z][a_x]
                if item == 'air':
                    a_x = 2
                    item = around[a_y][a_z][a_x]
    
    
    msg += item + "." + f"rel position is x={a_x}, y={a_y}, z={a_z}."
    print(msg)
    
    
    return item, msg

def get_hotbar_item(inventories, hotbar_id)->dict:
    # hotbar_id 从 0 开始
    for item in inventories:
        if item['id'] == hotbar_id:
            return item
    return {}

# 划分 action sequence
def split_action_sequence(action_sequence):
    """
    {
        0:move 1
        7:attack
        4:look 1
        7:attack
        4:look -1
        4:look -1
        7:attack
        4:look 1
    }
    """
    actions = []
    action_lines = re.findall(r'\{([^}]*)\}', action_sequence, re.DOTALL)
    for action_line in action_lines:
        individual_actions = action_line.split("\n")
        for act in individual_actions:
            act = act.strip()
            if act:
                actions.append(act)
    return actions

def load_scene_info(json_path):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        scene_info = json.load(json_file)
    
    # 删除 entity_graph entities /temp children 
    entity_graph = scene_info.get("entity_graph", {})
    if entity_graph:
        entity_graph["entities"]["/temp"]["children"] = []
        entity_graph["graph_structure"]["children"]["temp"]["children"] = {}
        for child in list(entity_graph["entities"].keys()):
            if child.startswith("/temp/"):
                del entity_graph["entities"][child]
    
    return scene_info

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
    parser.add_argument('--LLM', type=str, default='enable', help="enable or disable LLM")
    parser.add_argument('--MEM', type=str, default='enable', help="enable or disable MEM")
    parser.add_argument('--DETECT', type=str, default='enable', help="enable or disable MEM")
    parser.add_argument('--userinput', type=str, default='disable', help="enable or disable user input")
    
    
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server
    
    LLM_MODE = False
    MEM_MODE = True
    DETECT_MODE = False
    USERINPUT_MODE = False
    SUBMISSION_MODE = False
    # if args.LLM.lower() == 'enable':
    #     LLM_MODE = True
    # if args.MEM.lower() == 'enable':
    #     MEM_MODE = True
    # if args.DETECT.lower() == 'enable':
    #     DETECT_MODE = True
    # if args.userinput.lower() == 'disable':
    #     USERINPUT_MODE = False

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
    # <Grid name="around">
    #     <min x="-2" y="-2" z="-2" />
    #     <max x="2" y="2" z="2" />
    # </Grid>
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
    
    # # TODO 创建当前场景记忆 初始化历史记忆
    if MEM_MODE == True:
        cs = CurrentState()
        # 如果存在历史记忆那先读取历史记忆 并 删除 历史 temp 记忆
        if os.path.exists(f'scene_info.json'):
            scene_info = load_scene_info('scene_info.json')
        else:
            scene_info = mc_cap2scene_info(env.actions, env.actions_type, act_info_en, around_range)
        
        cs.init_Scene(scene_info)
        
    # 在当前目录下创建log文件夹,并获取当前时间作为log文件名
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
    rule_prompt = viewinfo_en + craftitem_en + str(craft_requirements) + "\n"
    
    # obs_prompt = f'\n 玩家周围的观测网格信息是 {around_range}, 观测为三维,第一维为y轴（第一层为y轴大的）,第二维为z轴（第一层为z轴大的）,第三层为x轴（第一层为x轴大的）\n'
    obs_prompt = f'\n the observation grid info around player is: {around_range},' + around_info_en + '\n'
        
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
        user_request = 'Make wooden axe'
        
        if user_request.lower() == 'exit':
            print("Exiting the experiment.")
            break
        
        # 获取初始化环境信息
        env.render()
        action = "jump 1"
        obs, reward, done, info = env.step_diy(action)
        inventories, around, entity = info_process(env, info)
        
        #进行图像识别
        obj_list = get_around_objects_precise_pos(entity, around, around_range)
        save_img(obs, env)
        if DETECT_MODE == True:
            test5_kimiV2()
            detect_obj = process_detect_from_json(entity)
            for do in detect_obj:
                obj_list.append(do)

        user_request_init = f"\nThe player's current inventory is: {inventories}\n"
        user_request_init += f"The player's current position and orientation is: {entity}\n"
        user_request_init += around_msg(around)
        user_request_init += f"Detected objects : {obj_list}\n"
        aimed_object, aimed_object_msg = get_aimed_object(entity.get('yaw'), around, env.view_angle)
        
        user_request_init += aimed_object_msg

        # TODO 根据 记忆 检索相关信息加入 prompt
        if MEM_MODE == True:
            # 更新短期空间记忆 + 短期-》长期 + 更新当前 空间 场景记忆 + 检索相关信息
            scene_info = record_short_space_memory(scene_info, obj_list, entity)
            # scene_info = short2long_space_memory(entity, around, scene_info)
            cs.update_Scene(scene_info)
            retrieval_rel_ans = cs.retrieval_Request(user_request+user_request_init)
            # 在 scene_info 中 对 retrieval_rel_ans 进行对比 ，找到相应的节点 
            rel_info = ""
            for node in retrieval_rel_ans:
                node_name = node.get('name', '')
                
                # 查看 entity_graph entities 中是否存在该节点 - 空间
                if node_name in scene_info.get('entity_graph', {}).get('entities', {}):
                    print(f"Found related spatial node in memory: {node_name}")
                    rel_info += f"\nrel info {scene_info.get('entity_graph', {}).get('entities', {}).get(node_name, {})}\n"
                # 查看 skill_specs 中是否存在该节点 - 技能
                elif node_name in scene_info.get('skill_specs', {}):
                    print(f"Found related skill node in memory: {node_name}")
                    rel_info += f"\nrel info {scene_info.get('skill_specs', {}).get(node_name, {})}\n"
            if rel_info != "":
                prompt += f"\nRelevant information from memory:{rel_info}\n"
        
        # 通过 llm 生成一系列动作
        # TODO 子任务拆解
        sub_mission = ""
        sub_mission_list = []
        if SUBMISSION_MODE == True:
            user_request_init += sub_mission_prompt_en
            sub_mission_list, messages = client.query_request(query="Decompose the following task into several sub-tasks: " + user_request,
                                                            info=user_request_init,
                                                            safe_rule=None,
                                                            prompt=prompt)
        else :
            sub_mission = user_request
            sub_mission_list.append(sub_mission)
        
        for sub_mission in sub_mission_list:
            print(f"Starting sub-mission: {sub_mission}")
            with open(log_file, 'a') as f:
                f.write(f"\nStarting sub-mission: {sub_mission}\n")
            
            
            sub_mission_init = f"\nThe player's current inventory is: {inventories}\n"
            sub_mission_init += f"The player's current position and orientation is: {entity}\n"
            aimed_object, aimed_object_msg = get_aimed_object(entity.get('yaw'), around, env.view_angle)
            sub_mission_init += aimed_object_msg
            sub_mission_init += around_msg(around)
            sub_mission_init += f"Detected objects : {obj_list}\n"
            
        
            action_sequence = []
            record_actions = []
            if LLM_MODE == True:
                action_sequence, messages = client.query_request(query=sub_mission,
                                                                info=sub_mission_init,
                                                                safe_rule=None,
                                                                prompt=prompt)
            
            user_input = ""
            
            while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):

                # add 根据当前环境和用户指令生成一系列动作
                action = 0
                
                if LLM_MODE and (action_sequence is None or len(action_sequence) == 0 or len(action_sequence) > 50):
                    print("No action sequence generated, exiting the episode.")
                    break
                elif LLM_MODE == False and USERINPUT_MODE == False:
                    # 读取文件中的 action_sequence
                    user_input = input("Enter action sequence in input_action.txt, or 'q' to quit: ")
                    if user_input.lower() == 'q':
                        user_input = 'q'
                        break
                    with open('input_action.txt', 'r') as f:
                        file_content = f.read()
                    action_sequence = split_action_sequence(file_content)
                    if len(action_sequence) == 0:
                        print("No valid actions entered, please try again.")
                        continue
                elif USERINPUT_MODE == True:
                    # 手动输入 action_sequence
                    user_input = input("Enter action sequence (format:{4:look 1}), or 'q' to quit: ")
                    if user_input.lower() == 'q':
                        user_input = 'q'
                        break
                    action_sequence = parse_action_sequence(user_input)
                    if len(action_sequence) == 0:
                        print("No valid actions entered, please try again.")
                        continue
                
                # 遍历 action_sequence
                cur_act_msg = ""

                for act in action_sequence:
                    # 解析动作字符串
                    act_idx, act_str = parse_action_string(act)
                    if act_idx is None:
                        print(f"Invalid action format: {act}, skipping.")
                        with open(log_file, 'a') as f:
                            f.write(f"Invalid action format: {act}, skipping.\n")
                        cur_act_msg += f"Invalid action format: {act}, the right format is 0:move 1.\n"
                        continue
                    action = act_str
                    
                    # 调试：用户决定是否执行 每5步
                    if USERINPUT_MODE == False and (steps+1) % 30 == 0:
                        print("enter to continue, input 'q' to quit:")
                        user_input = input(":")
                        if user_input.lower() == 'q':
                            break
                    print("\n" * 5)
                    
                    # 执行动作
                    with open(log_file, 'a') as f:
                        f.write("diy action: " + action + '\n')
                    print("diy action: " + action)
                    
                    env.render()
                    
                    
                    # TODO check inventory
                    obs, reward, done, info = env.step_diy(action)
                    
                    steps += 1

                    print("action: " + str(act_str))
                    # print("reward: " + str(reward))
                    # print("done: " + str(done))
                    inventories_bef = inventories
                    aimed_object_bef = aimed_object
                    obj_list_bef = obj_list.copy()
                    inventories, around, entity = info_process(env, info)
                    aimed_object, aimed_object_msg = get_aimed_object(entity.get('yaw'), around, env.view_angle)

                    cur_act_msg += around_msg(around)
                    cur_act_msg += aimed_object_msg
                    
                    # 将以上信息写入action.log
                    with open(log_file, 'a') as f:
                        f.write("action: " + str(action) + '\n')
                        f.write('reward: ' + str(reward) + '\n')
                        f.write('done: ' + str(done) + '\n')
                        # f.write('obs: ' + str(obs) + '\n')
                        f.write('Inventory: ' + str(inventories) + '\n')
                        f.write('around: ' + str(around) + '\n')
                        f.write('entity: ' + str(entity) + '\n')
                        f.write('-------------------------\n')
                        

                    

                    # LLM 做法
                    if LLM_MODE == True:
                        # 更新 cur_act_msg
                        cur_act_msg += f"action :{act_str}, entity info :{entity}\n"

                    # TODO 如果环境没有改变 则不更新 obs # 保存图像
                    if "inventory" not in act_str and "hotbar" not in act_str and "craft" not in act_str:
                        
                        save_img(obs, env)
                        
                        print("---------detect info---------")
                        # 根据 around 信息更新 obj_list 中物体的精确位置  对 obj_list 中物体进行精确定位 如果是在 around 范围内的物体 则进行精确定位 TODO 待测试 需要完善删除机制
                        obj_list = get_around_objects_precise_pos(entity, around, around_range)
                        
                        if DETECT_MODE == True:
                            test5_kimiV2()
                            
                            # 读取json文件打印识别到的物体和深度信息
                            detect_obj = process_detect_from_json(entity)
                            for do in detect_obj:
                                obj_list.append(do)
                        
                        cur_act_msg += f"Detected objects : {obj_list}\n"

                    elif "inventory" in act_str or "craft" in act_str:
                        cur_act_msg += f"Inventory info : {inventories}\n"
                        if "craft" in act_str:
                            # craft 相关动作 进行特殊处理,检查材料是否充足 如果不足则跳过,加提示词 或者 是检测生成物是否增加
                            craft_success, craft_msg = craft_check(action, inventories_bef, inventories)
                            cur_act_msg += f"Craft check info : {craft_msg}\n"
                    
                    
                        
                    if MEM_MODE == True: 
                        # TODO LLM+MEM 做法(包括 MEM 做法)
                        mem_record = mem_generation(action, inventories_bef, aimed_object_bef, obj_list_bef, inventories, obj_list, entity, env)
                        # TODO 需要记录之前的 信息 包括 aimed_obj inventories
                        record_actions.append(mem_record)
                        
                        # 更新短期空间记忆 + 短期-》长期 + 更新当前 空间 场景记忆 + 检索相关信息
                        scene_info = record_short_space_memory(scene_info, obj_list, entity)
                        # scene_info = short2long_space_memory(entity, around, scene_info)
                        cs.update_Scene(scene_info)
                        cur_act_msg += str(cs.retrieval_Request(sub_mission))
                        
                    time.sleep(1)

                # 整体退出
                if user_input.lower() == 'q':
                    break
                
                # 更新 llm messages 并根据当前场景继续完成任务
                if LLM_MODE == True:
                    messages.append({"role": "user", "content": cur_act_msg})
                    action_sequence, messages = client.query_request(messages=messages)
                    
           
                
            # TODO 判断该子任务是否完成 - 由 LLM 判断
            print(f"Sub-mission '{sub_mission}' ended.")
            # 如果完成则 将 record_actions 转换成 情景记忆 MEM
            if MEM_MODE == True:
                scene_info = skill2FIXED_mem(sub_mission, record_actions, scene_info)
                # 不用 update_Scene 会更新CS中的信息吗
                cs.update_Scene(scene_info)
                print("Memory updated with sub-mission actions.")
                
             # 整体退出
            if user_input.lower() == 'q':
                break
            # 打印messages最后一个 content 
            if LLM_MODE == True:
                print(messages[-1]['content'] if messages else "No messages.")
        # TODO user 任务 的记忆 记录 submission 
        
        # 整体退出
            if user_input.lower() == 'q':
                break

    env.close()


