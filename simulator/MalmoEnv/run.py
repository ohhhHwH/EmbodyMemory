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

# 加入 memory
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str("/home/hyl/robonix"))
from vlm_detect import test3_kimi

def get_observation_grid_range(source, grid_name=None):
    """
    从 Mission XML（字符串或文件路径）解析 ObservationFromGrid 中 Grid 的 min/max 范围。
    参数:
      - source: XML 字符串或 pathlib.Path/文件路径字符串
      - grid_name: 可选，指定 Grid 的 name 属性，若为 None 返回第一个匹配的 Grid
    返回:
      - dict: {'name': str, 'min': (xmin,ymin,zmin), 'max': (xmax,ymax,zmax)}
      - 若未找到返回 None
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path

    # 载入 xml 文本
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
            row = row[::-1]
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
        
    # 打印视野中心的深度值
    if depth is not None:
        center_depth = depth[h // 2, w // 2]
        print(f"depth in view: {center_depth}")

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
    
    # 添加 动作过滤器 所有动作都在这个范围内
    action_filter = {"move", "turn", "jump", "look", "jumpmove", "attack"}
    # action_filter = {"move", "jumpmove", "strafe", "jumpstrafe", "turn", "movenorth", "moveeast",
    #                 "movesouth", "movewest", "jumpnorth", "jumpeast", "jumpsouth", "jumpwest",
    #                 "jump", "look", "attack", "use", "jumpuse"}
    
    # 获取 xml 中ObservationFromGrid的 around 范围
    around_range = get_observation_grid_range(args.mission, grid_name='around')
    print(f"Around range from XML: {around_range}")

    # 将xml中内容传入并接卸
    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode,
             action_filter=action_filter,
             resync=args.resync)

    # 在当前目录下创建log文件夹，并获取当前时间作为log文件名
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)
    # log_file = log_dir / f'action_{time.strftime("%Y%m%d_%H%M%S")}.log'
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
        
    
    # # 删除 log 文件
    # if log_file.exists():
    #     log_file.unlink()
    #     print(f"Deleted existing log file: {log_file}")
    # # 调试退出
    # exit(0)

    for i in range(args.episodes):
        print("reset " + str(i))
        obs = env.reset()
        
        # 打开action.log将写入 episode i
        with open(log_file, 'a') as f:
            f.write('\n\n\n\nepisode ' + str(i) + '\n')
            f.write('======================\n')
            
            
        # ADD : 获取用户指令
        user_request = ''
        # user_request = input("Press Enter to continue, or type 'exit' to quit: ")
        if user_request.lower() == 'exit':
            print("Exiting the experiment.")
            break
        else:
            print("Continuing the experiment.")
            
            

        # add : 根据 env.actions 建立 Graph
        

        steps = 0
        done = False
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            # ADD:获取当前环境信息并更新 - 聚类，将环境中能聚在一起的物体整合成一个 box 迁出一个索引
            env.render()
            
            # 这里是随机指令
            # action = env.action_space.sample()
            
            # add 根据当前环境和用户指令生成一系列动作
            action_sequence = []

                
            # 调试 根据用户输入数字进行相应的操作
            # 打印索引并询问用户输入索引
            print("Available actions:")
            
            # 每5个换行
            for i, act in enumerate(env.action_space):
                print(f"{i}: {act}", end='\t')
                if (i + 1) % 5 == 0:
                    print()

            user_input = input("q")
            if user_input.lower() == 'q':
                break
            if not user_input.isdigit() or int(user_input) < 0 or int(user_input) >= len(env.actions):
                action = env.action_space.sample()
            else:
                action = int(user_input)
                
            
                
            # 将内容不删除但清空终端
            # print("\033c", end="")  # ANSI escape code to clear the terminal
            print("\n" * 5)
            
                
            # 打开action.log将当前指令写入文件
            with open(log_file, 'a') as f:
                f.write("action: " + str(action) + ',' + env.action_space[action] + '\n')
                
            print(action , str(type(action)))
            
            obs, reward, done, info = env.step(action)
            steps += 1
                
            print("action: " + str(action) + ',' + env.action_space[action])
            print("reward: " + str(reward))
            print("done: " + str(done))
            # 将 info 字符串 转成 info 字典
            info = eval(info)
            # 打印出 info 字典的 around 信息
            around = info.get('around', None)
            around = info_observation_grid_range_reserve(around, around_range)
            print("info around: " )
            for layer in around:
                print(layer, "len:", len(layer))
            # 获取entities -> list 打印 xyz yaw pitch
            entities = info.get('entities', [])
            for entity in entities:
                # print(f"xyz: ({entity.get('x')}, {entity.get('y')}, {entity.get('z')})")
                # print(f"yaw: {entity.get('yaw')}")
                # print(f"pitch: {entity.get('pitch')}")
                
                # 遍历 entity 的键值对
                print("Entity details:")
                for key, value in entity.items():
                    print(f"  {key}: {value}")
                    
                print("-------------------")
            
            
            # 取第一个 entity 作为参考
            entity = entities[0] if entities else {}
            
            # 保存图像
            save_img(obs, env)
            
            
            
            # 加载 mc 模型 识别图像信息
            # test3_kimi()
            
            # 读取json文件打印识别到的物体和深度信息
            # json_output_path = "detection_output_kimi.json"
            # obj_list = []
            # with open(json_output_path, 'r', encoding='utf-8') as json_file:
            #     detection_data = json.load(json_file) # detection_data 是一个列表
            #     print("Detected objects and their depth information:")
            #     # for obj in detection_data.get('objects', []): 
            #     for obj in detection_data:
            #         name = obj.get('label', 'unknown')
            #         depth = obj.get('depth', 'unknown')
            #         print(f"Object: {name}, Depth: {depth}")
            #         # 根据当前xy值和识别到的物体深度计算物体的绝对位置-粗略的-后续根据“雷达”信息精确定位
            #         obj_list.append(
            #             {
            #                 'name': name,
            #                 'depth': depth,
            #                 'x': entity.get('x'),
            #                 'y': entity.get('y'),
            #                 'z': entity.get('z')
            #             }
            #         )

            # 将以上信息写入action.log 图像存入 malmo_obs.png
            with open(log_file, 'a') as f:
                f.write('reward: ' + str(reward) + '\n')
                f.write('done: ' + str(done) + '\n')
                f.write('info: ' + str(info) + '\n')
                # f.write('detected objects:'+str(obj_list)+'\n')
                f.write('-------------------------\n')
                
    env.close()
