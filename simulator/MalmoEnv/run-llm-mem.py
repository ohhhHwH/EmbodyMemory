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
import xml.etree.ElementTree as ET

# 找到 brain 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brain.brain_deepseek import *
from memory.memory import *


    
# 加入 LLM
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

obj_prompt_en_mc = '''
在MC中的一个玩家，你将回答当前场景中的物体位置与其坐标信息，
根据环境的信息判断除分类器以外的物体的坐标及其属性，并将其打包成一个 json 格式返回

    as a player in minecraft, you will answer questions about the positions and coordinates of objects in the current scene.
    you will determine the coordinates and attributes of objects other than the classifier based on the information in the environment,
    and return them packaged in a json format.
    

'''

# 从 mission xml 中解析 ObservationFromGrid 的 min/max
def parse_observation_grid(xml_text, grid_name=None):
    """
    返回第一个匹配的 grid 的信息：{'name': str, 'min': (x,y,z), 'max': (x,y,z)}
    如果未找到返回 None。
    """
    ns = {'m': 'http://ProjectMalmo.microsoft.com'}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    for obs_grid in root.findall('.//m:ObservationFromGrid', ns):
        grid = obs_grid.find('m:Grid', ns)
        if grid is None:
            continue
        name = grid.get('name')
        if grid_name is not None and name != grid_name:
            continue
        min_elem = grid.find('m:min', ns)
        max_elem = grid.find('m:max', ns)
        if min_elem is None or max_elem is None:
            continue
        try:
            xmin = int(min_elem.get('x'))
            ymin = int(min_elem.get('y'))
            zmin = int(min_elem.get('z'))
            xmax = int(max_elem.get('x'))
            ymax = int(max_elem.get('y'))
            zmax = int(max_elem.get('z'))
        except (TypeError, ValueError):
            # 属性缺失或非整数
            continue
        return {'name': name, 'min': (xmin, ymin, zmin), 'max': (xmax, ymax, zmax)}
    return None

# 加入 memory
# 根据info来更新 memory
# 将所有可用技能转换成 json 格式 == scene_info
def mc_cap2scene_info(actions, actions_type, grid_info):
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
                "children": ["/entity"]
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
            }
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

def info2json(info):
    # TODO
    update_info = {}
    
    # 将 info 字符串 转成 info 字典
    parsed_info = {}
    if info:
        if isinstance(info, dict):
            parsed_info = info
        else:
            try:
                parsed_info = json.loads(info)
            except Exception:
                try:
                    parsed_info = ast.literal_eval(info)
                except Exception:
                    parsed_info = {}
    info = parsed_info
    
    
    
    # 打印出 info 字典的 around 信息 - 需要跟 xml 中 ObservationFromGrid 的 name 一致
    info_board = info.get('around', 'N/A')
    
    return update_info

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
    
    action_filter = {"move", "turn", "use", "attack", "look", "jump"}
    
    # 将xml中内容传入 并 解析
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
    log_file = log_dir / f'action_{time.strftime("%Y%m%d_%H%M%S")}.log'
    
    
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
    api_key = os.getenv("API_KEY")
    client = MCPClient(api_key=api_key)
    
    # ADD:获取当前环境信息并更新 - 聚类，将环境中能聚在一起的物体整合成一个 box 迁出一个索引 - 不对exfilter中的东西进行建图or只对filter中的东西建图
    # 解析 observation grid
    grid_info = parse_observation_grid(xml, grid_name="around")
    # 排除的物体
    exfilter = {"air", "water", "leaves2", "stone", "grass", "dirt", "sand"}
    
    cs = CurrentState()
    skills_memory = mc_cap2scene_info(env.actions, env.actions_type, grid_info)
    cs.init_Scene(skills_memory)

    for i in range(args.episodes):
        print("reset " + str(i))
        obs = env.reset()
        
        # 打开action.log将写入 episode i
        with open(log_file, 'a') as f:
            f.write('\n\n\n\nepisode ' + str(i) + '\n')
            f.write('======================\n')
            
            
        # 获取用户指令
        user_request = 'find water'
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
                
        # 获取所有可用actions, 生成 ["0:move 1", "1:move -1"]
        # all_actions = [act for act in env.action_space]
        # 'the available actions:\n0:move 1 [DiscreteMovement]\n1:move -1 [DiscreteMovement]\n2:turn 1 [DiscreteMovement]\n3:turn -1 [DiscreteMovement]\n4:jump 1 [DiscreteMovement]\n5:look 1 [DiscreteMovement]\n6:look -1 [DiscreteMovement]\n7:attack 1 [DiscreteMovement]\n8:use 1 [DiscreteMovement]\n9:jump 1 [ContinuousMovement]\n10:jump 0 [ContinuousMovement]\n11:move 1 [ContinuousMovement]\n12:move -1 [ContinuousMovement]\n13:turn 1 [ContinuousMovement]\n14:turn -1 [ContinuousMovement]\n15:attack 1 [ContinuousMovement]\n16:attack 0 [ContinuousMovement]\n17:use 1 [ContinuousMovement]\n18:use 0 [ContinuousMovement]'
        actions_prompt = "the available actions:\n" + ", ".join(f"{i}:{act} [{act_type}]" for i, (act, act_type) in enumerate(zip(env.actions, env.actions_type)))
        
        prompt = system_prompt_en_mc + actions_prompt
        
        prompt += f'\n the observation grid info around player is: {grid_info}\n'
        
        # 通过 memory 进行检索
        retrieval_Request = f"Based on the current scene, {user_request}"
        retrieval_Results = cs.retrieve(retrieval_Request, topk=5)
        
        # 通过 llm 生成一系列动作
        action_sequence, messages = client.query_request(query=user_request,
                                                          info=None,
                                                          safe_rule=None,
                                                          prompt=prompt)


        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):

            # add 根据当前环境和用户指令生成一系列动作
            # ['"entity":0:move 1', '"entity":1:move -1', '"entity":2:turn 1', '"entity":3:turn -1', '"entity":4:jump 1', '"entity":5:look 1', '"entity":6:look -1', '"entity":7:attack 1', '"entity":8:use 1']
            action = 0
            print("debug Generated action sequence:", action_sequence)
            
            
            
            if action_sequence is None or len(action_sequence) == 0:
                print("No action sequence generated, exiting the episode.")
                
                # TODO : 检测 任务 是否完成 如果完成，总结并作为一个 长期情景节点 保存
                
                break
            
            # 遍历 action_sequence
            cur_act_msg = ""
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
                

                
                                
                # 将info 转换成统一的json格式
                info = info2json(info, exfilter)
                
                # 根据 info 更新 memory - update_Scene
                cs.update_Scene(info, exfilter)
                
                # retrieve from memory
                retrieval_Request = f"Based on the current scene, {user_request}"
                retrieval_Results = cs.retrieve(retrieval_Request, topk=5)
                print("Retrieval Results:", retrieval_Results)
                
                # 生成 下次动作的提示词
                cur_act_msg += f"action {act_str}, info {retrieval_Results}\n"
                
                if args.saveimagesteps > 0 and steps % args.saveimagesteps == 0:
                    h, w, d = env.observation_space.shape
                    img = Image.fromarray(obs.reshape(h, w, d))
                    img.save('image' + str(args.role) + '_' + str(steps) + '.png')

                time.sleep(1)
            
            messages.append({"role": "user", "content": cur_act_msg})
            action_sequence, messages = client.query_request(query=user_request, messages=messages)
            
        # 打印messages最后一个 content 
        print(messages[-1]['content'] if messages else "No messages.")

    env.close()


