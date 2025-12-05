import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
import itertools
from enum import IntEnum
import json
from collections import deque

from memory.modules.skg.skg import * 
from memory.modules.retriever.retriever import *

from memory.modules.memory.memory import *



class NodeType(IntEnum):
    SHORT_TERM = 0   # short-term memory
    LONG_TERM  = 1   # long-term memory
    FIXED      = 2   # pinned memory, not forgotten

class NodeClass(IntEnum):
    CONTEXT   = 0      # space
    TIME    = 1      # time - ordered sub-nodes
    SPACE = 2      # contextual

@dataclass
class MemoryNode:
    # structure for memory nodes
    node_id: int
    node_type: NodeType
    node_class: NodeClass
    name: str = "" # Node name
    summary: str = "" # Summary of the node and its immediate child nodes - GPT summary / A has a1 a2 a3
    parent_id: int = -1 # Parent node ID
    weight: float = 1.0 # Node weight. Less than 0 indicates an error node?
    child_cnt: int = 0 # Has child nodes
    parents_cnt: int = 0 # Number of parent nodes
    
    # child_list: List[int] = None
    # parent_list: List[int] = None

    # embedding: np.ndarray # Stored in a separate graph
    # optional attributes for specific node types
    # timestamp: float = "" # timestamp for time nodes
    # x: float = 0.0        # optional - spatial node coordinates
    # y: float = 0.0
    # z: float = 0.0

class MemoryGraph:
    def __init__(self, map_type: int = 0, id_start: int = 0, max_id: int = 1000):
        self.G = nx.DiGraph()
        self._id_counters = itertools.count(id_start)  # short start from id_start ; long start from 1000
        self.max_id = max_id  # max ID
        self.time_threshold = 3600  # short-term memory forgetting threshold in seconds
        self._node_map = {} # auxiliary index to store nodes by ID 
        
    def forget_node(self)-> bool:
        # Short-term memory forgetting logic
        # Implement timestamp-based forgetting strategy
        if self.map_type == 0:
            # Short-term memory
            current_time = np.datetime64('now')
            for node_id, node in list(self._node_map[0].items()):
                if (current_time - node.timestamp) > self.time_threshold:
                    self.delete_node(node_id)
            return True
        else:
            # Long-term memory does not perform forgetting
            print("Long-term memory does not perform forgetting processing")
            return True

    def _get_next_id(self) -> int:
        # Get the next available ID
        next_id = next(self._id_counters)
        if next_id >= self.max_id:
            # Short-term memory performs overflow forgetting processing
            self.forget_node()
            next_id = next(self._id_counters)
        
        return next_id

    def add_node(self, node_type: NodeType, node_class: NodeClass,
                    name: str, summary: str,
                    parent_id: int = -1,
                    timestamp: int = -1.0,
                    weight: float = 1.0,
                    child_cnt: int = 0,
                    parents_cnt: int = 0,
                    # ordered: bool = False,
                    x: float = 0.0, y: float = 0.0, z: float = 0.0) -> int:
        # Add a node to the graph
        if node_type not in [0, 1, 2]:
            raise ValueError("node_type must be 0 (short-term) or 1 (long-term)")
        if node_class not in [0, 1, 2]:
            raise ValueError("node_class must be 0 (spatial), 1 (temporal-ordered), or 2 (contextual)")
        node_id = self._get_next_id()
        node = MemoryNode(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            name=name,
            summary=summary,
            parent_id=parent_id,
            weight=weight,
            child_cnt=child_cnt,
            parents_cnt=parents_cnt
        )

        # Add to the graph
        self.G.add_node(node_id, **node.__dict__)


        # add level information
        self.G.nodes[node_id]['level'] = 0
        if parent_id != -1:
            self.G.nodes[node_id]['level'] = self.G.nodes[parent_id]['level'] + 1

        if node_class == NodeClass.TIME :
            # If it is a short-term time node, set it as ordered
            # get the parent node's child count and update the parent's child count status
            if timestamp < 0:
                timestamp = self.get_child_num(parent_id) + 1
                # update the parent node's child count
                self.update_node(parent_id, child_cnt=timestamp)
            self.G.nodes[node_id]['timestamp'] = timestamp

        if node_class == NodeClass.CONTEXT and node_type == NodeType.SHORT_TERM :
            # If it is a short-term node, set the timestamp
            self.G.nodes[node_id]['timestamp'] = timestamp if timestamp >= 0 else np.datetime64('now')

        if node_class == NodeClass.SPACE:
            # If it is a spatial node, set coordinates
            self.G.nodes[node_id]['x'] = x
            self.G.nodes[node_id]['y'] = y
            self.G.nodes[node_id]['z'] = z

        # Update auxiliary indexes
        self._node_map[node_id] = node
        # Update parent-child relationships when initial node
        if parent_id != -1 and child_cnt == 0 and parents_cnt == 0:
            self.add_child(parent_id, node_id)
        elif parent_id != -1:
            self.G.add_edge(parent_id, node_id)
        return node_id

    def delete_node(self, node_id: int):
        # Delete the node and its associated relationships TODO ONLY check the CONTEXT node
        if node_id in self._node_map:
            node = self._node_map[node_id]
            # Delete child node relationships
            if node.child_cnt:
                # Based on child nodes in the graph
                for child_id in list(self.G.successors(node_id)):
                    self._node_map[child_id].parents_cnt -= 1
                    if self._node_map[child_id].parents_cnt == 0:
                        self.delete_node(child_id) # Recursively delete invalid child nodes
            # Remove the node from the graph
            self.G.remove_node(node_id)
            # Update auxiliary indexes 
            del self._node_map[node_id]

    # TODO: Update basic information + graph information
    def update_node(self, node_id: int, **kwargs):
        # Update node attributes

        if node_id in self._node_map:  # Corrected typo: [nt] -> [] (assuming nt was a typo)
            node = self._node_map[node_id]
            for k, v in kwargs.items():
                if hasattr(node, k):
                    setattr(node, k, v)
            # Update graph attributes
            attrs = {k: v for k, v in node.__dict__.items() 
                    if k not in ['children', 'node_id']}  # Corrected typo: ['children'] added (assuming original intent)
            self.G.nodes[node_id].update(attrs)

    # Basic information
    def get_node(self, node_id: int) -> Optional[MemoryNode]:
        # Query node
        if node_id in self._node_map:
            return self._node_map[node_id]
        return None

        # Get information from the graph
    
    def get_graph_node(self, node_id: int) -> Dict:
        # Get graph node information
        if node_id in self.G.nodes:
            return self.G.nodes[node_id]
        return {}

    def find_nodes(self, **filters) -> Dict[int, MemoryNode]:
        # Query nodes with multiple conditions
        results = {}
        for nid in self._node_map:
            node = self._node_map[nid]
            if all(getattr(node, k, None) == v for k, v in filters.items()):
                results[nid] = node
        return results


    def add_child(self, parent_id: int, child_id: int):
        # Add a child node
        if parent_id not in self._node_map:
            raise ValueError(f"Parent node {parent_id} does not exist")
        if child_id not in self._node_map:
            raise ValueError(f"Child node {child_id} does not exist")
        # Get parent and child node objects
        parent_node = self._node_map[parent_id]
        child_node = self._node_map[child_id]
        parent_node.child_cnt += 1
        # Update the parent count of the child node
        child_node.parents_cnt += 1
        # Update the edge relationship in the graph
        self.G.add_edge(parent_id, child_node.node_id)

    def delete_child(self, parent_id: int, child_id: int):
        # Delete edge relationships 
        if parent_id in self._node_map and child_id in self._node_map:
            if self.G.has_edge(parent_id, child_id):
                self.G.remove_edge(parent_id, child_id)  # Corrected typo: remochild_ide_edge -> remove_edge
                # Update child node relationships
                parent = self._node_map[parent_id]
                child = self._node_map[child_id]
                if child.node_id in parent.child_ids:
                    child.parents_cnt -= 1
                    parent.child_ids.remove(child.node_id)  # Corrected typo: remochild_ide -> remove
                    if parent.parents_cnt == 0:
                        parent.child_cnt -= 1
                    # If the child node has no parent nodes left, delete the child node
                    if child.parents_cnt == 0:
                        self.delete_node(child.node_id)

    def get_child_num(self, node_id: int) -> int:
        # Get the number of child nodes for a given node
        if node_id in self._node_map:
            return self._node_map[node_id].child_cnt
        return 0

    def get_childs(self, node_id: int) -> List[int]:
        # Get child nodes
        if node_id in self._node_map:
            return list(self.G.successors(node_id))
        return []
    

    def save_to_file(self, file_path: str):
        # Save the graph to a file in JSON format
        data = []
        for node_id, node in self._node_map.items():
        # for node in self.G.nodes(data=True):
            node_data = {
                "id": node.node_id,
                "type": node.node_type.name,
                "class": node.node_class.name,
                "name": node.name,
                "summary": node.summary,
                "parent": node.parent_id,
                "weight": node.weight,
                "child_cnt": node.child_cnt,
                "parents_cnt": node.parents_cnt
            }
            if node.node_class == NodeClass.SPACE:
                node_data["x"] = self.G.nodes[node_id].get('x', 0.0)
                node_data["y"] = self.G.nodes[node_id].get('y', 0.0)
                node_data["z"] = self.G.nodes[node_id].get('z', 0.0)
            if node.node_class == NodeClass.TIME:
                node_data["timestamp"] = self.G.nodes[node_id].get('timestamp', -1.0)
            
            data.append(node_data)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # unuse : due to the need to rely on the id number to determine the parent-child relationship, this method is not used for the time being
    def load_from_file(self, file_path: str):
        # Load graph data from a file
        with open(file_path, 'r') as f:
            data = json.load(f)
        for node_data in data:
            node_id = node_data["id"]
            node_type = NodeType[node_data["type"]]
            node_class = NodeClass[node_data["class"]]
            name = node_data["name"]
            summary = node_data["summary"]
            parent_id = node_data.get("parent", -1)
            weight = node_data.get("weight", 1.0)
            child_cnt = node_data.get("child_cnt", 0)
            parents_cnt = node_data.get("parents_cnt", 0)
            print(f"Loading node {node_id}: type={node_type}, class={node_class}, name={name}, parent_id={parent_id}, weight={weight}, child_cnt={child_cnt}, parents_cnt={parents_cnt}")
            if node_class == NodeClass.SPACE:
                x = node_data.get("x", 0.0)
                y = node_data.get("y", 0.0)
                z = node_data.get("z", 0.0)
                # Create the node
                self.add_node(node_type, node_class, name, summary,
                            parent_id, weight=weight,
                            child_cnt=child_cnt, parents_cnt=parents_cnt,
                            x=x, y=y, z=z)
            elif node_class == NodeClass.TIME:
                timestamp = node_data.get("timestamp", -1.0)
                # Create the node
                self.add_node(node_type, node_class, name, summary,
                            parent_id, weight=weight,
                            child_cnt=child_cnt, parents_cnt=parents_cnt,
                            timestamp=timestamp)
            else:
                # Create the node without spatial or temporal attributes
                self.add_node(node_type, node_class, name, summary,
                        parent_id, weight=weight,
                        child_cnt=child_cnt, parents_cnt=parents_cnt)

    # Convert the graph to a visualization image and save it
    def visualize(self, file_path: str):
        # according to the level information self.G.nodes[node_id]['level'] to locate the position of the point, and increase the same interval for points at the same level
        plt.figure(figsize=(16, 10))
        pos = {}
        levels = {}
        for node_id, data in self.G.nodes(data=True):
            print(f"Node ID: {node_id}, Level: {data}")
            level = data['level']
            if level not in levels:
                levels[level] = []
            levels[level].append(node_id)
        # calculate the spacing for each level
        level_spacing = 1.0  
        for level, node_ids in levels.items():
            y = level * level_spacing
            for i, node_id in enumerate(node_ids):
                x = i * level_spacing
                pos[node_id] = (x, y)
        nx.draw(self.G, pos,
                node_size=700, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', arrows=True, arrowsize=20)
        labels = {node_id: f"{data['name']}" for node_id, data in self.G.nodes(data=True)}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8, font_color='black')
        plt.title("Memory Graph Visualization")
        plt.axis('off')
        plt.savefig(file_path, format='png', bbox_inches='tight')
        plt.close()

class CurrentState:
    def __init__(self, init_memory_graph: str = "basic_memory_graph.json"):
        
        self.memory_graph = MemoryGraph()
        # 添加根节点
        self.root_node_id = self.memory_graph.add_node(
            node_type=NodeType.FIXED,
            node_class=NodeClass.CONTEXT,
            name="Root",
            summary="Root of the memory graph",
            parent_id=-1,
            weight=1.0,
            child_cnt=0,
            parents_cnt=0
        )
        
        self.root_space_node_id = -1
        self.root_memory_node_id = -1
        
        # self.memory_graph.load_from_file(init_memory_graph)
        # self.current_space = ""

    def funcall_to_node(self, query : str, tool_calls : list) -> MemoryNode:
        
        # query 作为情景节点 - 短时间记忆
        query_node_id = self.memory_graph.add_node(
            node_type=NodeType.SHORT_TERM,
            node_class=NodeClass.CONTEXT,
            name=query,
            summary="",
            parent_id=-1,  # No parent for the root context node
            weight=1.0,
            child_cnt=len(tool_calls),
            parents_cnt=0,
            timestamp=np.datetime64('now')  # Current time as timestamp
        )
        # tool calls 作为情景节点下面的子时间节点 - 参数中带name的作为时间节点下的空间节点
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            # Create a time node for each tool call
            time_node_id = self.memory_graph.add_node(
                node_type=NodeType.SHORT_TERM,
                node_class=NodeClass.TIME,
                name=tool_name,
                summary=str(tool_args),
                parent_id=query_node_id,  # Parent is the query context node
                weight=1.0,
                child_cnt=0,  # No children for now
                parents_cnt=1,  # One parent (the query context node)
            )
            # Add the tool call as a child of the query context node
            self.memory_graph.add_child(query_node_id, time_node_id)

            # 解析args参数，提取地点信息
            if "name" in tool_args:
                name = tool_args["name"]
                # Create a spatial node for the state
                space_node_id = self.memory_graph.add_node(
                    node_type=NodeType.SHORT_TERM,
                    node_class=NodeClass.SPACE,
                    name=name,
                    summary=f"State information for {name}",
                    parent_id=time_node_id,  # Parent is the time node
                    weight=1.0,
                    child_cnt=0,  # No children for now
                    parents_cnt=1,  # One parent (the time node)
                )
                # Add the spatial node as a child of the time node
                self.memory_graph.add_child(time_node_id, space_node_id)

        # 地点参数等信息作为时间节点下关联的空间节点


        
        
        pass
    
    def init_Scene(self, scene_info: dict):
        with open("scene_info.json", "w") as f:
            json.dump(scene_info, f, indent=4)

        entity_graph = scene_info.get("entity_graph", {})
        skill_specs = scene_info.get("skill_specs", {})
        graph_structure = entity_graph.get("graph_structure", {})
        
        # print("Graph Structure:", graph_structure)

        # 根据graph_structure构建空间节点
        # "graph_structure": {
        #     "name": "/",
        #     "path": "/",
        #     "skills": [],
        #     "children": {...}
        # }

        # 创建根空间节点 递归添加子空间节点
        def add_space_nodes_recursively(node_info : dict, parent_id):
            
            # print("Adding node:", node_info,"\n\n\n\n")
            
            name = node_info.get("name", "")
            path = node_info.get("path", "")
            skills = node_info.get("skills", [])
            children = node_info.get("children", [])
            
            # print("Name:", name)
            # print("Path:", path)
            # print("Skills:", skills)
            # print("Children:", children)
            # print("\n\n")
            
            # Create a space node
            # 如果名称中含有 /temp/ 则 node_type 设置为 SHORT_TERM
            node_type = NodeType.FIXED
            if "/temp/" in path:
                node_type = NodeType.SHORT_TERM
            space_node_id = self.memory_graph.add_node(
                node_type=node_type,
                node_class=NodeClass.SPACE,
                name=name,
                summary=f"Space node for {name} at {path}",
                parent_id=parent_id,
                weight=1.0,
                child_cnt=len(children),
                parents_cnt=1 if parent_id != -1 else 0
            )
            # Add as child to parent
            if parent_id != -1:
                self.memory_graph.add_child(parent_id, space_node_id)

            # 添加技能 - 创建情景子节点
            for skill_name in skills:
                # get skill dict form skill_specs
                skill = skill_specs.get(skill_name, {})

                skill_name = skill.get("name", "")
                skill_desc = skill.get("description", "")
                skill_input = skill.get("input", {})
                skill_output = skill.get("output", {})
                skill_dependencies = skill.get("dependencies", [])
                
                if skill_desc != "":
                    skill_summary = skill_desc
                else :
                    skill_summary = f"Skill {skill_name} with input {skill_input} and output {skill_output}, dependencies: {skill_dependencies}"
                
                # Create a context node for the skill
                skill_node_id = self.memory_graph.add_node(
                    node_type=NodeType.FIXED,
                    node_class=NodeClass.CONTEXT,
                    name=skill_name,
                    summary=skill_summary,
                    parent_id=space_node_id,
                    weight=1.0,
                    child_cnt=0,
                    parents_cnt=1
                )
                # Add the skill node as a child of the space node
                self.memory_graph.add_child(space_node_id, skill_node_id)
            
            # TODO 对 SKILL的子节点遍历处理
            
            # Recursively add children is a dict
            # 遍历children
            # 如果 children 是列表 ，遍历列表
            if isinstance(children, list):
                for child_info in children:
                    add_space_nodes_recursively(child_info, space_node_id)
            elif isinstance(children, dict):
                for _, child_info in children.items():
                    add_space_nodes_recursively(child_info, space_node_id)
        self.root_space_node_id = add_space_nodes_recursively(graph_structure, self.root_node_id)
        
        

    def init_Skill(self, skill_info_list: dict):
        # 生成一个 skill_info_id_list 记录 存入当前graph后id的位置，便于确定技能节点的位置
        
        # skill_info_id_list 设置跟 skill_info_list 大小相同
        skill_info_id_list = [0] * len(skill_info_list)
        
        # 遍历 skill_info_list 仿照load_from_file 方法写
        for idx, node_data in enumerate(skill_info_list):
            node_id = node_data["id"]
            node_type = NodeType[node_data["type"]]
            node_class = NodeClass[node_data["class"]]
            name = node_data["name"]
            summary = node_data["summary"]
            parent_id = node_data.get("parent", -1)
            parent_id = skill_info_id_list[parent_id]
            weight = node_data.get("weight", 1.0)
            child_cnt = node_data.get("child_cnt", 0)
            parents_cnt = node_data.get("parents_cnt", 0)
            
            # print(f"Loading node {node_id}: type={node_type}, class={node_class}, name={name}, parent_id={parent_id}, weight={weight}, child_cnt={child_cnt}, parents_cnt={parents_cnt}")
            cur_node_id = -1
            
            if node_class == NodeClass.SPACE:
                x = node_data.get("x", 0.0)
                y = node_data.get("y", 0.0)
                z = node_data.get("z", 0.0)
                # Create the node
                cur_node_id = self.memory_graph.add_node(node_type, node_class, name, summary,
                            parent_id, weight=weight,
                            child_cnt=child_cnt, parents_cnt=parents_cnt,
                            x=x, y=y, z=z)
            elif node_class == NodeClass.TIME:
                timestamp = node_data.get("timestamp", -1.0)
                # Create the node
                cur_node_id = self.memory_graph.add_node(node_type, node_class, name, summary,
                            parent_id, weight=weight,
                            child_cnt=child_cnt, parents_cnt=parents_cnt,
                            timestamp=timestamp)
            else:
                # Create the node without spatial or temporal attributes
                cur_node_id = self.memory_graph.add_node(node_type, node_class, name, summary,
                        parent_id, weight=weight,
                        child_cnt=child_cnt, parents_cnt=parents_cnt)
            skill_info_id_list[idx] = cur_node_id
    
    # update scene
    def update_Scene(self, scene_info: dict):
        with open("scene_info.json", "w") as f:
            json.dump(scene_info, f, indent=4)
        # TODO 增量更新 # 目前将所有scene信息重新初始化 
       
        del self.memory_graph
        # 新建 当前graph
        self.memory_graph = MemoryGraph()
        self.init_Scene(scene_info)
        
        # 保存当前graph
        self.memory_graph.save_to_file("memory/data/current_memory_graph.json")
        self.memory_graph.visualize("memory/data/current_memory_graph.png")
        
        
        
    
    # 场景所有信息更新到memory/data/retrieval_memory.json中
    def update_retrieval(self):
        # 遍历当前graph存储id序号和节点信息 summary
        nodes_list = []
        edges_list = []
        
        for node_id, node in self.memory_graph._node_map.items():
            node_data = {
                "id": node.node_id,
                "event": node.summary if node.summary else node.name,
            }
            nodes_list.append(node_data)
            # edges
            for child_id in self.memory_graph.get_childs(node_id):
                edge_data = {
                    "source": node_id,
                    "target": child_id,
                    "relation": "next"  # Placeholder relation
                }
                edges_list.append(edge_data)
        
        # Save to retrieval_memory.json
        retrieval_data = {
            "nodes": nodes_list,
            "edges": edges_list
        }
        
        with open("memory/data/retrieval_memory.json", 'w') as f:
            json.dump(retrieval_data, f, indent=4, ensure_ascii=False)
        
    # TODO FIX 适配当前graph的检索 - query 分为两种类型 用户请求 和 安全规则
    def retrieval_Request(self, query: str, top_k: int = 5) -> List[dict]:
        
        # 将 场景所有信息更新到memory/data/retrieval_memory.json中
        self.update_retrieval()
    
        # 加载图谱 & 构建事件语料
        skg = ScriptKnowledgeGraph("memory/data/retrieval_memory.json")
        all_events = list(skg.nodes.values())

        # 向量检索相关事件
        retriever = Retriever(all_events)
        retrieved_events = retriever.retrieve(query)

        # 根据最相关事件查找事件链
        event_chain = skg.find_event_chain(retrieved_events[0]) if retrieved_events else []

        # 将 event_chain 整合summary 作为检索结果返回
        # print("Event Chain:", event_chain)
        
        # TODO 待优化 应该返回两个列表，一个名字列表，一个对应的事件链列表
        # 这里先遍历 current_memory_graph.json 根据 summary 找到对应的 name
        event_chain_with_names = []
        for event in event_chain:
            # 遍历 memory_graph 找到对应 summary 的 name
            for node_id, node in self.memory_graph._node_map.items():
                if node.summary == event or node.name == event:
                    event_chain_with_names.append({
                        "name": node.name,
                        "event": event
                    })
                    break
        

        return event_chain_with_names

def read_scene_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        scene_info = json.load(f)
    return scene_info

def test():
    cs = CurrentState()
    
    # 读取json文件
    scene_info = read_scene_json("memory/data/scene_data.json")
    # print(scene_info)
    
    cs.init_Scene(scene_info.get("scene_info", {}))
    
    skills_memory = read_scene_json("memory/data/skill_data.json")
    
    cs.init_Skill(skills_memory)
    

    query = "get a cup of milk"
    
    print("Retrieval Query:\n", query)
    
    results = cs.retrieval_Request(query)
    
    print("Retrieval Results:\n", results)


    
    # 将 graph 保存至json并可视化 
    #cs.memory_graph.save_to_file("memory/data/current_memory_graph.json")
    #cs.memory_graph.visualize("memory/data/current_memory_graph.png")
    


# TODO short -> long term skill 的转变
        
def main():
    mg = MemoryGraph()

    mg.load_from_file("memory_graph.json")

    # check
    print("All nodes in the graph:")
    for node_id, node in mg._node_map.items():
        print(f"Node ID: {node_id}, Name: {node.name}, Summary: {node.summary}, Type: {node.node_type}, Class: {node.node_class}")
    print("\nNode 1 details:", mg.get_node(1).__dict__)
    print("Node 2 details:", mg.get_node(2).__dict__)
    print("Node 3 details:", mg.get_node(3).__dict__)
    print("Graph Node 3 details:", mg.get_graph_node(3))

    print("Node 18 details:", mg.get_node(18).__dict__)

    # visualize
    mg.visualize("memory_graph2.png")

    mg.save_to_file("memory_graph2.json")

if __name__ == "__main__":
    test()
