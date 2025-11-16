import json
from collections import defaultdict
import os

class ScriptKnowledgeGraph:
    def __init__(self, path=None):
        if path is None:
            # 自动定位项目根目录下的 data/skg.json
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            path = os.path.join(root_dir, 'data', 'skg.json')

        if not os.path.exists(path):
            raise FileNotFoundError(f"cant find skg at path : {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        self.nodes = {node['id']: node['event'] for node in data['nodes']}
        self.graph = defaultdict(list)
        for edge in data['edges']:
            self.graph[edge['source']].append((edge['target'], edge['relation']))


    def find_event_chain(self, start_event):
        # 查找从某事件出发的完整因果链（简单 BFS）
        chain = []
        for id, evt in self.nodes.items():
            if start_event in evt:
                current = id
                while current in self.graph:
                    chain.append(self.nodes[current])
                    current = self.graph[current][0][0]
                chain.append(self.nodes.get(current, ''))
                break
        return chain
