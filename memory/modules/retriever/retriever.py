from sentence_transformers import SentenceTransformer, util
import os
import torch


class Retriever:
    def __init__(self, events):
        # 模型路径
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(root_dir, 'all-MiniLM-L6-v2')
        # model_path = "/home/hyl/DeepEmbody/memory/gte-Qwen2-7B-instruct"


        print(f"Loading SentenceTransformer model from {model_path}...")
        # 加载模型
        self.model = SentenceTransformer(model_path)

        # 移动模型到 GPU（如果可用）
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model = self.model.to(self.device)
            # print("✅ SentenceTransformer 已加载到 GPU")
        else:
            self.device = 'cpu'
            print("⚠️ 未检测到 GPU，使用 CPU 模式")

        self.events = events
        self.embeddings = self.model.encode(events, convert_to_tensor=True, device=self.device)

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True, device=self.device)
        scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        
        print("events size:", len(self.events))  # 调试输出事件数量
        print("Sorce embedding size:", scores.size())  # 调试输出查询嵌入尺寸
        print("events:", self.events)  # 调试输出事件列表
        print("Scores:", scores)  # 调试输出相似度分数
        
        # 输出到memory/data/debug_scores.txt文件 以分数为大小进行排序
        debug_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'debug_scores.txt')
        with open(debug_path, 'w') as f:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            f.write(f"Query: {query}\n\n")
            for idx, score in zip(sorted_indices, sorted_scores):
                f.write(f"{score.item():.4f}\t{self.events[idx]}\n")
        
        top_idx = scores.topk(k=top_k).indices
        return [self.events[i] for i in top_idx]
