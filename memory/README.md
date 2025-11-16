# Memory

Stores information across and during tasks. Can include:

- JIT prompts
- World model fragments (object location, room map)
- Short-term memory (task context, intermediate results)
- Long-term skill usage patterns

Memory enables contextual reasoning, personalization, and short-term task coherence.




add memory node in memory.py as follow:

```python
# 根节点：智能体本体
    node1_id = mg.add_node(NodeType.FIXED, NodeClass.CONTEXT, "embody ai assistant", "作为具身智能助手")

    # 核心技能节点（主流程）
    node2_id = mg.add_node(NodeType.LONG_TERM, NodeClass.CONTEXT, "skill:get_milk", "牛奶服务流程", 
                        parent_id=node1_id)

    # 空间场景节点（厨房环境）
    node3_id = mg.add_node(NodeType.LONG_TERM, NodeClass.SPACE, "location:kitchen", "厨房场景（含冰箱、水池、餐桌）", 
                        parent_id=node1_id, x=1.0, y=1.0, z=1.0)  # 厨房中心坐标

    # 冰箱实体（存储牛奶）
    node4_id = mg.add_node(NodeType.LONG_TERM, NodeClass.SPACE, "appliance:fridge", "家用冰箱（含牛奶）", 
                        parent_id=node3_id, x=0.8, y=1.5, z=1.0)  # 冰箱位置

    # 牛奶实体（初始在冰箱内）
    node5_id = mg.add_node(NodeType.LONG_TERM, NodeClass.SPACE, "object:milk", "未开封牛奶盒", 
                        parent_id=node4_id, x=0.9, y=1.6, z=1.2)  # 牛奶在冰箱内的位置

    # 水杯实体（初始在餐桌）
    node6_id = mg.add_node(NodeType.LONG_TERM, NodeClass.SPACE, "object:cup", "清洁水杯（空）", 
                        parent_id=node3_id, x=1.5, y=0.8, z=0.9)  # 水杯在餐桌的位置

    # ---------------------- 主流程时间节点（按事件顺序） ----------------------
    # e1: 走向冰箱
    node7_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:go_to_fridge", "我走向冰箱", 
                        parent_id=node2_id)

    # e2: 打开冰箱门
    node8_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:open_fridge", "我打开冰箱门", 
                        parent_id=node2_id)

    # e3: 检测牛奶是否过期（决策节点）
    node9_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:check_milk_expiry", "我检测牛奶是否过期", 
                        parent_id=node2_id)
    
    # 【情景分支1：牛奶未过期】
    node29_id = mg.add_node(NodeType.LONG_TERM, NodeClass.CONTEXT, "action:milk_expiry", "牛奶过期", 
                        parent_id=node9_id)
    # 【情景分支2：牛奶已过期】
    node30_id = mg.add_node(NodeType.LONG_TERM, NodeClass.CONTEXT, "action:milk_noexpiry", "牛奶没过期", 
                        parent_id=node9_id)

    
    node10_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:take_milk", "我取出牛奶", 
                        parent_id=node29_id)  # 正常流程子节点

    
    node11_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:dispose_milk", "我丢弃过期牛奶", 
                        parent_id=node30_id)  # 替代流程子节点

    # ---------------------- 正常流程（牛奶未过期） ----------------------
    # e4: 取出牛奶后走向水池
    node12_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:go_to_sink", "我走向水池", 
                        parent_id=node29_id)

    # e5: 拿起水杯
    node13_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:pick_cup", "我拿起杯子", 
                        parent_id=node29_id)

    # e6: 清洗杯子
    node14_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:clean_cup", "我清洗杯子", 
                        parent_id=node29_id)

    # e7: 将牛奶倒入杯子
    node15_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:pour_milk", "我将牛奶倒入杯子", 
                        parent_id=node29_id)

    # e8: 加热牛奶（微波炉）
    node16_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:heat_milk", "我加热牛奶（微波炉）", 
                        parent_id=node29_id)

    # e9: 递给用户
    node17_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:deliver_milk", "我将牛奶递给用户", 
                        parent_id=node29_id)

    # ---------------------- 过期处理流程 ----------------------
    # e10: 过期牛奶丢弃至垃圾桶
    node18_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:go_to_trash", "我走向厨房垃圾桶", 
                        parent_id=node30_id)

    node19_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:open_trash", "我打开垃圾桶盖", 
                        parent_id=node30_id)

    node20_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:dispose_expired", "我丢弃牛奶", 
                        parent_id=node30_id)

    node21_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:close_trash", "我关闭垃圾桶盖", 
                        parent_id=node30_id)

    # ---------------------- 环境维护 技能 ----------------------
    # e11: 清理地面（独立于主流程）
    node22_id = mg.add_node(NodeType.LONG_TERM, NodeClass.CONTEXT, "skill:clean", "清洁技能", 
                        parent_id=node1_id)  # 与主流程并行

    node23_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:sweep_floor", "我开启扫地模式", 
                        parent_id=node22_id)

    node24_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:finish_sweep", "我清理扫帚上的灰尘", 
                        parent_id=node22_id)

    # e12: 充电（独立于主流程）
    node25_id = mg.add_node(NodeType.LONG_TERM, NodeClass.CONTEXT, "skill:check_battery", "充电流程", 
                        parent_id=node1_id)  # 与主流程并行

    node26_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:navigate_charger", "我导航至充电桩", 
                        parent_id=node25_id)

    node27_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:dock_charger", "我对接充电桩", 
                        parent_id=node25_id)

    node28_id = mg.add_node(NodeType.LONG_TERM, NodeClass.TIME, "action:start_charge", "我开始充电", 
                        parent_id=node25_id)
```



Memory TODO list:

memory exchange, short-term memory to long-term memory conversion

Integrate with LLM

LLM output structuring and visualization

Structuring and unstructuring

Real-world/Simulation
Implementation
Parallelization
Retrieval

