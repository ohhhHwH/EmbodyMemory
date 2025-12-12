# EmbodyMemory



我的世界的种子


```
<DefaultWorldGenerator forceReset="true" seed="115102116540990794" />
<Placement x="-176.5" y="63.0" z="250.5"/>

<Placement x="-203.5" y="81.0" z="217.5"/>
```



# 测试任务设计

## Rule
- 所有方法使用：
    - 相同的 **高阶动作空间**（8 类抽象动作）
    - 相同的 **控制器实现**
    - 相同的 **提示模板（prompt template）**
    - 相同的 **环境接口**
- 每次运行使用 **不同种子（seed）** 和 **随机出生点** - LLM
- 学习场景 - 手动学习一个世界的场景 - 形成记忆
每个任务 **运行 30 次**，不同任务运行时起始位置和随机种子均不同，每次任务运行LLM、LLM+MEM与LLM+MEM+MODEs。
### 评价指标

| Evaluation Metrics | Description            |     |
| ------------------ | ---------------------- | --- |
| 成功率                | 最终是否达到目标               |     |
| 运行步数               | 成功的情况下需要执行的步数是多少       |     |
| 记忆相关性              | 查找到的记忆与问题的相关性          | MEM |
| 记忆查询时间             | 测试中查询记忆的占比             | MEM |
| token长度            | 最终执行完成的情况下使用token数     |     |
| 无用action占比         | 除运动外action 有没有输出/与手动对照 |     |

## Task

### Task List
basic,easy,Medium,hard,complex

| 原始任务                 | 寻找类子任务 | 采集类子任务  | 制作类子任务 |
| -------------------- | ------ | ------- | ------ |
| mine log             | 寻找 木头  | 采集 木头   |        |
| mine sand            | 寻找 沙子  | 采集 沙子   |        |
| mine sapling         | 寻找 树苗  | 采集 树苗   |        |
| mine wheat seeds     | 寻找 小麦草 | 采集 小麦种子 |        |
| mine dirt            | 寻找 泥土  | 采集 泥土   |        |
| mine grass           | 寻找 草方块 | 采集 草    |        |
| craft plank          |        | 采集 木头   | 制作 木板  |
| craft stick          |        | 采集 木板   | 制作 木棍  |
| craft button         |        | 采集 木板   | 制作 木按钮 |
| craft crafting table |        | 采集 木板   | 制作 工作台 |

| 原始任务                        | 寻找类子任务 | 采集类子任务   | 制作类子任务  |
| --------------------------- | ------ | -------- | ------- |
| craft chest                 |        | 采集 木板    | 制作 箱子   |
| craft bowl                  |        | 采集 木板    | 制作 木碗   |
| craft boat                  |        | 采集 木板    | 制作 木船   |
| craft wooden slab           |        | 采集 木板    | 制作 木台阶  |
| craft wooden pressure plate |        | 采集 木板    | 制作 木压力板 |
| craft ladder                |        | 采集 木棍    | 制作 梯子   |
| craft barrel                |        | 采集 木板    | 制作 木桶   |
| craft wooden axe            |        | 采集 木板、木棍 | 制作 木斧   |
| craft wooden pickaxe        |        | 采集 木板、木棍 | 制作 木镐   |
| craft wooden sword          |        | 采集 木板、木棍 | 制作 木剑   |

| 原始任务                | 寻找类子任务 | 采集类子任务   | 制作类子任务      |
| ------------------- | ------ | -------- | ----------- |
| mine cobblestone    | 寻找 石头  | 采集 圆石    |             |
| mine coal ore       | 寻找 煤矿  | 采集 煤炭    |             |
| craft furnace       |        | 采集 圆石    | 制作 熔炉       |
| craft lever         |        | 采集 圆石、木棍 | 制作 拉杆       |
| craft stone pickaxe |        | 采集 圆石、木棍 | 制作 石镐       |
| craft stone axe     |        | 采集 圆石、木棍 | 制作 石斧       |
| craft stone hoe     |        | 采集 圆石、木棍 | 制作 石锄       |
| craft stone shovel  |        | 采集 圆石、木棍 | 制作 石铲       |
| craft stone sword   |        | 采集 圆石、木棍 | 制作 石剑       |
| mine iron ore       | 寻找 铁矿  | 采集 铁矿石   |             |
| smelt glass         | 寻找 沙子  | 采集 沙子    | 制作 玻璃（熔炉烧制） |

| 原始任务                  | 寻找类子任务 | 采集类子任务        | 制作类子任务    |
| --------------------- | ------ | ------------- | --------- |
| smelt iron ingot      | 寻找 铁矿  | 采集 铁矿石、煤炭     | 制作 铁锭（熔炉） |
| craft iron bars       |        | 采集 铁锭         | 制作 铁栏杆    |
| craft carpentry table |        | 采集 木材 / 铁（若需） | 制作 木工台    |
| craft iron pickaxe    |        | 采集 铁锭、木棍      | 制作 铁镐     |
| craft iron door       |        | 采集 铁锭         | 制作 铁门     |
| craft iron trapdoor   |        | 采集 铁锭         | 制作 铁活板门   |
| craft rail            |        | 采集 铁锭、木棍      | 制作 铁轨     |
| craft cauldron        |        | 采集 铁锭         | 制作 炼药锅    |

| 原始任务                  | 寻找类子任务    | 采集类子任务         | 制作类子任务  |
| --------------------- | --------- | -------------- | ------- |
| obtain diamond        | 寻找 钻石矿    | 采集 钻石（通常需铁镐）   |         |
| mine redstone         | 寻找 红石矿    | 采集 红石粉         |         |
| craft dropper         |           | 采集 圆石、红石       | 制作 投掷器  |
| craft redstone torch  |           | 采集 红石、木棍       | 制作 红石火把 |
| craft compass         | 寻找 铁矿、红石矿 | 采集 铁锭、红石       | 制作 指南针  |
| craft clock           | 寻找 金矿     | 采集 金锭、红石       | 制作 时钟   |
| craft piston          |           | 采集 圆石、铁锭、木板、红石 | 制作 活塞   |
| craft diamond pickaxe | 寻找 钻石矿    | 采集 钻石、木棍       | 制作 钻石镐  |
| craft diamond sword   | 寻找 钻石矿    | 采集 钻石、木棍       | 制作 钻石剑  |
| craft raw gold block  | 寻找 金矿     | 采集 原金          | 制作 原金块  |

### LLM List
好像deepseek只有语言模型，适合做base line
LLM
```
deepseek/deepseek-v3.2 - baseline

qwen/qwen3-32b
moonshotai/kimi-k2-0905
openai/chatgpt-4o-latest
```
VLM
[模型推理价格说明 - Moonshot AI 开放平台 - Kimi 大模型 API 服务](https://platform.moonshot.cn/docs/pricing/chat#%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B-kimi-latest)
```
qwen/qwen3-vl-32b-instruct
moonshotai/moonshot-v1-32k-vision-preview - 官网提供的API
openai/chatgpt-4o-latest
```
### Task.config

llm.config
```
LLM_MODE = True

MEM_MODE = False

DETECT_MODE = False

SUBMISSION_MODE = False

Task Name = `craft wooden slab`

LLM_MODLE = ""

```
llm+mem.config
```
LLM_MODE = True

MEM_MODE = True

SUBMISSION_MODE = False

DETECT_MODE = False

Task Name = `craft clock`

LLM_MODLE = ""

```

llm+mem+other mode.config
```
LLM_MODE = True

MEM_MODE = True

SUBMISSION_MODE = True

DETECT_MODE = True

Task Name = `craft clock`

LLM_MODLE = ""

VLLM_MODLE = ""

```



## OutPut
