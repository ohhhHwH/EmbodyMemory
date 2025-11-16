
import malmoenv
import time


# 将指令序列转换成 action



class ActionAgent:

    def __init__(self, env=None, action_filter=None):
        self.env = env
        self.actions = env.actions
        self.action_space = env.action_space
        if action_filter is not None:
            self.actions = [action for action in self.actions if any(af in action for af in action_filter)]
            print(f"Filtered actions: {self.actions}")
        

    def description_action(self):
        '''
        打印当前有什么动作
        '''
        actions = [
            "move 'steps'",     # 前后移动
            "turn 'steps'",     # 左右转动
            "look",     # 上下视角
            "jump 'times'",     # 跳跃
            "use",      # 使用物品
            "attack"    # 攻击
        ]

        
        print("Available actions:")
        for action in actions:
            print(" - " + action)
            
        print("\n")
        
        user_input = input("q")
        if user_input.lower() == 'q':
            return None
        if not user_input.isdigit() or int(user_input) < 0 or int(user_input) >= len(self.env.actions):
            action = self.env.action_space.sample()
        else:
            action = int(user_input)


    # 模拟“离散移动一步”的效果
    def move_steps(self, step_long=5):
        '''
        模拟“离散移动 step_long 步”的效果
        '''
        while step_long > 0:
            obs, reward, done, info = self.env.step("move 1")
            time.sleep(0.2)
            obs, reward, done, info = self.env.step("move 0")
            step_long -= 1

    # 模拟“离散转动一步”的效果
    def turn_steps(self, step_angle=15):
        '''
        模拟“离散转动 step_angle 度”的效果
        '''
        
        step_angle = abs(step_angle)
        while step_angle > 0:
            obs, reward, done, info = self.env.step("turn 1")
            time.sleep(0.2)
            obs, reward, done, info = self.env.step("turn 0")
            step_angle -= 1