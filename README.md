# CodeWithReinforcementLearning
郭宪 宋俊潇 方勇纯-《深入浅出强化学习:编程实战》  代码

在写代码过程中要比较细心，比如在写YuanYangEnv环境过程中，如果出现了碰撞是不能像下一个状态走的，因为在坐标转换的过程当中会出现，list out of range的错误，因为在撞墙之后他一定会出圈。

所有的代码py文件名称均与书中一致。

#### [Chapter0先导篇](https://github.com/Vaild/CodeWithReinforcementLearning/tree/main/Chapter0%E5%85%88%E5%AF%BC%E7%AF%87)

- k-摇臂赌博机:  [KB_Game.py](https://github.com/Vaild/CodeWithReinforcementLearning/blob/main/Chapter0先导篇/KB_Game_RL/KB_Game.py)

- YuanYang环境:  [YuanYangEnv.py](https://github.com/Vaild/CodeWithReinforcementLearning/blob/main/Chapter0先导篇/MarkovEnv/YuanYangEnv.py)

- 环境的依赖文件用于加载图片:  [load.py](https://github.com/Vaild/CodeWithReinforcementLearning/blob/main/Chapter0先导篇/MarkovEnv/load.py)

[**Chapter1基于值函数的方法**](https://github.com/Vaild/CodeWithReinforcementLearning/tree/main/Chapter1%E5%9F%BA%E4%BA%8E%E5%80%BC%E5%87%BD%E6%95%B0%E7%9A%84%E6%96%B9%E6%B3%95)

- 策略迭代:  [DP_policy_Iter.py](https://github.com/Vaild/CodeWithReinforcementLearning/blob/main/Chapter1基于值函数的方法/基于动态规划的方法/DP_policy_Iter.py)

- 值迭代:  [DP_Value_Iter.py](https://github.com/Vaild/CodeWithReinforcementLearning/blob/main/Chapter1基于值函数的方法/基于动态规划的方法/DP_Value_Iter.py)