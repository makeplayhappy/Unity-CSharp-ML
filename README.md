# This is a pure C# implementation of some ML algorythms
Note this is a fork of [EpicSpaces : Reinforcement-Learning-c-sharp-Unity-ppo-ddpg-dqn](https://github.com/EpicSpaces/Reinforcement-Learning-c-sharp-Unity-ppo-ddpg-dqn) Which is a rewrite of a few of PyTorch's algorythms in C#.

I've attempted to optimise some of the Unity specific sections. The ball shows balance after around 150 episodes. 
I've changed the rewards to mimic those in the [Unity ML Agents Ball3D example](https://github.com/Unity-Technologies/ml-agents/blob/main/Project/Assets/ML-Agents/Examples/3DBall/Scripts/Ball3DAgent.cs)

While running you'll experience some large performance spikes. These are the matrix operations. This is why the PyTorch implementation and optimisations are the prefered solution.




## Original README kept for clarity
## Reinforcement Learning completely written from PyTorch to C# Unity Asset 
have fun !
![Alt text](Screenshot_1.gif?raw=true "pic")