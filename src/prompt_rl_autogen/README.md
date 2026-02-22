# prompt_rl_autogen

独立的 Microsoft AutoGen 实验目录，用于验证：

- 使用 RL 优化 protocol-generation prompt
- 让生成的协议注入 AutoGen 多智能体协作过程
- 在标准化协作任务上观察成功率与效率变化

该目录与 `src/prompt_rl_guess` 实验逻辑解耦，单独维护以下内容：

- `train.py`: 训练入口
- `autogen_env.py`: AutoGen 回合封装（把一次任务执行变成可打分 episode）
- `reward.py`: 奖励函数
- `eval_checkpoint.py`: checkpoint 评估
- `inference.py`: LoRA 推理
- `prompt_history/`, `protocol_history/`, `checkpoints/`, `eval_outputs/`: 实验产物

## 依赖

建议额外安装：

```bash
pip install autogen-agentchat autogen-ext[openai]
```

## 运行

```bash
python -m src.prompt_rl_autogen.train
python -m src.prompt_rl_autogen.eval_checkpoint
```
