# MARL Prompt RL

用于多智能体协作协议生成的强化学习实验项目，包含两条主要实验链路：
- `prompt_rl_guess`：猜数字协作环境
- `prompt_rl_gsm8k`：GSM8K 协作解题环境

## 快速开始

1) 创建环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) 创建本地配置文件（自行填写 API）

```bash
cp src/provider/config.example.py src/provider/config.py
```

然后编辑 `src/provider/config.py`：
- `API_KEY`: 你的接口密钥
- `API_URL`: OpenAI 兼容接口地址（例如 `http://localhost:8000/v1` 或你的服务地址）

> `src/provider/config.py` 已在 `.gitignore` 中，不会被提交。

## 运行示例

猜数字链路：

```bash
python -m src.prompt_rl_guess.train
python -m src.prompt_rl_guess.eval_checkpoint
```

GSM8K 链路：

```bash
python -m src.prompt_rl_gsm8k.trl_train
python -m src.prompt_rl_gsm8k.compare_eval
```

## 目录

```text
src/
  generator/
  provider/
  prompt_rl_guess/
  prompt_rl_gsm8k/
paper/
```