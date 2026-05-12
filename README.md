---
title: Fairy AI News Lab
emoji: "🤖"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
short_description: 新闻问答、分类、情感分析与语音交互演示
---

# Fairy AI News Lab

一个适合课程结课展示的 Python 智能新闻助手项目，在原有新闻 RAG Agent 基础上，补上了深度学习和智能语音能力。

## 项目亮点

- 新闻 RAG 检索：基于 `SentenceTransformer` 做相似度召回
- 新闻情感分析：支持 `正面 / 负面 / 中立` 判断
- 新闻分类：使用自定义 `Transformer Encoder` 训练 `体育 / 科技 / 财经 / 娱乐`
- 大模型总结：接入 `DeepSeek` 对检索结果进行归纳总结
- 语音输入：前端浏览器麦克风识别
- Whisper 预留接口：后端支持接入 Whisper ASR
- 语音播报：浏览器 `speechSynthesis` 自动朗读回答
- 舆情分析：统计类别分布、情感分布和关键词，适合答辩展示

## 项目结构

```text
AI Agent/
├── app.py                         # FastAPI 服务入口
├── agent_v5.py                    # 主 Agent 调度
├── deepseek_client.py             # DeepSeek 调用封装
├── news_intelligence.py           # 情感分析 / 分类 / 摘要 / 舆情
├── voice_service.py               # Whisper 语音识别接口
├── train_transformer_news.py      # Transformer 分类训练脚本
├── scheduler.py                   # 自动抓取与训练调度
├── docs/
│   └── VSCODE_FIX.md              # VS Code 环境问题说明
├── data/
│   ├── news.json
│   ├── news_embeddings.npy
│   ├── news_train.jsonl
│   └── news_train_generated.jsonl
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── scripts/
│   └── activate_env.sh            # conda 环境激活脚本
└── requirements.txt
```

## 环境准备

建议使用 VSCode + Python 虚拟环境。

```bash
pip install -r requirements.txt
```

## 环境变量

在项目根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

QWEATHER_API_KEY=你的和风天气密钥
QWEATHER_HOST=你的和风天气域名

# 可选：如果你已经下载了 Whisper 模型，再开启这一项
WHISPER_MODEL_ID=openai/whisper-tiny
WHISPER_LANGUAGE=zh

# 新闻自动更新
NEWS_AUTO_REFRESH_ENABLED=true
NEWS_REFRESH_ON_START=true
NEWS_REFRESH_INTERVAL_SECONDS=1800
NEWS_AUTO_TRAIN_ENABLED=true
NEWS_TRAIN_EPOCHS=4
NEWS_TRAIN_BATCH_SIZE=8
```

## 训练 Transformer 新闻分类模型

项目内置了一份小型中文样本数据集，可直接训练一个演示版分类模型：

```bash
python train_transformer_news.py
```

训练完成后会生成：

- `models/news_transformer/model.pt`
- `models/news_transformer/vocab.json`
- `models/news_transformer/config.json`

## 启动项目

如果你沿用项目原来的 conda 环境，先执行：

```bash
conda activate ai_agent
```

```bash
python run_server.py
```

默认会监听 `0.0.0.0:8000`，终端会打印两个地址：

```text
Local access:   http://127.0.0.1:8000
LAN access:     http://你的局域网IP:8000
```

也可以继续使用 Uvicorn 命令手动启动：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 让别的电脑和手机访问

### 方案一：同一 Wi-Fi 下直接访问

这是最适合课堂展示和局域网演示的方式。

1. 电脑运行：

```bash
python run_server.py
```

2. 让访问者和你的电脑连接同一个路由器或同一个手机热点。
3. 在对方电脑或手机浏览器里打开终端打印出来的 `LAN access` 地址，例如：

```text
http://192.168.1.23:8000
```

如果打不开，通常需要检查两件事：

- 电脑防火墙是否拦截了 `8000` 端口
- 手机和电脑是否真的在同一个局域网

### 方案二：公网部署成真正的网址

如果你想让不在同一个网络的人也能访问，就需要把这个 FastAPI 项目部署到一台有公网 IP 的机器或云平台上。  
`Docker` 的作用是把项目和运行环境一起打包，方便在别的服务器上稳定启动；但 `Docker` 本身不等于“自动变成公网网址”，真正对外访问还需要公网服务器、开放端口，或者绑定域名。

建议部署命令：

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

部署后把平台分配的 `https://...` 地址发给别人，他们在电脑和手机上都可以打开。

## Docker 部署

### 1. 准备环境变量

先复制一份示例文件：

```bash
cp .env.example .env
```

然后把 `.env` 里的 API Key 改成你自己的。

### 2. 本地构建 Docker 镜像

```bash
docker build -t fairy-ai-news-lab .
```

默认会使用：

```text
docker.m.daocloud.io/library/python:3.11-slim
```

作为 Python 基础镜像，这样在当前网络环境下通常比直接访问 Docker Hub 更稳定。  
如果你在海外服务器或网络畅通环境下想切回官方源，也可以这样构建：

```bash
docker build \
  --build-arg PYTHON_BASE_IMAGE=python:3.11-slim \
  -t fairy-ai-news-lab .
```

### 3. 本地启动容器

```bash
docker run --env-file .env -p 8000:8000 fairy-ai-news-lab
```

打开：

```text
http://127.0.0.1:8000
```

### 4. 使用 Docker Compose 启动

项目已经自带 [docker-compose.yml](/Users/yongchengwang/Desktop/projects/AI%20Agent/docker-compose.yml:1)，可以直接运行：

```bash
docker compose up -d --build
```

如果你想在 `docker compose` 中改用官方基础镜像，可以在 `.env` 中设置：

```env
PYTHON_BASE_IMAGE=python:3.11-slim
```

查看日志：

```bash
docker compose logs -f
```

默认会自动开启新闻后台更新：

- 启动时立即抓取一次新闻
- 每 `1800` 秒，也就是每 `30` 分钟自动刷新
- 自动更新 `data/news.json` 和 `data/news_embeddings.npy`
- 自动生成弱监督训练集 `data/news_train_generated.jsonl`
- 默认自动重训一次 Transformer 分类模型，下一次新闻问答会热重载新模型
- `docker compose` 会把宿主机的 `./data` 和 `./models` 挂载到容器内，避免生成文件留在容器里丢失

如果你只想更新新闻，不想自动训练，可以在 `.env` 中设置：

```env
NEWS_AUTO_TRAIN_ENABLED=false
```

停止服务：

```bash
docker compose down
```

## 把 Docker 放到公网服务器

最稳的方式是准备一台 Linux 云服务器，然后把这个项目放上去。

### 基本步骤

1. 购买或准备一台带公网 IP 的服务器
2. 安装 Docker 和 Docker Compose
3. 把项目上传到服务器
4. 在服务器项目目录配置 `.env`
5. 运行：

```bash
docker compose up -d --build
```

6. 打开服务器防火墙或安全组端口，例如 `8000`
7. 其他人访问：

```text
http://你的服务器公网IP:8000
```

### 如果想直接用 80 端口

把 `.env` 中的：

```env
PUBLIC_PORT=8000
```

改成：

```env
PUBLIC_PORT=80
```

然后重新启动：

```bash
docker compose up -d --build
```

这样别人就可以直接访问：

```text
http://你的服务器公网IP
```

### 如果想绑定域名和 HTTPS

这时通常会再加一层反向代理，例如 Nginx 或 Caddy，让：

- `https://your-domain.com` 对外提供 HTTPS
- 反向代理把请求转发到容器内的 `8000` 端口

这一步是“正式上线”常见做法，特别适合手机访问，因为很多移动端权限和语音能力在 HTTPS 下更稳定。

## 免费固定网址方案

如果你不想买服务器，但又希望网址固定、不会像临时 tunnel 一样总变化，推荐使用 **Hugging Face Spaces（Docker Space）**。

它的优点是：

- 免费可用
- 有固定网址
- 支持 Docker
- 很适合课程展示和作品集链接

Space 创建后，你的网址会固定成类似：

```text
https://你的用户名-你的space名.hf.space
```

### Hugging Face Spaces 部署步骤

1. 注册一个 Hugging Face 账号
2. 新建一个 `Space`
3. 选择 `Docker` SDK
4. 把这个项目代码推送到那个 Space 仓库
5. 在 Space 的 `Settings -> Variables and secrets` 里配置：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
QWEATHER_API_KEY=你的和风天气密钥
QWEATHER_HOST=你的和风天气域名
```

6. 等待构建完成
7. 访问固定网址

### 为什么这个方案比临时 tunnel 稳定

因为这次不再是“你电脑开着，然后借别人一个临时公网入口”，而是“平台帮你真正托管这个容器”。  
所以：

- 网址固定
- 不依赖你电脑一直开机
- 不依赖一条 SSH 隧道一直保持在线

## 额外配置

可以通过环境变量控制启动行为：

```env
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=true
PUBLIC_PORT=8000

# 如果前后端分开部署，可额外放开指定来源
CORS_ALLOW_ORIGINS=https://你的前端域名
```

## 功能演示建议

可以直接输入或语音提问：

- `最近AI行业有什么新闻？`
- `帮我分析一下最近财经新闻的情感倾向`
- `总结一下今天的科技热点`
- `北京今天天气怎么样`

## 说明

- 如果没有配置 `DEEPSEEK_API_KEY`，系统仍然可以进行本地新闻检索、分类、情感分析和舆情总结。
- 如果浏览器不支持原生语音识别，文本输入功能仍可正常使用。
- 如果没有配置 Whisper 模型，前端默认使用浏览器语音识别做课堂演示。
- 手机端已经适配基础聊天操作，但不同手机浏览器对麦克风和语音识别的支持不完全一致。
- iPhone 上如果遇到语音不可用，优先改用文本输入，或把项目部署到 HTTPS 域名后再测试语音能力。
- `.env` 不应该提交到 Git，也不应该打包进 Docker 镜像；项目现在已经通过 `.dockerignore` 排除了它。
