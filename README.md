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
├── data/
│   ├── news.json
│   ├── news_embeddings.npy
│   └── news_train.jsonl
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
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

```bash
uvicorn app:app --reload
```

浏览器打开：

```text
http://127.0.0.1:8000
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
