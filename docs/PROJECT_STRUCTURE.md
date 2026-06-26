# Fairy Project Structure

这个项目按“入口、核心服务、设备端、训练、前端、数据、测试”分层。

```text
AI Agent/
├── app.py                 # FastAPI 入口，保持在根目录，方便 Hugging Face / Docker 启动
├── run_server.py          # 本地与局域网启动脚本
├── fairy_core/            # Fairy 后端核心能力
├── frontend/              # 网页端 UI
├── devices/               # 边缘设备端程序
├── training/              # 模型训练与视觉训练配置
├── data/                  # 本地数据、知识库、SQLite 历史记录
├── models/                # 本地模型权重
├── scripts/               # 辅助脚本
├── tests/                 # 单元测试与集成测试
└── docs/                  # 项目说明文档
```

## 放置规则

- 新增 Fairy 后端智能体或服务：放到 `fairy_core/`。
- 新增网页界面：改 `frontend/`。
- 新增 K230、ESP32 等硬件脚本：放到 `devices/<device_name>/`。
- 新增训练脚本、数据集配置、视觉模型训练说明：放到 `training/`。
- 新增运行数据、知识库、SQLite 历史库：放到 `data/`，不要放进 `fairy_core/`。
- 新增测试：放到 `tests/`。
- 根目录只保留启动、部署、依赖、总说明文件。

## 当前核心模块

```text
fairy_core/
├── agent_v5.py              # 主 Agent 路由
├── deepseek_client.py       # 大模型调用封装
├── news_intelligence.py     # 新闻分类、摘要、情感与舆情分析
├── news_retrieval.py        # 新闻向量库与检索
├── travel_agent.py          # NLP 地名旅游问答 Agent
├── vision_service.py        # K230 视觉导览后端服务
├── environment_service.py   # ESP32 环境监测后端服务
├── voice_service.py         # 语音识别服务
├── paths.py                 # 统一项目路径
└── travel_qa/               # 地名解析与本地旅游知识库逻辑
```
