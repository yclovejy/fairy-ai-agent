#!/bin/bash
# 激活虚拟环境脚本

echo "=== 激活 AI Agent 虚拟环境 ==="
echo "当前目录: $(pwd)"
echo ""

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: conda 未安装或不在PATH中"
    echo "请先安装conda或将其添加到PATH"
    exit 1
fi

# 激活虚拟环境
echo "激活虚拟环境: ai_agent"
conda activate ai_agent

# 验证激活
if [[ $CONDA_DEFAULT_ENV == "ai_agent" ]]; then
    echo "✓ 虚拟环境激活成功: $CONDA_DEFAULT_ENV"
    echo "Python路径: $(which python)"
    echo "Python版本: $(python --version)"
    echo ""
    echo "现在可以运行: python run_server.py"
else
    echo "✗ 虚拟环境激活失败"
    echo "当前环境: $CONDA_DEFAULT_ENV"
    echo "请手动运行: conda activate ai_agent"
fi

# 保持终端打开
echo ""
echo "按 Ctrl+C 退出"
