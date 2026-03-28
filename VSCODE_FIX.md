# VS Code 导入问题解决方案

## 问题描述
虚拟环境已安装所有依赖库，但VS Code仍显示导入错误。

## 已验证
1. ✅ 所有依赖库已正确安装在 `ai_agent` 虚拟环境中
2. ✅ Python解释器路径: `/opt/anaconda3/envs/ai_agent/bin/python`
3. ✅ 脚本可以正常运行
4. ✅ 所有导入测试通过

## 解决方案

### 方法1：重新加载VS Code窗口
1. 按 `Cmd+Shift+P` (Mac) 或 `Ctrl+Shift+P` (Windows/Linux)
2. 输入 `Developer: Reload Window`
3. 按回车

### 方法2：选择正确的Python解释器
1. 按 `Cmd+Shift+P` (Mac) 或 `Ctrl+Shift+P` (Windows/Linux)
2. 输入 `Python: Select Interpreter`
3. 选择: `/opt/anaconda3/envs/ai_agent/bin/python`
4. 等待VS Code重新加载Python环境

### 方法3：使用工作区文件
1. 在VS Code中打开 `ai_agent.code-workspace` 文件
2. 选择 "Open Workspace"
3. VS Code会自动应用正确的配置

### 方法4：手动配置
如果以上方法无效，手动检查：
1. 检查VS Code底部状态栏的Python版本
2. 确保显示的是 `ai_agent` 环境
3. 如果没有，点击状态栏的Python版本进行切换

### 方法5：清除缓存
1. 关闭VS Code
2. 删除以下文件夹：
   - `~/.vscode/extensions/ms-python.python-*`
   - `~/.vscode/extensions/ms-python.vscode-pylance-*`
3. 重新打开VS Code并重新安装Python扩展

## 验证步骤
运行测试脚本确认一切正常：
```bash
cd "/Users/yongchengwang/Desktop/projects/AI Agent"
python test_import.py
```

## 常见问题
1. **VS Code缓存问题**：VS Code的Python扩展有时会缓存旧的解释器信息
2. **环境变量问题**：确保VS Code继承了正确的环境变量
3. **扩展冲突**：某些扩展可能干扰Python环境检测

## 快速修复命令
在VS Code终端中运行：
```bash
# 激活虚拟环境
conda activate ai_agent

# 验证Python路径
python -c "import sys; print(sys.executable)"

# 重新启动Python语言服务器
# 按 Cmd+Shift+P -> "Python: Restart Language Server"
```

如果问题仍然存在，建议：
1. 重启VS Code
2. 重启电脑
3. 重新创建虚拟环境（最后手段）