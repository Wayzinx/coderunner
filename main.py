import aiohttp
import json
import os
import asyncio
import shutil
import subprocess
import signal
import traceback
import time
from urllib.parse import urlparse
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import Plain, Image
from astrbot.api import logger
import uuid

# 配置常量
TIMEOUT_SECONDS = 10  # 代码执行超时时间
MAX_OUTPUT_SIZE = 10000  # 输出最大字符数
SUPPORTED_LANGUAGES = ["python", "javascript", "bash"]
MAX_DEBUG_ATTEMPTS = 3  # 最大调试尝试次数

# 代码保存目录 - 修改为插件目录下的子目录
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_STORAGE_DIR = os.path.join(PLUGIN_DIR, "generated_code")
os.makedirs(CODE_STORAGE_DIR, exist_ok=True)

# 创建shared目录和output目录
SHARED_DIR = os.path.join(PLUGIN_DIR, "shared")
os.makedirs(SHARED_DIR, exist_ok=True)

# 代码生成提示词
CODE_GENERATION_PROMPT = """
你是一个专业的代码生成工具。根据以下描述，生成可执行的代码：

任务描述: {task_description}
编程语言: {language}

你必须生成完整的、正确的、可以直接执行的代码。确保包含所有必要的函数定义、导入语句和主代码部分。
请确保生成的代码是连贯的、没有缩进错误的，并且能够直接运行执行。

可用的输出API:
1. send_text(text): 发送文本消息
2. send_image(image_path): 发送图像文件，图像文件必须先保存到output目录
3. send_file(file_path): 发送任意类型文件，文件必须先保存到output目录

Python示例:
```python
from shared.api import send_text, send_image, send_file
import matplotlib.pyplot as plt
import numpy as np

# 生成正弦波图像
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('正弦波函数')
plt.savefig('output/sin_wave.png')

# 发送文本和图像
send_text('这是一个正弦波函数图像')
send_image('output/sin_wave.png')
```

JavaScript示例:
```javascript
const fs = require('fs');
const { send_text, send_image, send_file } = require('./shared/api');

// 创建一个JSON文件
const data = {
  name: 'Example',
  value: 42
};
fs.writeFileSync('output/data.json', JSON.stringify(data, null, 2));

// 发送文本和文件
send_text('已创建JSON文件');
send_file('output/data.json');
```

重要提示：不要包含任何解释或注释。只返回可执行的代码。格式如下：

```
代码写在这里，确保格式正确，不要有缩进错误
必须包含完整的代码，包括函数定义和必要的调用
```
"""

# 代码调试提示词
CODE_DEBUG_PROMPT = """
你是一个专业的代码调试工具。请分析以下代码执行时遇到的错误，并提供修复后的完整代码。

原始代码:
```{language}
{code}
```

执行错误:
```
{error}
```

请提供修复后的完整代码，不要包含任何解释或注释。确保代码可以直接运行并解决上述错误。
"""

# 运行程序的配置
LANGUAGE_CONFIGS = {
    "python": {
        "file_ext": ".py",
        "command": ["python3", "{filename}"],
        "timeout": TIMEOUT_SECONDS
    },
    "javascript": {
        "file_ext": ".js",
        "command": ["node", "{filename}"],
        "timeout": TIMEOUT_SECONDS
    },
    "bash": {
        "file_ext": ".sh",
        "command": ["bash", "{filename}"],
        "timeout": TIMEOUT_SECONDS
    }
}


async def generate_code(task_description, language, session_id, context):
    """使用LLM生成代码"""
    try:
        # 使用提示词生成代码
        prompt = CODE_GENERATION_PROMPT.format(
            task_description=task_description,
            language=language
        )
        
        llm_response = await context.get_using_provider().text_chat(
            prompt=prompt,
            session_id=f"{session_id}_code_generation"
        )
        
        # 提取代码内容
        code = llm_response.completion_text.strip()
        
        # 如果代码被Markdown代码块包围，提取内部内容
        if "```" in code:
            # 找到第一个和最后一个代码块标记
            start_index = code.find("```")
            end_index = code.rfind("```")
            
            if start_index != end_index:  # 确保找到了开始和结束标记
                # 提取第一个```之后的内容
                start_content = code[start_index + 3:]
                # 跳过可能的语言标识行
                if "\n" in start_content:
                    first_line_end = start_content.find("\n")
                    if first_line_end > 0:  # 如果第一行有内容(可能是语言标识)
                        lang_identifier = start_content[:first_line_end].strip()
                        if lang_identifier and not lang_identifier.startswith("import") and not lang_identifier.startswith("def") and not lang_identifier.startswith("#"):
                            # 这可能是语言标识，跳过它
                            start_content = start_content[first_line_end + 1:]
                        
                # 移除最后的```
                end_content_index = start_content.rfind("```")
                if end_content_index > 0:
                    code = start_content[:end_content_index].strip()
                else:
                    code = start_content.strip()
            
        # 如果是Python代码，检查是否缺少函数定义
        if language == "python" and ("print(" in code or "return " in code) and "def " not in code:
            # 可能缺少函数定义，添加一个封装
            if "fibonacci" in task_description.lower() or "斐波那契" in task_description:
                code = "def fibonacci(n):\n    " + code.replace("\n", "\n    ") + "\n\n# 执行函数\nprint(fibonacci(10))"
            else:
                code = "def main():\n    " + code.replace("\n", "\n    ") + "\n\n# 执行函数\nmain()"
                
        # 对JavaScript代码做类似处理
        elif language == "javascript" and ("console.log" in code or "return " in code) and "function " not in code:
            if "fibonacci" in task_description.lower() or "斐波那契" in task_description:
                code = "function fibonacci(n) {\n    " + code.replace("\n", "\n    ") + "\n}\n\n// 执行函数\nconsole.log(fibonacci(10));"
            else:
                code = "function main() {\n    " + code.replace("\n", "\n    ") + "\n}\n\n// 执行函数\nmain();"
                
        return code
    except Exception as e:
        logger.error(f"代码生成错误: {e}")
        return None


async def debug_code(code, error, language, session_id, context):
    """使用LLM调试代码"""
    try:
        # 使用提示词调试代码
        prompt = CODE_DEBUG_PROMPT.format(
            code=code,
            error=error,
            language=language
        )
        
        llm_response = await context.get_using_provider().text_chat(
            prompt=prompt,
            session_id=f"{session_id}_code_debug"
        )
        
        # 提取代码内容
        debug_code = llm_response.completion_text.strip()
        
        # 如果代码被Markdown代码块包围，提取内部内容
        if debug_code.startswith("```") and debug_code.endswith("```"):
            lines = debug_code.split("\n")
            if len(lines) > 2:
                # 移除第一行的```和最后一行的```
                debug_code = "\n".join(lines[1:-1])
                # 如果第一行包含语言标识，移除它
                if lines[0].startswith("```") and len(lines[0]) > 3:
                    debug_code = "\n".join(lines[2:-1])
        
        return debug_code
    except Exception as e:
        logger.error(f"代码调试错误: {e}")
        return None


async def cleanup_old_projects(max_projects=5, max_age_days=7):
    """清理旧的项目文件，保留最近的项目"""
    try:
        if not os.path.exists(CODE_STORAGE_DIR):
            return
            
        # 列出所有项目目录及其修改时间
        projects = []
        for item in os.listdir(CODE_STORAGE_DIR):
            item_path = os.path.join(CODE_STORAGE_DIR, item)
            if os.path.isdir(item_path):
                mtime = os.path.getmtime(item_path)
                projects.append((item_path, mtime))
        
        # 如果项目数量超过限制，按修改时间排序并删除最旧的
        if len(projects) > max_projects:
            projects.sort(key=lambda x: x[1])  # 按修改时间排序
            to_remove = projects[:-max_projects]  # 保留最新的max_projects个
            for path, _ in to_remove:
                try:
                    shutil.rmtree(path)
                    logger.info(f"清理旧项目目录: {path}")
                except Exception as e:
                    logger.error(f"清理目录失败 {path}: {e}")
        
        # 删除超过指定天数的项目
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        for path, mtime in projects:
            if current_time - mtime > max_age_seconds:
                try:
                    shutil.rmtree(path)
                    logger.info(f"清理过期项目: {path}")
                except Exception as e:
                    logger.error(f"清理目录失败 {path}: {e}")
                    
        # 输出当前项目数量
        remaining = sum(1 for _ in os.listdir(CODE_STORAGE_DIR) if os.path.isdir(os.path.join(CODE_STORAGE_DIR, _)))
        logger.info(f"当前保存的代码项目数: {remaining}")
    except Exception as e:
        logger.error(f"清理过程出错: {e}")


async def run_code(code, language, project_id):
    """执行生成的代码并返回结果"""
    if language not in LANGUAGE_CONFIGS:
        return False, f"不支持的编程语言: {language}. 支持的语言: {', '.join(SUPPORTED_LANGUAGES)}", None
    
    config = LANGUAGE_CONFIGS[language]
    project_dir = None
    
    try:
        # 创建项目目录
        project_dir = os.path.join(CODE_STORAGE_DIR, project_id)
        os.makedirs(project_dir, exist_ok=True)
        
        # 创建output目录
        output_dir = os.path.join(project_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建shared目录软链接或复制文件
        project_shared_dir = os.path.join(project_dir, "shared")
        os.makedirs(project_shared_dir, exist_ok=True)
        
        # 复制api.py到项目shared目录
        shutil.copy(os.path.join(SHARED_DIR, "api.py"), os.path.join(project_shared_dir, "api.py"))
        
        # 创建代码文件
        file_name = f"main{config['file_ext']}"
        file_path = os.path.join(project_dir, file_name)
        
        # 写入代码到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 生成唯一的魔术码，用于标识输出
        magic_code = uuid.uuid4().hex[:8]
        
        # 准备命令
        cmd = [c.format(filename=file_path) for c in config["command"]]
        
        # 创建子进程执行代码
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_dir,
            # 限制资源使用
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "MAGIC_CODE": magic_code
            }
        )
        
        # 设置超时
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=config["timeout"]
            )
            
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            # 限制输出大小
            if len(stdout_str) > MAX_OUTPUT_SIZE:
                stdout_str = stdout_str[:MAX_OUTPUT_SIZE] + "\n... (输出过长已截断)"
            
            if len(stderr_str) > MAX_OUTPUT_SIZE:
                stderr_str = stderr_str[:MAX_OUTPUT_SIZE] + "\n... (错误输出过长已截断)"
            
            # 检查是否有错误
            if stderr_str:
                return False, f"错误:\n{stderr_str}", stderr_str
            
            # 解析输出，检查特殊输出标记
            result_components = []
            has_special_output = False
            text_pattern = f"\\[CODERUNNER_TEXT_OUTPUT#{magic_code}\\]: (.*)"
            image_pattern = f"\\[CODERUNNER_IMAGE_OUTPUT#{magic_code}\\]: (.*)"
            file_pattern = f"\\[CODERUNNER_FILE_OUTPUT#{magic_code}\\]: (.*)"
            
            import re
            
            # 处理文本输出
            for line in stdout_str.split('\n'):
                text_match = re.match(text_pattern, line)
                image_match = re.match(image_pattern, line)
                file_match = re.match(file_pattern, line)
                
                if text_match:
                    has_special_output = True
                    result_components.append(("text", text_match.group(1)))
                elif image_match:
                    has_special_output = True
                    image_path = image_match.group(1)
                    # 确保图像路径存在
                    full_path = os.path.join(project_dir, image_path)
                    if os.path.exists(full_path):
                        result_components.append(("image", full_path))
                    else:
                        result_components.append(("text", f"图像文件不存在: {image_path}"))
                elif file_match:
                    has_special_output = True
                    file_path = file_match.group(1)
                    # 确保文件路径存在
                    full_path = os.path.join(project_dir, file_path)
                    if os.path.exists(full_path):
                        result_components.append(("file", full_path))
                    else:
                        result_components.append(("text", f"文件不存在: {file_path}"))
            
            # 如果没有特殊输出，则将整个stdout作为普通文本输出
            if not has_special_output:
                if stdout_str:
                    result = f"输出:\n{stdout_str}"
                else:
                    result = "代码执行完成，无输出"
                return True, result, None
            else:
                # 将特殊输出组件转换为结果字符串
                result = "代码执行结果:\n"
                for output_type, content in result_components:
                    if output_type == "text":
                        result += f"{content}\n"
                    elif output_type == "image":
                        result += f"[图像: {os.path.basename(content)}]\n"
                    elif output_type == "file":
                        result += f"[文件: {os.path.basename(content)}]\n"
                
                # 返回成功结果，同时包含特殊输出组件
                return True, result, None, result_components
            
        except asyncio.TimeoutError:
            # 超时处理
            if process.returncode is None:
                try:
                    # 尝试终止进程
                    process.send_signal(signal.SIGTERM)
                    await asyncio.sleep(0.5)
                    if process.returncode is None:
                        process.kill()
                except:
                    pass
            
            return False, f"执行超时（超过{config['timeout']}秒）", "执行超时"
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"执行错误: {error_traceback}")
        return False, f"执行错误: {str(e)}", str(e)
    finally:
        # 不立即删除项目目录，因为可能需要使用其中的图像或文件
        # 而是在处理完后通过异步任务清理
        pass


@register("code_runner", "wayzinx", "代码执行工具 - 支持自然语言描述生成并执行代码，支持多种输出（文本/图像/文件）", "1.2.0")
class CodeRunnerPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        # 启动时清理一次，删除所有项目
        asyncio.create_task(cleanup_old_projects(max_projects=0, max_age_days=0))

    @filter.llm_tool(name="run_code")
    async def llm_run_code(self, event, task_description: str, language: str = "python"):
        '''根据自然语言描述生成并执行代码
        
        Args:
            task_description(string): 任务描述，用自然语言说明要执行什么操作
            language(string): 编程语言，可选值：python, javascript, bash，默认为python
        '''
        # 检查语言是否支持
        language = language.lower()
        if language not in SUPPORTED_LANGUAGES:
            return f"不支持的编程语言: {language}。支持的语言: {', '.join(SUPPORTED_LANGUAGES)}"
        
        # 生成项目ID
        if isinstance(event, Context):
            session_id = event._queue[0].session_id if hasattr(event, '_queue') and len(event._queue) > 0 else "ai_session"
        else:
            session_id = event.session_id
            
        project_id = f"{language}_{session_id}_{hash(task_description) % 10000:04d}"
        
        # 生成代码
        code = await generate_code(task_description, language, session_id, self.context)
        
        if not code:
            return "代码生成失败，请尝试更清晰的描述"
        
        # 执行代码并自动调试
        result = await self.execute_with_debug(code, language, session_id, project_id)
        
        # 解析执行结果
        if len(result) == 3:  # 旧格式：(success, final_code, result)
            success, final_code, result_text = result
            # 构建结果消息
            status = "执行成功" if success else "执行失败"
            result_message = f"我已生成{language}代码并执行完毕:\n\n```{language}\n{final_code}\n```\n\n执行结果({status}):\n{result_text}"
            return result_message
        else:  # 新格式：(success, final_code, result_text, components)
            success, final_code, result_text, components = result
            
            # 检查是消息事件还是Context
            if hasattr(event, 'chain_result'):
                # 这是一个消息事件，可以返回Chain
                from astrbot.api.message_components import Plain, Image, File
                
                # 第一部分：代码和状态
                status = "执行成功" if success else "执行失败"
                code_block = f"我已生成{language}代码并执行完毕:\n\n```{language}\n{final_code}\n```\n\n"
                
                # 创建消息链
                message_chain = [Plain(text=code_block)]
                
                # 添加各种输出组件
                for output_type, content in components:
                    if output_type == "text":
                        message_chain.append(Plain(text=content))
                    elif output_type == "image":
                        message_chain.append(Image(file=content))
                    elif output_type == "file":
                        # 简单起见，将文件转换为文本链接描述
                        message_chain.append(Plain(text=f"生成的文件: {os.path.basename(content)}"))
                
                # 为组件之间添加换行
                final_chain = []
                for i, component in enumerate(message_chain):
                    final_chain.append(component)
                    if i < len(message_chain) - 1 and isinstance(component, Plain) and isinstance(message_chain[i+1], Plain):
                        # 确保文本组件之间有换行
                        if not component.text.endswith('\n'):
                            final_chain.append(Plain(text='\n'))
                
                # 返回消息链
                return event.chain_result(final_chain)
            else:
                # 这是一个Context或其他类型，只能返回文本
                status = "执行成功" if success else "执行失败"
                text_parts = [f"我已生成{language}代码并执行完毕:\n\n```{language}\n{final_code}\n```\n\n执行结果({status}):"]
                
                # 添加文本组件
                for output_type, content in components:
                    if output_type == "text":
                        text_parts.append(content)
                    elif output_type == "image":
                        text_parts.append(f"[生成了图像: {os.path.basename(content)}]")
                    elif output_type == "file":
                        text_parts.append(f"[生成了文件: {os.path.basename(content)}]")
                
                # 组合所有文本
                return "\n".join(text_parts)

    @filter.llm_tool(name="run_python_code")
    async def llm_run_python(self, event, task_description: str):
        '''根据自然语言描述生成并执行Python代码
        
        Args:
            task_description(string): 任务描述，用自然语言说明要执行什么Python操作
        '''
        return await self.llm_run_code(event, task_description, "python")

    @filter.llm_tool(name="run_javascript_code")
    async def llm_run_javascript(self, event, task_description: str):
        '''根据自然语言描述生成并执行JavaScript代码
        
        Args:
            task_description(string): 任务描述，用自然语言说明要执行什么JavaScript操作
        '''
        return await self.llm_run_code(event, task_description, "javascript")

    @filter.llm_tool(name="run_bash_code")
    async def llm_run_bash(self, event, task_description: str):
        '''根据自然语言描述生成并执行Bash脚本
        
        Args:
            task_description(string): 任务描述，用自然语言说明要执行什么Bash操作
        '''
        return await self.llm_run_code(event, task_description, "bash")

    async def execute_with_debug(self, code, language, session_id, project_id):
        """执行代码并在失败时进行调试"""
        debug_attempts = 0
        current_code = code
        
        while debug_attempts < MAX_DEBUG_ATTEMPTS:
            # 执行代码
            result = await run_code(current_code, language, project_id)
            
            # 检查返回值数量来确定是哪种格式
            if len(result) == 3:  # 旧格式
                success, output, error = result
            else:  # 新格式
                success, output, error, components = result
            
            if success:
                if len(result) == 3:
                    return success, current_code, output
                else:
                    return success, current_code, output, components
            
            # 如果执行失败且有错误信息，尝试调试
            if error and debug_attempts < MAX_DEBUG_ATTEMPTS - 1:
                debug_attempts += 1
                debug_message = f"代码执行失败，正在进行第{debug_attempts}次调试..."
                
                # 调试代码
                debugged_code = await debug_code(current_code, error, language, session_id, self.context)
                
                if debugged_code:
                    current_code = debugged_code
                    continue
            
            # 如果调试失败或达到最大尝试次数，返回最后的错误
            if len(result) == 3:
                return False, current_code, output
            else:
                return False, current_code, output, []
        
        # 达到最大尝试次数
        if len(result) == 3:
            return False, current_code, "达到最大调试尝试次数，无法修复代码"
        else:
            return False, current_code, "达到最大调试尝试次数，无法修复代码", []

    # 添加一个方法，用于在处理完后清理项目目录
    async def schedule_cleanup(self, project_dir, delay_seconds=60):
        """安排延迟清理项目目录"""
        async def delayed_cleanup():
            try:
                await asyncio.sleep(delay_seconds)
                if os.path.exists(project_dir):
                    try:
                        shutil.rmtree(project_dir)
                        logger.debug(f"已清理项目目录: {project_dir}")
                    except Exception as e:
                        logger.error(f"清理项目目录失败 {project_dir}: {e}")
            except Exception as e:
                logger.error(f"延迟清理任务出错: {e}")
        
        # 创建异步任务
        asyncio.create_task(delayed_cleanup())

    @filter.command("run_python", "python")
    async def run_python(self, event: AstrMessageEvent, task_description: str):
        """根据描述生成并执行Python代码，支持自动调试"""
        if not task_description:
            yield event.chain_result([Plain("请提供任务描述")])
            return
        
        # 生成项目ID
        project_id = f"python_{event.session_id}_{hash(task_description) % 10000:04d}"
        
        # 生成代码
        yield event.chain_result([Plain("正在根据描述生成Python代码...")])
        code = await generate_code(task_description, "python", event.session_id, self.context)
        
        if not code:
            yield event.chain_result([Plain("代码生成失败，请尝试更清晰的描述")])
            return
        
        # 执行代码并自动调试
        yield event.chain_result([Plain(f"生成的代码:\n```python\n{code}\n```\n\n正在执行...")])
        result = await self.execute_with_debug(code, "python", event.session_id, project_id)
        
        # 解析执行结果
        if len(result) == 3:  # 旧格式：(success, final_code, result)
            success, final_code, result_text = result
            
            # 如果代码被修改过，显示最终代码
            if final_code != code:
                yield event.chain_result([Plain(f"调试后的代码:\n```python\n{final_code}\n```\n\n执行结果:")])
            
            # 返回结果
            status = "执行成功" if success else "执行失败"
            yield event.chain_result([Plain(f"{status}:\n{result_text}")])
        else:  # 新格式：(success, final_code, result_text, components)
            success, final_code, result_text, components = result
            
            # 如果代码被修改过，显示最终代码
            if final_code != code:
                yield event.chain_result([Plain(f"调试后的代码:\n```python\n{final_code}\n```\n\n执行结果:")])
            
            # 构建消息链
            message_chain = []
            status = "执行成功" if success else "执行失败"
            message_chain.append(Plain(text=f"{status}:\n"))
            
            # 添加各种输出组件
            for output_type, content in components:
                if output_type == "text":
                    message_chain.append(Plain(text=content))
                elif output_type == "image":
                    message_chain.append(Image(file=content))
                elif output_type == "file":
                    # 简单起见，将文件转换为文本链接描述
                    message_chain.append(Plain(text=f"生成的文件: {os.path.basename(content)}"))
            
            # 返回消息链
            yield event.chain_result(message_chain)
        
        # 安排清理项目目录，延迟60秒
        if len(result) > 3:  # 如果是新格式，我们需要延迟清理以保留文件
            project_dir = os.path.join(CODE_STORAGE_DIR, project_id)
            await self.schedule_cleanup(project_dir, 60)

    @filter.command("run_js", "javascript", "js")
    async def run_javascript(self, event: AstrMessageEvent, task_description: str):
        """根据描述生成并执行JavaScript代码，支持自动调试"""
        if not task_description:
            yield event.chain_result([Plain("请提供任务描述")])
            return
        
        # 生成项目ID
        project_id = f"js_{event.session_id}_{hash(task_description) % 10000:04d}"
        
        # 生成代码
        yield event.chain_result([Plain("正在根据描述生成JavaScript代码...")])
        code = await generate_code(task_description, "javascript", event.session_id, self.context)
        
        if not code:
            yield event.chain_result([Plain("代码生成失败，请尝试更清晰的描述")])
            return
        
        # 执行代码并自动调试
        yield event.chain_result([Plain(f"生成的代码:\n```javascript\n{code}\n```\n\n正在执行...")])
        success, final_code, result = await self.execute_with_debug(code, "javascript", event.session_id, project_id)
        
        # 如果代码被修改过，显示最终代码
        if final_code != code:
            yield event.chain_result([Plain(f"调试后的代码:\n```javascript\n{final_code}\n```\n\n执行结果:")])
        
        # 返回结果
        status = "执行成功" if success else "执行失败"
        yield event.chain_result([Plain(f"{status}:\n{result}")])

    @filter.command("run")
    async def run_custom(self, event: AstrMessageEvent, task_or_lang: str, description: str = ""):
        """根据描述生成并执行指定语言的代码，支持自动调试"""
        # 解析参数
        language = "python"  # 默认为Python
        task_description = ""
        
        # 检查是否指定了语言
        if task_or_lang.startswith("language="):
            lang_param = task_or_lang.split("=", 1)
            if len(lang_param) == 2:
                language = lang_param[1].lower()
                task_description = description
        else:
            # 如果第一个参数不是语言参数，将其作为任务描述的一部分
            task_description = task_or_lang
            if description:
                task_description += " " + description
        
        if not task_description:
            yield event.chain_result([Plain("请提供任务描述")])
            return
        
        # 检查语言是否支持
        if language not in SUPPORTED_LANGUAGES:
            yield event.chain_result([Plain(f"不支持的编程语言: {language}。支持的语言: {', '.join(SUPPORTED_LANGUAGES)}")])
            return
        
        # 生成项目ID
        project_id = f"{language}_{event.session_id}_{hash(task_description) % 10000:04d}"
        
        # 生成代码
        yield event.chain_result([Plain(f"正在根据描述生成{language}代码...")])
        code = await generate_code(task_description, language, event.session_id, self.context)
        
        if not code:
            yield event.chain_result([Plain("代码生成失败，请尝试更清晰的描述")])
            return
        
        # 执行代码并自动调试
        yield event.chain_result([Plain(f"生成的代码:\n```{language}\n{code}\n```\n\n正在执行...")])
        success, final_code, result = await self.execute_with_debug(code, language, event.session_id, project_id)
        
        # 如果代码被修改过，显示最终代码
        if final_code != code:
            yield event.chain_result([Plain(f"调试后的代码:\n```{language}\n{final_code}\n```\n\n执行结果:")])
        
        # 返回结果
        status = "执行成功" if success else "执行失败"
        yield event.chain_result([Plain(f"{status}:\n{result}")])

    @filter.command("run_with_args")
    async def run_with_args(self, event: AstrMessageEvent, *args):
        """使用可变参数的方式执行代码生成和运行"""
        if not args:
            yield event.chain_result([Plain("请提供任务描述")])
            return
            
        # 默认使用Python
        language = "python"
        task_description = ""
        
        # 解析第一个参数是否是语言选择
        if args[0].startswith("language="):
            lang_param = args[0].split("=", 1)
            if len(lang_param) == 2:
                language = lang_param[1].lower()
                # 移除第一个参数（语言选择）
                task_description = " ".join(args[1:])
        else:
            # 使用所有参数作为任务描述
            task_description = " ".join(args)
        
        if not task_description:
            yield event.chain_result([Plain("请提供任务描述")])
            return
            
        # 检查语言是否支持
        if language not in SUPPORTED_LANGUAGES:
            yield event.chain_result([Plain(f"不支持的编程语言: {language}。支持的语言: {', '.join(SUPPORTED_LANGUAGES)}")])
            return
        
        # 生成项目ID
        project_id = f"{language}_{event.session_id}_{hash(task_description) % 10000:04d}"
        
        # 生成代码
        yield event.chain_result([Plain(f"正在根据描述生成{language}代码...")])
        code = await generate_code(task_description, language, event.session_id, self.context)
        
        if not code:
            yield event.chain_result([Plain("代码生成失败，请尝试更清晰的描述")])
            return
        
        # 执行代码并自动调试
        yield event.chain_result([Plain(f"生成的代码:\n```{language}\n{code}\n```\n\n正在执行...")])
        success, final_code, result = await self.execute_with_debug(code, language, event.session_id, project_id)
        
        # 如果代码被修改过，显示最终代码
        if final_code != code:
            yield event.chain_result([Plain(f"调试后的代码:\n```{language}\n{final_code}\n```\n\n执行结果:")])
        
        # 返回结果
        status = "执行成功" if success else "执行失败"
        yield event.chain_result([Plain(f"{status}:\n{result}")])

    @filter.command("clean_code")
    async def clean_storage(self, event: AstrMessageEvent):
        """手动清理代码存储目录"""
        yield event.chain_result([Plain("正在清理代码存储目录...")])
        await cleanup_old_projects(max_projects=0, max_age_days=0)  # 彻底清理所有项目
        yield event.chain_result([Plain("清理完成！所有代码文件已删除。")])

    async def terminate(self):
        # 退出时清理所有文件
        await cleanup_old_projects(max_projects=0, max_age_days=0) 