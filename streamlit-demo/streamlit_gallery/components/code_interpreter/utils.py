import base64
import os
import queue
import re
from io import BytesIO
from subprocess import PIPE

import jupyter_client
from PIL import Image
from loguru import logger

IPYKERNEL = os.environ.get('IPYKERNEL', 'llm')  # 从环境变量获取IPYKERNEL名称，默认值为'llm'

class CodeKernel:  # 定义CodeKernel类
    def __init__(  # 初始化方法
        self,
        kernel_name='kernel',  # 内核名称，默认值为'kernel'
        kernel_id=None,  # 内核ID，默认值为None
        kernel_config_path="",  # 内核配置路径，默认值为空字符串
        python_path=None,  # Python路径，默认值为None
        ipython_path=None,  # IPython路径，默认值为None
        init_file_path="./startup.py",  # 初始化文件路径，默认值为"./startup.py"
        verbose=1,  # 是否详细输出，默认值为1
    ):

        self.kernel_name = kernel_name  # 设置内核名称
        self.kernel_id = kernel_id  # 设置内核ID
        self.kernel_config_path = kernel_config_path  # 设置内核配置路径
        self.python_path = python_path  # 设置Python路径
        self.ipython_path = ipython_path  # 设置IPython路径
        self.init_file_path = init_file_path  # 设置初始化文件路径
        self.verbose = verbose  # 设置是否详细输出

        if python_path is None and ipython_path is None:  # 如果Python路径和IPython路径都未设置
            env = None  # 环境变量为None
        else:
            env = {"PATH": self.python_path + ":$PATH", "PYTHONPATH": self.python_path}  # 设置环境变量

        # 初始化后台内核
        self.kernel_manager = jupyter_client.KernelManager(
            kernel_name=IPYKERNEL,  # 内核名称
            connection_file=self.kernel_config_path,  # 连接文件路径
            exec_files=[self.init_file_path],  # 执行文件
            env=env,  # 环境变量
        )
        if self.kernel_config_path:  # 如果内核配置路径存在
            self.kernel_manager.load_connection_file()  # 加载连接文件
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)  # 启动内核
            logger.info("Backend kernel started with the configuration: {}".format(
                self.kernel_config_path))  # 记录内核启动信息
        else:
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)  # 启动内核
            logger.info("Backend kernel started with the configuration: {}".format(
                self.kernel_manager.connection_file))  # 记录内核启动信息

        if verbose:  # 如果详细输出
            logger.info(self.kernel_manager.get_connection_info())  # 记录内核连接信息

        # 初始化代码内核
        self.kernel = self.kernel_manager.blocking_client()  # 获取阻塞客户端
        # self.kernel.load_connection_file()
        self.kernel.start_channels()  # 启动通信通道
        logger.info("Code kernel started!")  # 记录内核启动信息

    def execute(self, code):  # 执行代码的方法
        self.kernel.execute(code)  # 执行代码
        try:
            shell_msg = self.kernel.get_shell_msg(timeout=30)  # 获取shell消息
            io_msg_content = self.kernel.get_iopub_msg(timeout=30)['content']  # 获取iopub消息内容
            while True:
                msg_out = io_msg_content  # 设置输出消息
                try:
                    io_msg_content = self.kernel.get_iopub_msg(timeout=30)['content']  # 获取iopub消息内容
                    if 'execution_state' in io_msg_content and io_msg_content['execution_state'] == 'idle':  # 如果执行状态为空闲
                        break  # 退出循环
                except queue.Empty:
                    break  # 退出循环
            return shell_msg, msg_out  # 返回shell消息和输出消息
        except Exception as e:  # 捕获异常
            logger.error(e)  # 记录错误信息
            return None  # 返回None

    def execute_interactive(self, code, verbose=False):  # 交互式执行代码的方法
        shell_msg = self.kernel.execute_interactive(code)  # 交互式执行代码
        if shell_msg is queue.Empty:  # 如果shell消息为空
            if verbose:
                logger.warning("Timeout waiting for shell message.")  # 记录超时警告
        self.check_msg(shell_msg, verbose=verbose)  # 检查消息
        return shell_msg  # 返回shell消息

    def inspect(self, code, verbose=False):  # 检查代码的方法
        _ = self.kernel.inspect(code)  # 检查代码
        shell_msg = self.kernel.get_shell_msg(timeout=30)  # 获取shell消息
        if shell_msg is queue.Empty:  # 如果shell消息为空
            if verbose:
                logger.warning("Timeout waiting for shell message.")  # 记录超时警告
        self.check_msg(shell_msg, verbose=verbose)  # 检查消息
        return shell_msg  # 返回shell消息

    def get_error_msg(self, msg, verbose=False) -> str | None:  # 获取错误消息的方法
        if msg['content']['status'] == 'error':  # 如果消息状态为错误
            try:
                error_msg = msg['content']['traceback']  # 获取错误消息
            except:
                try:
                    error_msg = msg['content']['traceback'][-1].strip()  # 获取最后一条错误消息
                except:
                    error_msg = "Traceback Error"  # 设置默认错误消息
            if verbose:
                logger.error("Error: ", error_msg)  # 记录错误消息
            return error_msg  # 返回错误消息
        return None  # 返回None

    def check_msg(self, msg, verbose=False):  # 检查消息的方法
        status = msg['content']['status']  # 获取消息状态
        if status == 'ok':  # 如果状态为ok
            if verbose:
                logger.success("Execution succeeded.")  # 记录成功消息
        elif status == 'error':  # 如果状态为错误
            for line in msg['content']['traceback']:  # 遍历错误消息
                if verbose:
                    logger.error(line)  # 记录每行错误消息

    def shutdown(self):  # 关闭内核的方法
        # 关闭后台内核
        self.kernel_manager.shutdown_kernel()  # 关闭内核管理器中的内核
        logger.info("Backend kernel shutdown.")  # 记录内核关闭消息
        # 关闭代码内核
        self.kernel.shutdown()  # 关闭代码内核
        logger.info("Code kernel shutdown.")  # 记录代码内核关闭消息

    def restart(self):  # 重启内核的方法
        # 重启后台内核
        self.kernel_manager.restart_kernel()  # 重启内核管理器中的内核

    def interrupt(self):  # 中断内核的方法
        # 中断后台内核
        self.kernel_manager.interrupt_kernel()  # 中断内核管理器中的内核

    def is_alive(self):  # 检查内核是否存活的方法
        return self.kernel.is_alive()  # 返回内核是否存活的状态


def b64_2_img(data):  # 将base64编码转换为图像的方法
    buff = BytesIO(base64.b64decode(data))  # 解码base64数据并存储在缓冲区中
    return Image.open(buff)  # 打开缓冲区中的图像并返回


def clean_ansi_codes(input_string):  # 清除ANSI代码的方法
    ansi_escape = re.compile(r'(\x9B|\x1B\[|\u001b\[)[0-?]*[ -/]*[@-~]')  # 定义ANSI转义序列的正则表达式
    return ansi_escape.sub('', input_string)  # 使用正则表达式替换ANSI转义序列为空字符串


def execute(code, kernel: CodeKernel) -> tuple[str, str | Image.Image]:  # 执行代码的方法
    res = ""  # 初始化结果字符串
    res_type = None  # 初始化结果类型
    code = code.replace("", "")  # 替换代码中的空字符串
    code = code.replace("interpreter", "")  # 替换代码中的"interpreter"字符串
    code = code.replace("", "")  # 替换代码中的空字符串
    code = code.replace("", "")  # 替换代码中的空字符串
    code = code.replace("", "")  # 替换代码中的空字符串
    msg, output = kernel.execute(code)  # 执行代码并获取消息和输出

    if msg['metadata']['status'] == "timeout":  # 如果消息状态为超时
        return res_type, 'Timed out'  # 返回超时结果
    elif msg['metadata']['status'] == 'error':  # 如果消息状态为错误
        return res_type, clean_ansi_codes('\n'.join(kernel.get_error_msg(msg, verbose=True)))  # 返回清除ANSI代码后的错误消息

    if 'text' in output:  # 如果输出中包含文本
        res_type = "text"  # 设置结果类型为文本
        res = output['text']  # 获取输出文本
    elif 'data' in output:  # 如果输出中包含数据
        for key in output['data']:  # 遍历输出数据的键
            if 'text/plain' in key:  # 如果键中包含纯文本
                res_type = "text"  # 设置结果类型为文本
                res = output['data'][key]  # 获取输出数据中的文本
            elif 'image/png' in key:  # 如果键中包含PNG图像
                res_type = "image"  # 设置结果类型为图像
                res = output['data'][key]  # 获取输出数据中的图像
                break  # 退出循环

    if res_type == "image":  # 如果结果类型为图像
        return res_type, b64_2_img(res)  # 返回图像结果
    elif res_type == "text" or res_type == "traceback":  # 如果结果类型为文本或回溯
        res = res  # 设置结果为自身
    return res_type, res  # 返回结果类型和结果


def extract_code(text: str) -> str:  # 提取代码的方法
    pattern = r'```([^\n]*)\n(.*?)```'  # 定义提取代码的正则表达式模式
    matches = re.findall(pattern, text, re.DOTALL)  # 使用正则表达式查找匹配项
    return matches[-1][1]  # 返回最后一个匹配项中的代码


def postprocess_text(text: str) -> str:  # 后处理文本的方法
    text = text.replace("\(", "$")  # 替换文本中的左圆括号
    text = text.replace("\)", "$")  # 替换文本中的右圆括号
    text = text.replace("\[", "$$")  # 替换文本中的左方括号
    text = text.replace("\]", "$$")  # 替换文本中的右方括号
    text = text.replace("", "")  # 替换文本中的空字符串
    text = text.replace("", "")  # 替换文本中的空字符串
    text = text.replace("", "")  # 替换文本中的空字符串
    text = text.replace("", "")  # 替换文本中的空字符串
    return text.strip()  # 返回去除首尾空格后的文本

