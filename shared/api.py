import os
import uuid

def _get_magic_code():
    """获取唯一标识符，防止注入攻击"""
    return os.getenv("MAGIC_CODE", "default")

def send_text(text: str):
    """发送文本消息
    
    Args:
        text (str): 要发送的文本内容
    """
    print(f"[CODERUNNER_TEXT_OUTPUT#{_get_magic_code()}]: {text}")

def send_image(image_path: str):
    """发送图像文件
    
    Args:
        image_path (str): 图像文件路径，应该在output目录下
    """
    if not os.path.exists(image_path):
        raise Exception(f"图像文件不存在: {image_path}")
    print(f"[CODERUNNER_IMAGE_OUTPUT#{_get_magic_code()}]: {image_path}")

def send_file(file_path: str):
    """发送任意类型文件
    
    Args:
        file_path (str): 文件路径，应该在output目录下
    """
    if not os.path.exists(file_path):
        raise Exception(f"文件不存在: {file_path}")
    print(f"[CODERUNNER_FILE_OUTPUT#{_get_magic_code()}]: {file_path}") 