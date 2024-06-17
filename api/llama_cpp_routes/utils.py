from api.models import LLM_ENGINE
from api.utils.request import llama_outer_lock, llama_inner_lock


def get_llama_cpp_engine():
    # NOTE: This double lock allows the currently streaming model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    llama_outer_lock.acquire()  # 获取外层锁，确保当前流式模型可以检查同一线程中是否有其他挂起的请求，并在需要时取消流。
    release_outer_lock = True  # 标记是否需要释放外层锁
    try:
        llama_inner_lock.acquire()  # 获取内层锁，确保当前流式模型在处理期间不会被其他请求干扰
        try:
            llama_outer_lock.release()  # 释放外层锁，允许其他请求在内层锁处于占用状态时检查和取消流
            release_outer_lock = False  # 内层锁已经占用，无需释放外层锁
            yield LLM_ENGINE  # 返回 LLama C++ 引擎实例
        finally:
            llama_inner_lock.release()  # 最终释放内层锁，确保锁的正常释放
    finally:
        if release_outer_lock:
            llama_outer_lock.release()  # 如果需要释放外层锁，则在最终处理中释放它

