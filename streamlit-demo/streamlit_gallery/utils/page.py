from typing import Callable, Optional

import streamlit as st


def page_group(param):
<<<<<<< HEAD
    # 构建唯一的键名，基于参数和当前模块名
    key = f"{__name__}_page_group_{param}"

    # 如果键名不在会话状态中，则创建一个新的PageGroup对象并添加到会话状态中
    if key not in st.session_state:
        st.session_state.update({key: PageGroup(param)})

    # 返回对应于键名的PageGroup对象
=======
    key = f"{__name__}_page_group_{param}"

    if key not in st.session_state:
        st.session_state.update({key: PageGroup(param)})

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    return st.session_state[key]


class PageGroup:
<<<<<<< HEAD
    def __init__(self, param):
        self._param: str = param  # 参数名称
        self._default = None  # 默认选择页面
        self._selected = None  # 当前选定的页面回调

        # 用于解决同一运行中多个页面选择的回滚问题
=======

    def __init__(self, param):
        self._param: str = param
        self._default = None
        self._selected = None

        # Fix some rollback issues when multiple pages are selected in the same run.
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        self._backup: Optional[str] = None

    @property
    def selected(self):
<<<<<<< HEAD
        # 获取查询参数的字典表示
        params = st.query_params.to_dict()
        # 如果参数存在于字典中，则返回参数的值；否则返回默认值
        return params[self._param] if self._param in params else self._default

    def item(self, label: str, callback: Callable, default=False) -> None:
        self._backup = None  # 清除备份

        # 构建唯一的键名，基于参数名、标签名和当前模块名
        key = f"{__name__}_{self._param}_{label}"
        # 标准化标签名，去除空格并转换为小写，用连字符代替空格
        page = self._normalize_label(label)

        # 如果设置为默认，则更新默认选择页面
        if default:
            self._default = page

        # 检查当前页面是否被选择
        selected = (page == self.selected)

        # 如果页面被选择，则更新选定回调
        if selected:
            self._selected = callback

        # 将选择状态存储到会话状态中
        st.session_state[key] = selected
        # 创建复选框，显示标签和键，禁用已选择的复选框，设置变更时的回调函数和参数
        st.checkbox(label, key=key, disabled=selected, on_change=self._on_change, args=(page,))

    def show(self) -> None:
        # 如果存在选定的回调，则显示选定的页面内容；否则显示“404 Not Found”
=======
        params = st.query_params.to_dict()
        return params[self._param] if self._param in params else self._default

    def item(self, label: str, callback: Callable, default=False) -> None:
        self._backup = None

        key = f"{__name__}_{self._param}_{label}"
        page = self._normalize_label(label)

        if default:
            self._default = page

        selected = (page == self.selected)

        if selected:
            self._selected = callback

        st.session_state[key] = selected
        st.checkbox(label, key=key, disabled=selected, on_change=self._on_change, args=(page,))

    def show(self) -> None:
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        if self._selected is not None:
            self._selected()
        else:
            st.title("🤷 404 Not Found")

    def _on_change(self, page: str) -> None:
<<<<<<< HEAD
        # 获取查询参数的字典表示
        params = st.query_params.to_dict()

        # 如果备份为空，则备份当前参数值
=======
        params = st.query_params.to_dict()

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        if self._backup is None:
            if self._param in params:
                self._backup = params[self._param][0]
            params[self._param] = [page]
        else:
<<<<<<< HEAD
            # 恢复备份的参数值
            params[self._param] = [self._backup]

        # 更新查询参数
        for key in params:
            st.query_params[key] = params[key]
        # 清空会话状态中的消息
        st.session_state.messages = []

    def _normalize_label(self, label: str) -> str:
        # 标准化标签，将非ASCII字符转换为小写并去除空格，用连字符代替空格
        return "".join(char.lower() for char in label if char.isascii()).strip().replace(" ", "-")

=======
            params[self._param] = [self._backup]

        for key in params:
            st.query_params[key] = params[key]
        st.session_state.messages = []

    def _normalize_label(self, label: str) -> str:
        return "".join(char.lower() for char in label if char.isascii()).strip().replace(" ", "-")
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
