import re
from typing import List, Optional, Any

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

__all__ = [
    "ChineseRecursiveTextSplitter",
    "zh_title_enhance",
]


def _split_text_with_regex_from_end(
<<<<<<< HEAD
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # 现在我们有了分隔符，开始对文本进行分割
    if separator:
        if keep_separator:
            # 如果要保留分隔符，使用括号将分隔符包围在正则表达式中
            _splits = re.split(f"({separator})", text)
            # 将分隔符和分割后的部分组合起来，形成最终的分割结果
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            # 如果剩余部分为奇数个，则将最后一个部分添加到结果中
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            # 不保留分隔符，直接使用正则表达式分割文本
            splits = re.split(separator, text)
    else:
        # 如果没有指定分隔符，按字符列表处理文本
        splits = list(text)

    # 返回结果列表，去除空字符串
=======
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
<<<<<<< HEAD
        final_chunks = []

        # 获取适合使用的分隔符
        separator = separators[-1]
        new_separators = []

        # 遍历分隔符列表，找到合适的分隔符
=======
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

<<<<<<< HEAD
        # 递归合并长文本，并按长度切分
=======
        # Now go merging things, recursively splitting longer texts.
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
<<<<<<< HEAD

        # 处理剩余的 _good_splits
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        # 移除多余的换行符并返回结果
=======
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]


def under_non_alpha_ratio(text: str, threshold: float = 0.5):
<<<<<<< HEAD
    """
    检查文本片段中非字母字符的比例是否超过给定的阈值。

    这有助于防止像 "-----------BREAK---------" 这样的文本被误认为是标题或叙述文本。
    比例不考虑空格。

    Parameters
    ----------
    text : str
        要测试的输入字符串。
    threshold : float, optional
        如果非字母字符的比例超过此阈值，则函数返回 False，默认为 0.5。

    Returns
    -------
    bool
        如果非字母字符的比例小于阈值，则返回 True；否则返回 False。
    """
    # 检查输入文本是否为空，若为空则直接返回 False
    if len(text) == 0:
        return False

    # 计算文本中的字母字符和总字符数（不包括空格）
    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])

    try:
        # 计算字母字符占总字符数的比例
        ratio = alpha_count / total_count
        return ratio < threshold  # 比较比例和阈值，返回结果
    except Exception as e:
        # 如果计算比例时出现异常（例如除以零），则返回 False
        return False



=======
    """Checks if the proportion of non-alpha characters in the text snippet exceeds a given
    threshold. This helps prevent text like "-----------BREAK---------" from being tagged
    as a title or narrative text. The ratio does not count spaces.

    Parameters
    ----------
    text
        The input string to test
    threshold
        If the proportion of non-alpha characters exceeds this threshold, the function
        returns False
    """
    if len(text) == 0:
        return False

    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    try:
        ratio = alpha_count / total_count
        return ratio < threshold
    except Exception as e:
        return False


>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
def is_possible_title(
    text: str,
    title_max_word_length: int = 20,
    non_alpha_threshold: float = 0.5,
) -> bool:
    """Checks to see if the text passes all the checks for a valid title.

    Parameters
    ----------
    text
        The input text to check
    title_max_word_length
        The maximum number of words a title can contain
    non_alpha_threshold
        The minimum number of alpha characters the text needs to be considered a title
    """

    # 文本长度为0的话，肯定不是title
    if len(text) == 0:
        print("Not a title. Text is empty.")
        return False

    # 文本中有标点符号，就不是title
    ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
    ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
    if ENDS_IN_PUNCT_RE.search(text) is not None:
        return False

    # 文本长度不能超过设定值，默认20
    # NOTE(robinson) - splitting on spaces here instead of word tokenizing because it
    # is less expensive and actual tokenization doesn't add much value for the length check
    if len(text) > title_max_word_length:
        return False

    # 文本中数字的占比不能太高，否则不是title
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    # NOTE(robinson) - Prevent flagging salutations like "To My Dearest Friends," as titles
    if text.endswith((",", ".", "，", "。")):
        return False

    if text.isnumeric():
        print(f"Not a title. Text is all numeric:\n\n{text}")  # type: ignore
        return False

    # 开头的字符内应该有数字，默认5个字符内
    if len(text) < 5:
        text_5 = text
    else:
        text_5 = text[:5]
    alpha_in_text_5 = sum(list(map(lambda x: x.isnumeric(), list(text_5))))
    if not alpha_in_text_5:
        return False

    return True


def zh_title_enhance(docs: List[Document]) -> List[Document]:
<<<<<<< HEAD
    title = None  # 初始化标题为 None
    if len(docs) > 0:  # 如果文档列表不为空
        for doc in docs:  # 遍历文档列表中的每个文档对象
            if is_possible_title(doc.page_content):  # 判断文档内容是否可能是标题
                doc.metadata['category'] = 'cn_Title'  # 将文档的元数据中的分类设置为 'cn_Title'
                title = doc.page_content  # 将当前文档内容设置为标题
            elif title:  # 如果存在标题
                doc.page_content = f"下文与({title})有关。{doc.page_content}"  # 在文档内容前添加与标题相关的描述
        return docs  # 返回处理后的文档列表
    else:
        print("文件不存在")  # 如果文档列表为空，则打印文件不存在的消息


if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则执行以下代码

    # 创建一个 ChineseRecursiveTextSplitter 的实例，用于文本分割
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,    # 是否保留分隔符，默认为 True
        is_separator_regex=True,  # 分隔符是否为正则表达式，默认为 True
        chunk_size=50,          # 每个分割块的大小限制，默认为 50
        chunk_overlap=0         # 分割块之间的重叠部分大小，默认为 0
    )

=======
    title = None
    if len(docs) > 0:
        for doc in docs:
            if is_possible_title(doc.page_content):
                doc.metadata['category'] = 'cn_Title'
                title = doc.page_content
            elif title:
                doc.page_content = f"下文与({title})有关。{doc.page_content}"
        return docs
    else:
        print("文件不存在")


if __name__ == "__main__":
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=50,
        chunk_overlap=0
    )
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
        ]
    for inum, text in enumerate(ls):
<<<<<<< HEAD
        # 使用 enumerate 函数遍历列表 ls 中的每个元素，inum 是索引，text 是元素内容
        print(inum)  # 打印当前索引值 inum
        chunks = text_splitter.split_text(text)  # 使用 text_splitter 对文本 text 进行分割，得到分割后的块列表 chunks
        for chunk in chunks:
            print(chunk)  # 打印每个分割后的文本块 chunk

=======
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
