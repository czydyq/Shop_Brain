import os
import re
from typing import Tuple, List, Dict, Any

import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):

    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分的核心逻辑入口
        Args:
            state:

        Returns:

        """


        config = self.config
        # 1. 参数校验
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state, config)

        # 2. 切分（一级策略：根据md文档中的标题来切分）多个章节（章节：标题之间的内容）
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)

        # 3. 二次切分或者合并
        final_section = self._split_and_merge(sections, max_content_length, min_content_length)

        # 4. 组装成chunk对象
        final_chunks = self._assemble_chunks(final_section)

        # 5. 数据备份
        self._back_up(final_chunks, state)

        # 6. 更新state(chunks)
        state['chunks'] = final_chunks

        # 7. 返回
        return state

    def _validate_state(self, state: ImportGraphState, config) -> Tuple[str, str, int, int]:

        self.log_step("step1", "切分文档的参数校验以及获取...")

        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 \
                or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        """
        根据标题来切分，无论几级标题都有可能
        Args:
            md_content:md文档
            file_title:上传文档的标题

        Returns:
            Dict[str, Any]:切分后的章节

        """
        in_fence = False # 不是代码块
        body_lines = []
        sections = [] # 最终收集到的章节对象
        current_title = ""
        hierarchy = [""] * 7 # 存储标题作为sections的父标题使用 hierarchy 层级追踪\
        current_level = 0

        def _flush() -> List[Dict[str, Any]]:
            """
            打包section
            {
            "body": "收集到的所有行"
            "title": "当前内容的标题"
            "parent_title": "当前内容的副标题"
            "file_title": "文档标题"
            }
            Returns:

            """
            parent_title = ""
            # 1. 处理内容行
            body = "\n".join(body_lines)
            if current_title or body:
                # 2. 处理父标题
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                # 3. 如果没有父标题
                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title,
                    "parent_title": parent_title,
                    "file_title": file_title
                })


        # 1. 根据\n来切分md_content
        md_lines = md_content.split("\n")

        # 2. 根据正则去找标题
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3. 遍历md_lines找标题
        for md_line in md_lines:

            # 3.1 先判断是否是代码块
            if md_line.startswith("```") or md_line.startswith("~~~"):
                in_fence = not in_fence

            # 3.2 判读是否要走正则
            match = heading_re.match(md_line) if not in_fence else None

            # 3.3 判断match是否有
            if match:
                # 将收集到的普通行打包成section
                _flush()
                # 当前的标题
                current_title = md_line
                # 当前标题的层级
                level = len(match.group(1))
                # 赋值给hierarchy
                current_level = level
                hierarchy[level] = current_title

                for i in range(level + 1, 7):
                    hierarchy[i] = ""
            else:
                # 普通行或代码块
                body_lines.append(md_line)

        _flush()
        return sections
    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int) -> List[Dict[str, Any]]:
        """
        切分较大章节（section）以及合并较小章节（section）
        Args:
            sections: 所有经过一级切分后的章节
            max_content_length: 最大内容长度
            min_content_length: 最小内容长度（父标题相同的section才会合并）

        Returns:

        """
        # 1. 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self._split_long_section(section, max_content_length))

        # 2. 合并
        final_sections = self._merger_short_section(current_sections, min_content_length)

        return final_sections

    def _split_long_section(self, section: Dict[str, Any], max_content_length: int) -> List[Dict[str, Any]]:
        """
        切分长章节
        Args:
            section: 当前的章节
            max_content_length: 最大长度阈值

        Returns:

        """
        # 1. 获取section的属性
        body = section.get('body')
        title = section.get('title')
        parent_title = section.get('parent_title')
        file_title = section.get('file_title')

        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)
            section['body'] = body

        # 2. 获取标题前缀
        title_prefix = f"{title}\n\n"

        # 3. 获取总长度
        total_length = len(body) + len(title_prefix)

        # 4. 判断总长度是否超过阈值
        if total_length <= max_content_length:
            return [section]

        # 5. 能够切分的长度计算出来
        body_length = max_content_length - len(title_prefix)

        # 6. 切分器（递归切分器）
        # 6.1 切分器对象
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=body_length,
                                                       chunk_overlap=0,
                                                       separators=["\n", "\n\n", "。", "？", "！", "；", ".", "?", ";", " ", ""],
                                                       keep_separator=True)

        # 6.2 切分器对象切分
        sections = text_splitter.split_text(body)

        # 6.3 验证
        if len(sections) == 1:
            return [section]

        # 6.3 遍历
        sub_sections = []
        for index, section in enumerate(sections):
            sub_sections.append({
                "body": section,
                "title": f"{title}_{index + 1}",
                "parent_title": parent_title,
                "file_title": file_title,
            })

        return sub_sections

    def _merger_short_section(self, current_sections, min_content_length: int) -> List[Dict[str, Any]]:
        """
        合并短的章节
        Args:
            current_sections:经过二次切分后的所有章节
            min_content_length:最小的section长度

        Returns:
            合并之后的section对象

        """
        current_section = current_sections[0]
        final_sections = []

        # 2. 遍历合并
        for next_section in current_sections[1:]:
            same_parent = (current_section['parent_title'] == next_section['parent_title'])

            if same_parent and len(current_section.get('body')) < min_content_length:
                # 合并body
                current_section['body'] = (current_section.get('body').rstrip()+ "\n\n" + next_section.get('body').lstrip())

                # 标题退回父标题
                current_section['title'] = current_section['parent_title']
            else:
                # 封箱
                final_sections.append(current_section)
                current_section = next_section
        # 最后一个封箱
        final_sections.append(current_section)

        return final_sections

    def _assemble_chunks(self, final_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        拼接最终的sessions成为chunks
        Args:
            final_sections:

        Returns:

        """
        final_chunks = []
        for section in final_sections:
            body = section.get('body')
            title = section.get('title')
            parent_title = section.get('parent_title')
            file_title = section.get('file_title')

            content = f"{title}\n\n{body}"

            final_chunks.append({
                "content": content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title,
            })
        self.logger.info(f"最终切割能进入嵌入节点的chunks的个数为：{len(final_chunks)}")
        return final_chunks

    def _back_up(self, final_chunks, state: ImportGraphState):
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")



if __name__ == '__main__':
    document_split_node = DocumentSplitNode()
    md_path = r"D:\Python project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用_new.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    init_state = {
        "md_content": md_content,
        "file_title": "万用表的使用",
        "file_dir": r"D:\Python project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir",
    }
    document_split_node.process(state=init_state)








