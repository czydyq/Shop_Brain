import base64
import re
from collections import deque
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Tuple, List, Set, Optional, Dict, Deque

from openai import OpenAI

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass()
class ImageContext:
    """
    一张图片的上下文信息
    """
    head: str           #上文标题内容
    pre_text: str       #上文内容
    post_text: str      #下文内容

@dataclass()
class ImageInfo:
    """"

    """
    name: str
    path: str
    imag_context: ImageContext



class _MdFileHandler:
    """
    1.读取 md内容、md_path、图片目录
    2.备份新的md_content(方便测试观察)
    """
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self,state: ImportGraphState) -> Tuple[str,Path,Path]:
        """
        1.读取md内容
        2.读取md文件路径
        3.读取图片目录
        Args:
            state: 上一个节点更新后的state
        Returns:
            Tuple[str,Path,Path]
        """

        # 1.获取md文件路径
        md_path = state.get("md_path","")
        if not md_path:
            raise StateFieldError(node_name=self.node_name,field_name="md_path",expected_type=str)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name,field_name="md_path",expected_type=Path)

        # 2.读取md内容
        try:
            with open(md_path_obj,"r",encoding="utf-8") as f:
                md_content = f.read()
        except IOError as e:
            self.logger.error(f"MD文件:{md_path_obj}打开失败")
            raise FileProcessingError(node_name=self.node_name,message="文件打开失败")

        # 3.获取图片目录
        img_dir_obj = md_path_obj.parent / "images"

        return md_content, md_path_obj , img_dir_obj

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise FileProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class _ImageScanner:
    """
    1.根据图片目录，得到该目录下有效的图片文件
    2.去到md文件中定位到图片的位置
    3.获取改图片在md中的上下文内容，帮助模型识别的更加准确
    4.最终组装所有图片的上下文内容
    """
    def __init__(self, logger: Logger):
        self.logger = logger
    def scan_imgs_dir(self, img_dir_obj: Path, md_content: str, image_extensions: Set[str], img_content_length: int) -> List[ImageInfo]:
        """
        1. 扫描指定图片目录下的所有图片文件
        2. 遍历每一个图片文件去MD文件中获取到位置
        3. 将每一个（图片的上下文:ImageContext）放到最终封装每一个图片的完整信息放到ImageInfo中
        4. 返回容器
        Args:
            image_extensions: 图片文件名的扩展
            img_dir_obj:图片目录
            md_content:md内容
            img_content_length:上下文的最大长度

        Returns:
            List[ImageInfo]
        """
        img_info_list = []
        # 1. 遍历图片目录
        for img_path in img_dir_obj.iterdir():
            # 1.1 过滤子目录
            if not img_path.is_file():
                self.logger.error(f"{img_path}不是一个有效文件")
                continue
            # 1.2 过滤不合法的图片文件
            if not img_path.suffix in image_extensions:
                self.logger.error(f"{img_path}不是允许的后缀名")
                continue

            # 1.3 找该图片的上下文
            ctx = self._find_context(img_path.name, md_content, img_content_length)
            if not ctx:
                self.logger.info(f"未找到{img_path}的上下文信息")
                continue

            # 1.4 封装ImageInfo对象并且放到容器中
            img_info_list.append(ImageInfo(
                name=img_path.name,
                path =str(img_path),
                imag_context= ctx
            ))

        self.logger.info(f"MD中找到{len(img_info_list)}个有效的图片引用")
        return img_info_list

    def _find_context(self,img_name:str, md_content:str, img_content_length: int) -> Optional[ImageContext]:
        """
        查找图片的上下文
        Args:
            img_name: 图片名
            md_content: MD内容
            img_content_length: 上下文长度

        Returns:
            找到了------>图片的上下文信息
            没找到------>None
        """
        #1. 预编译
        pattern = re.compile(r"!\[.*?]\(.*?" + re.escape(img_name) + r".*?\)")

        # 2. 按行切割md_content
        md_lines = md_content.split("\n")

        # 3. 遍历每一行以及对应的行索引
        for md_idx, md_line in enumerate(md_lines):

            # 3.1 判断当前行是不是图片
            if not pattern.search(md_line):
                continue

            # 上文
            head, prev_index = self._find_heading_up(md_lines, md_idx)
            pre_lines = md_lines[prev_index + 1:md_idx]
            pre_context = self._extract_limited_context(pre_lines,img_content_length,direction = "front")

            # 下文
            next_index = self._find_heading_down(md_lines, md_idx)
            next_lines = md_lines[md_idx+ 1:next_index]
            post_context = self._extract_limited_context(next_lines,img_content_length,direction = "back")

            return ImageContext(
                head=head,
                pre_text=pre_context,
                post_text=post_context
            )
        return None

    def _find_heading_up(self, md_lines: List[str], from_idx: int) -> Tuple[str, int]:
        """

        Args:
            md_lines: 整个MD内容
            from_idx: 图片的索引

        Returns:
            当前图片最近的上文标题+索引
        """
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i

        return "", -1

    def _find_heading_down(self,md_lines: List[str], from_idx: int) ->  int:
        """

        Args:
            md_lines:整个MD内容
            from_idx:图片的索引

        Returns:
            当前图片最近的下文标题
        """
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i

        return len(md_lines)

    def _extract_limited_context(self, extracted_md_lines: List[str], img_content_length: int, direction: str) -> str:

        """

        Args:
            extracted_md_lines: 上（下）文
            img_content_length: 上下文的长度
            direction: 方向

        Returns:
            上（下）文的内容
        """
        current_paragraph = []
        paragraphs = []

        for line in extracted_md_lines:

            is_blank_line = not line.strip()

            is_other_image = re.match(
                r"^!\[.*?]\(.*?\)$", line.strip()
            )

            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        if direction == "front":
            paragraphs.reverse()

        total = 0
        selected = []
        for paragraph in paragraphs:
            if total + len(paragraph) > img_content_length and selected :
                break
            selected.append(paragraph)
            total += len(paragraph)

        if direction == "front":
            selected.reverse()

        return "\n\n".join(selected)



class _VLMSummarizer:
    """
    主要根据
    """
    def __init__(self, logger: Logger):
        self.logger = logger

    def _summary_all(self, document_name: str, img_info_list: List[ImageInfo], vl_model: str) -> Dict[str, str]:
        """
        为所有图片生成摘要
        Args:
            document_name:文档的名字
            img_info_list:所有的图片信息
            vl_model:vlm模型名字

        Returns:
            Dict[str, str]:{"image_name":""summary}
        """
        summaries = {}
        request_timestamps : Deque[float] = deque()
        # 1. 获取VLM客户端
        try:
            vlm_client = AIClients.get_vlm_client()
        except Exception as e:
            self.logger.error(f"获取VLM客户端失败: {e}")
            for img_info in img_info_list:
                summaries[img_info.name] = "暂无摘要"
            return summaries

        for img_info in img_info_list:
            summaries[img_info.name] =self._summary_one(document_name,img_info, vlm_client, vl_model)

        self.logger.info(f"生成{len(summaries)}图片摘要")
        return summaries

    def _summary_one(self,document_name: str, img_info: ImageInfo, vlm_client: OpenAI, vl_model: str) -> str:
        """
        为当前图片生成摘要
        Args:
            img_info: 图片信息
            vlm_client: 模型客户端
            vl_model: 模型名称

        Returns:
            str:摘要信息

        """
        # 1. 构造VLM需要的上下文
        parts = [p for p in (img_info.imag_context.head, img_info.imag_context.pre_text, img_info.imag_context.post_text) if p]

        # 2. 构建最终的上下文
        final_context = "\n".join(parts) if parts else "暂无上下文"

        # 3. 根据图片地址获取到图片的内容
        try:
            with open(img_info.path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
        except IOError as e:
            return "暂无图片信息"

        # 4. 将上下文传给大模型
        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img_info.path}: {e}")
            return "图片描述"


class _ImageUploader:
    """

    """
    def __init__(self, logger: Logger):
        self.logger = logger

    def upload_and_replace(self, object_dir_name: str, md_content: str, img_info_list: List[ImageInfo],
                           summaries: Dict[str, str],
                           minio_url: str, minio_bucket_name: str) -> str:
        """
        上传文件图片到minio并且更新md中的图片地址以及摘要
        Args:
            object_dir_name:    minio对象目录
            md_content:         md的内容
            img_info_list:      图片信息
            summaries:          图片摘要
            minio_url:          minio地址
            minio_bucket_name:  桶名

        Returns:
            更新后的md内容

        """
        # 1. 上传
        remote_urls = self._upload_all(object_dir_name, img_info_list,  minio_url, minio_bucket_name)

        # 2. 更新
        md_content = self._update_md(md_content, summaries, remote_urls)

        return md_content

    def _upload_all(self, object_dir_name: str, img_info_list: List[ImageInfo], minio_url: str, minio_bucket_name: str) -> Dict[str, str]:
        # 1. 得到minio客户端
        remote_urls = {}
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            for img_info in img_info_list:
                remote_urls[img_info.name] = img_info.path
            return remote_urls

        # 2. 遍历上传每一个
        for img_info in img_info_list:
            object_name = f"{object_dir_name}/{img_info.name}"
            try:
                # 2.1 上传图片到minio
                minio_client.fput_object(minio_bucket_name, object_name, img_info.path)
                remote_urls[img_info.name] = f"{minio_url}/{minio_bucket_name}/{object_name}"
            except Exception as e:
                self.logger.warn(f"图片{img_info.name}的远程地址失效，用本地地址替代")
                remote_urls[img_info.name] = img_info.path

        self.logger.info(f"获取到远程的{len(remote_urls)}个图片地址")
        return remote_urls

    def _update_md(self, md_content: str, summaries: Dict[str, str], remote_urls: Dict[str, str]) -> str:
        """
        更新md的图片描述和远程的图片地址
        Args:
            md_content:   md的内容
            summaries:    vlm生成的摘要
            remote_urls:  minio生成的图片地址

        Returns:
            更新后的md内容

        """
        # 利用正则寻找
        pattern = re.compile(r"!\[(.*?)]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            for img_name, img_summary in summaries.items():
                img_path = match.group(2)
                img_name_in_md = Path(img_path).name

                if img_name == img_name_in_md:
                    return f"![{img_summary}]({remote_urls[img_name]})"

        return pattern.sub(replacer, md_content)



class MarkdownToImageNode(BaseNode):
    """
    1.得到四个类的实例对象
    2.分别调用四个实例对象的处理方法
    """

    name = "md_to_image_node"

    def __init__(self):
        super().__init__()
        self._md_file_handler = _MdFileHandler(self.logger,self.name)
        self._img_scanner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger)
        self._img_uploader = _ImageUploader(self.logger)


    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑
        """
        # 1.操作 md_file_handler 获取md内容，md路径，以及image目录
        config = self.config
        self.log_step("step1","获取md内容，md路径，以及image目录")
        md_content, md_path_obj, img_dir_obj = self._md_file_handler.validate_and_read_md(state)

        if not img_dir_obj.exists():
            state["md_content"] = md_content
            return state

        # 2. 操作_img_scaner
        self.log_step("step2","准备开始扫描图片目录")
        img_info_list:List[ImageInfo] = self._img_scanner.scan_imgs_dir(img_dir_obj,
                                                                        md_content,
                                                                        config.image_extensions,
                                                                        config.img_content_length)
        # 3. 操作_vlm_summarizer
        summaries: Dict[str,str]= self._vlm_summarizer._summary_all(
            md_path_obj.stem,
            img_info_list,
            config.vl_model)

        self.logger.info(f"共生成 {len(summaries)} 个图片摘要：")
        for img_name, summary in summaries.items():
            self.logger.info(f"  {img_name} → {summary[:100]}")  # 只显示前100个字符

        # 4. 操作_img_uploader
        new_md_content = self._img_uploader.upload_and_replace(md_path_obj.stem, md_content, img_info_list, summaries, config.get_minio_base_url(),config.minio_bucket)

        # 5. 备份调配
        self._md_file_handler.backup(md_path_obj, new_md_content)

        state['md_content'] = new_md_content

        return state


if __name__ == "__main__":
    setup_logging()
    md_img_node = MarkdownToImageNode()
    init_state = {
        "md_path": r"D:\Python project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md"
    }
    md_img_node.process(init_state)
