import subprocess
import time,json
from typing import Tuple
from pathlib import Path
from knowledge.processor.import_processor.base import BaseNode,setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError,PdfConversionError


class PdfToMdNode(BaseNode):

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点的处理逻辑入口
        :param state:导入图谱节点状态
        :return:
        """
        # 1.获取导入文件的路径和输出的目录
        import_file_path_obj,file_dir_obj = self._validate_state(state)

        # 2.执行mineru解析
        processed_code = self._execute_mineru_parse(import_file_path_obj,file_dir_obj)
        if processed_code != 0:
            raise PdfConversionError(message="MinerU解析PDF失败",node_name=self.name)

        # 3.获取解析后的md_path
        md_path = self.get_md_path(import_file_path_obj,file_dir_obj)

        # 4.更新state
        state['md_path'] = md_path

        return state

    def _validate_state(self,state:ImportGraphState) -> Tuple[Path, Path]:
        """
        :param state:导入图谱节点状态
        :return:导入文件的路径以及输出目录
        """

        self.log_step("step1","准备获取和检验解析文件路径和输出目录")

        # 1.获取解析的文件路径Path
        import_file_path = state.get('import_file_path','')

        # 2.判断文件路径是否为空
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_pate", expected_type=str)

        # 3.标准化解析文件的路径
        import_file_path_obj = Path(import_file_path)

        # 4.判断是否是一个有效路径
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_pate", expected_type=str,
                                  message="解析文件的路径不存在")

        # 5.获取输出文件目录
        file_dir = state.get('file_dir','')

        # 6. 判断输出文件目录是否为空
        if not file_dir:
            file_dir = import_file_path_obj.parent

        # 7.标准化输出目录
        file_dir_obj = Path(file_dir)

        # 8.判断是否是一个有效目录
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str,
                                  message="输出目录不存在")

        self.logger.info(f"解析的文件路径{import_file_path_obj}")
        self.logger.info(f"解析的输出目录{file_dir_obj}")

        return import_file_path_obj,file_dir_obj

    def _execute_mineru_parse(self,import_file_path_obj: Path,file_dir_obj: Path) -> int:
        """
        :param import_file_path_obj:解析的文件路径
        :param file_dir_obj:解析的输出目录
        :return:状态（0或非0）
        0：表示解析成功
        非0：失败
        """
        # 1.定义cmd
        cmd = [
            "mineru",
            "-p",
            str(import_file_path_obj),
            "-o",
            str(file_dir_obj),
            "--source",
            "local"
        ]

        # 2.利用子进程执行cmd命令
        start_time = time.time()
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            encoding="utf-8",
            bufsize=1
        )

        # 3.实时打印日志
        for line in proc.stdout:
            self.logger.info(f"MinerU解析产生的日志：{line}")

        # 4.主线程等待子进程做完
        processed_result = proc.wait()
        end_time = time.time()
        if processed_result ==0 :
            self.logger.info(f"MinerU解析PDF成功，耗时：{end_time - start_time:.2f}s")
        else:
            self.logger.info(f"MinerU解析PDF失败")

        return processed_result

    def get_md_path(self, import_file_path_obj: Path, file_dir_obj: Path) -> str:
        """
        :param import_file_path_obj:解析的文件路径
        :param file_dir_obj:解析的输出目录
        :return:
        """
        file_name = import_file_path_obj.stem

        return str(file_dir_obj/ file_name / "hybrid_auto" / f"{file_name}.md")

if __name__ == '__main__':

    setup_logging()

    pdf_to_md_node = PdfToMdNode()

    init_state = {
        "import_file_path":r"D:\Python project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\致准研究生-2026版.pdf",
        "file_dir":r"D:\Python project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir",
    }

    result = pdf_to_md_node.process(init_state)

    result_str = json.dumps(result, ensure_ascii=False,indent=4)

    print(result_str)

