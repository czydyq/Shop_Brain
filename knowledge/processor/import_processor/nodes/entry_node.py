from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState


class EntryNode(BaseNode):

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state:

        Returns:

        """
        # 1. 获取上文的文件
        import_file_path = state.get('import_file_path', '')
        file_dir = state.get('file_dir', '')

        # 2. 判断
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str)
        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str)

        # 3. Path标准化
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)

        # 4. 判读
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=Path)
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="file_dir_obj", expected_type=Path)
        # 5. 获取文件的后缀
        if import_file_path_obj.suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = str(import_file_path_obj)
        elif import_file_path_obj.suffix == '.md':
            state['is_md_read_enabled'] = True
            state['md_path'] = str(import_file_path_obj)
        else:
            self.logger.error(f"该文件的后缀名{import_file_path_obj.suffix}不支持")
            raise ValidationError(node_name=self.name, message="该文件的后缀名{import_file_path_obj.suffix}不支持")


        # 6. 获取到上传文件的表题，更新到state中
        state['file_title'] = import_file_path_obj.stem

        return state