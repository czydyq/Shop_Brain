from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType


from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, LLMError, MilvusError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.processor.import_processor import config


class ItemNameRecognitionNode(BaseNode):

    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        1. 构建商品识别的LLM上下文
        2. 调用LLM模型进行商品名识别
        3. 将LLM提取到的商品名进行嵌入
        4. 存储到milvus
        Args:
            state:

        Returns:

        """
        # 1. 参数校验
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._validate_state(state)

        # 2. 构建上下文
        item_name_context = self._prepare_llm_context(chunks, item_name_chunk_k)

        # 3. 调用LLM模型
        item_name = self._recognition_item_name(item_name_context, file_title)

        # 4. 向量化
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5. 入库
        self._insert_milvus(dense_vector, sparse_vector, file_title, item_name)

        # 6. 回填
        self._fill_item_name(state, chunks, item_name)

        return state

    def _validate_state(self, state) -> Tuple[str, List, int, int]:
        """

        Args:
            state:

        Returns:

        """
        # 1. 获取文档的标题
        file_title = state['file_title']

        # 2. 判断文档标题
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)

        # 3. 获取chunks
        chunks = state['chunks']

        # 4. 判断chunks
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        # 5. 获取item_name_chunk_k, item_name_chunk_size
        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size

        # 6. 判断item_name_chunk_k, item_name_chunk_size
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(message="商品名识别的辅助切片数不合法")

        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(message="商品名识别的辅助切片长度不合法")

        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_llm_context(self, chunks: List[Dict], item_name_chunk_k: int) -> str:
        """
        准备商品名识别的上下文
        Args:
            chunks:文档的所有切片
            item_name_chunk_k:准备使用的块数

        Returns:
            上下文信息

        """
        # 1. 遍历
        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):

            # 1.1 不是字典类型直接过滤该块
            if not isinstance(chunk, dict):
                continue

            # 1.2 获取chunk中的content
            content = chunk.get('content')
            splice_context = f"【切片】- {index} - {content}"
            final_context.append(splice_context)

        return "\n".join(final_context)

    def _recognition_item_name(self, item_name_context: str, file_title: str) -> str:
        """
        调用LLM模型获得商品名
        Args:
            item_name_context: 商品名的上下文
            file_title: 文件的标题

        Returns:
            商品名

        """
        # 1. 创建LLM客户端
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"OpenAI 的LLM客户端创建失败：{str(e)}")
            return file_title
        
        # 2. 调用LLM

        # 2.1 获取商品名识别的llm系统提示词
        system_prompt = ITEM_NAME_SYSTEM_PROMPT

        # 2.2 获取商品名识别的用户提示词
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)

        # 3. 调用
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            llm_result = llm_response.content.strip('')
            if not llm_result or llm_result == 'UNKNOWN':
                self.logger.error(f"LLM提取商品名失败")
                return file_title
            self.logger.info(f"LLM为文档：{file_title}，提取到的商品名为：{llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM提取商品名失败{str(e)}")
            raise LLMError(message=str(e))
            return file_title

    def _embedding_item_name(self, item_name: str) -> Tuple[Optional[List], Optional[Dict[str, Any]]]:
        """
        商品名嵌入
        Args:
            item_name: 商品名

        Returns:
            稠密向量和稀疏向量

        """
        # 1. 获取模型
        try:
            bge_m3_client: BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE_M3客户端模型创建失败：{str(e)}")
            return None, None
        try:
            # 2. 计算稀疏和稠密向量
            vector_result = bge_m3_client.encode_documents(documents=[item_name])

            # 3. 解析稠密向量和稀疏向量
            # 3.1 获取稠密向量
            dense_vector = vector_result.get('dense')[0].tolist()

            # 3.2 获取稀疏向量矩阵csr(解开)
            sparse_csr = vector_result.get('sparse')
            # 3.2.1 获取行索引
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            # 3.2.2 获取token_id
            token_id = sparse_csr.indices[start_index:end_index].tolist()
            # 3.2.3 获取weight
            weight = sparse_csr.data[start_index:end_index].tolist()
            # 3.2.4 构建{"token_id":weight}字典结构
            sparse_vector = dict(zip(token_id, weight))

            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"BGE模型计算向量失败：{str(e)}")

    def _insert_milvus(self, dense_vector: List, sparse_vector: Dict[str, Any], file_title: str, item_name: str):
        """
        将LLM识别到的商品名保存到milvus数据库中
        row行记录{"dense_vector":"值","sparse_vector":"值","file_title":"文档","item_name":"商品名"}
        Args:
            dense_vector:商品名的稠密向量
            sparse_vector:商品名的稀疏向量
            file_title:文档名
            item_name:商品名

        Returns:

        """
        # 1. 判断稀疏向量和稠密向量是否存在
        if not dense_vector or not sparse_vector:
            return

        # 2. 获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端连接失败，{str(e)}")
            raise MilvusError(node_name=self.name, message="Milvus客户端连接失败", cause=e)

        # 3. 获取集合名字
        item_name_collection_name = self.config.item_name_collection
        try:
            # 4. 创建Milvus集合
            if not milvus_client.has_collection(item_name_collection_name):
                self._create_item_name_collection(item_name_collection_name, milvus_client)

            # 5. 构建数据行
            item_name_data_row = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }

            # 6. 插入数据
            inserted_result = milvus_client.insert(collection_name=item_name_collection_name, data=[item_name_data_row])

            # 7. 记录插入结果
            self.logger.info(f"插入的结果:{inserted_result},主键值:{inserted_result.get('ids')}")
        except Exception as e:
            self.logger.error(f"商品名{item_name}插入失败：{str(e)}")
            raise MilvusError(
                node_name=self.name,
                message=f"商品名{item_name}插入集合{item_name_collection_name}失败",
                cause=e
            )

    def _create_item_name_collection(self, item_name_collection_name: str, milvus_client: MilvusClient):
        """
        创建商品名集合
        Args:
            item_name_collection_name:集合的名字
            milvus_client:milvus客户端

        Returns:

        """
        # 1. 创建schema约束
        schema = milvus_client.create_schema()

        # 1.1 创建主键字段约束
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=10)

        # 1.2 创建标量字段的约束
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)

        # 1.3 创建向量字段约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 3. 创建集合
        milvus_client.create_collection(collection_name=item_name_collection_name, schema=schema, index_params=index_params)

        self.logger.info(f"创建{item_name_collection_name}集合成功")

    def _fill_item_name(self, state: ImportGraphState, chunks: List[Dict], item_name: str):
        """
        回填item_name
        位置一：回填给chunk, 方便下游模型使用
        位置二：回填给state, 方便其他节点使用
        Args:
            state:
            item_name:

        Returns:

        """
        # 1. 更新chunk
        for chunk in chunks:
            chunk["item_name"] = item_name

        # 2. 更新state
        state["item_name"] = item_name

import json
from knowledge.processor.import_processor.base import setup_logging

if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json

    temp_dir = Path(r"D:\Python project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir")
    chunk_json_path = temp_dir / "chunks.json"
    output_path = temp_dir / "chunks_item_name.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)


    # 2. 构建state
    state = {
        "file_title": "万用表的使用",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")

    print(f"item_name:{result.get('item_name')}生成完成，结果已保存至:\n{output_path}")







        













