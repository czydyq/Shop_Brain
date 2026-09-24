import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymilvus import CollectionSchema, DataType, MilvusClient

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import (
    ConfigurationError,
    MilvusError,
    ValidationError,
)
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

# Milvus 自增主键，回填到 chunk 上供下游引用，但不能再写回库
_PRIMARY_KEY_FIELD = "chunk_id"
_DENSE_VECTOR_FIELD = "dense_vector"
_SPARSE_VECTOR_FIELD = "sparse_vector"


@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content",      datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title",         datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title",  datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title",    datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name",     datatype=DataType.VARCHAR, max_length=65535),
)

_SCALAR_FIELD_NAMES: Tuple[str, ...] = tuple(spec.field_name for spec in _SCALAR_FIELDS)


class _MilvusSchemaBuilder:
    @staticmethod
    def build(client: MilvusClient, dim: int) -> CollectionSchema:
        logger.info("开始构建 Schema...")
        schema = client.create_schema(enable_dynamic_field=True)

        schema.add_field(field_name=_PRIMARY_KEY_FIELD, datatype=DataType.INT64,
                         is_primary=True, auto_id=True)
        schema.add_field(field_name=_DENSE_VECTOR_FIELD,
                         datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name=_SPARSE_VECTOR_FIELD,
                         datatype=DataType.SPARSE_FLOAT_VECTOR)

        for spec in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {
                "field_name": spec.field_name,
                "datatype": spec.datatype,
            }
            if spec.max_length is not None:
                kwargs["max_length"] = spec.max_length
            schema.add_field(**kwargs)

        logger.info("Schema 构建完成")
        return schema


class _MilvusIndexBuilder:
    @staticmethod
    def build(client: MilvusClient):
        logger.info("开始构建索引...")
        index = client.prepare_index_params()

        index.add_index(field_name=_DENSE_VECTOR_FIELD, index_name="dense_vector_index",
                        index_type="AUTOINDEX", metric_type="COSINE")
        index.add_index(field_name=_SPARSE_VECTOR_FIELD, index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        logger.info("索引构建完成")
        return index


class _MilvusInserter:
    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """插入记录并把返回的主键回填到入参 chunks 上（原地修改，同时返回）。"""
        logger.info(f"开始插入 {len(chunks)} 条记录到 Milvus...")

        # auto_id=True 时服务端不接受客户端提供主键，重跑/重试时 chunk 上可能已带回填的 chunk_id
        payload = [
            {k: v for k, v in chunk.items() if k != _PRIMARY_KEY_FIELD}
            for chunk in chunks
        ]

        result = self._client.insert(
            collection_name=self._collection_name, data=payload
        )
        ids = list(result.get("ids", []))
        insert_count = result.get("insert_count", 0)

        if insert_count != len(chunks) or len(ids) != len(chunks):
            raise MilvusError(
                message=(
                    f"插入结果与输入数量不一致：输入 {len(chunks)} 条，"
                    f"insert_count={insert_count}，返回主键 {len(ids)} 个"
                )
            )

        for chunk, chunk_id in zip(chunks, ids):
            chunk[_PRIMARY_KEY_FIELD] = chunk_id

        logger.info(f"插入完成：{insert_count} 条记录，已回填 {_PRIMARY_KEY_FIELD}")
        return chunks


class MilvusImportNode(BaseNode):
    name = "milvus_import_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        validated_chunks, dim = self._validate_state(state)

        milvus_client = self._get_milvus_client()
        collection = self._get_collection_name()
        self._create_collection(milvus_client, collection, dim)

        inserter = _MilvusInserter(client=milvus_client, collection_name=collection)
        state["chunks"] = inserter.insert(chunks=validated_chunks)

        return state

    def _get_milvus_client(self) -> MilvusClient:
        try:
            return StorageClients.get_milvus_client()
        except (ConnectionError, EnvironmentError) as e:
            self.logger.error(f"Milvus 客户端创建失败：{e}")
            raise MilvusError(node_name=self.name, message="Milvus 客户端创建失败", cause=e)

    def _get_collection_name(self) -> str:
        collection = self.config.chunks_collection
        if not collection:
            raise ConfigurationError(
                node_name=self.name, message="未配置 CHUNKS_COLLECTION 环境变量"
            )
        return collection

    def _validate_state(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], int]:
        self.log_step("validate", "参数校验")
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("待入库的 chunks 为空或类型无效", self.name)

        validated: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(
                    f"chunks[{i}] 类型无效：期望 dict，实际为 {type(chunk).__name__}",
                    self.name,
                )

            # schema 中声明的标量字段均为非 nullable，缺 key 会导致整批插入失败
            missing = [name for name in _SCALAR_FIELD_NAMES if name not in chunk]
            if missing:
                raise ValidationError(
                    f"chunks[{i}] 缺少 schema 字段：{', '.join(missing)}",
                    self.name,
                )

            if chunk.get(_DENSE_VECTOR_FIELD) and chunk.get(_SPARSE_VECTOR_FIELD):
                validated.append(chunk)
            else:
                self.logger.warning(f"chunks[{i}] 缺少混合向量，已跳过")

        if not validated:
            raise ValidationError("所有 chunk 均无有效向量，无法入库", self.name)

        dim = len(validated[0][_DENSE_VECTOR_FIELD])
        for i, chunk in enumerate(validated[1:], start=1):
            chunk_dim = len(chunk[_DENSE_VECTOR_FIELD])
            if chunk_dim != dim:
                raise ValidationError(
                    f"chunk 向量维度不一致：chunks[{i}] 为 {chunk_dim}，期望 {dim}",
                    self.name,
                )

        self.logger.info(f"有效 chunks：{len(validated)}，向量维度：{dim}")
        return validated, dim

    def _create_collection(self, client: MilvusClient, collection_name: str, dim: int) -> None:
        self.log_step("collection", f"检查集合 {collection_name}")
        if client.has_collection(collection_name=collection_name):
            self._check_collection_dim(client, collection_name, dim)
            self.logger.info(f"集合 {collection_name} 已存在，跳过创建")
            return

        schema = _MilvusSchemaBuilder.build(client, dim)
        index = _MilvusIndexBuilder.build(client)
        client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index
        )
        self.logger.info(f"集合 {collection_name} 创建完成")

    def _check_collection_dim(self, client: MilvusClient, collection_name: str, dim: int) -> None:
        """已存在的集合不能改维度，提前拦截不一致，避免插入时报难懂的错。"""
        try:
            fields = client.describe_collection(collection_name=collection_name).get("fields", [])
        except Exception as e:
            self.logger.warning(f"集合 {collection_name} 维度校验跳过：{e}")
            return

        for field in fields:
            if field.get("name") != _DENSE_VECTOR_FIELD:
                continue
            existing_dim = (field.get("params") or {}).get("dim") or field.get("dim")
            if existing_dim and existing_dim != dim:
                raise MilvusError(
                    node_name=self.name,
                    message=(
                        f"集合 {collection_name} 的 {_DENSE_VECTOR_FIELD} 维度为 {existing_dim}，"
                        f"当前 chunk 为 {dim}，请检查集合名与嵌入模型"
                    ),
                )
            return


if __name__ == "__main__":

    setup_logging()

    temp_dir = Path(r"D:\Python project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir")
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content.get("chunks")}

    node = MilvusImportNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    logger.info(f"结果已保存至: {output_path}")
