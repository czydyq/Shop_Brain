import json
from langgraph.graph.state import StateGraph, CompiledStateGraph
from langgraph.graph import END

from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.md_to_image_node import MarkdownToImageNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_processor.nodes.import_milvus_node import MilvusImportNode
from knowledge.processor.import_processor.state import ImportGraphState


"""
编排节点

定义节点
定义条件边
定义顺序边
运行整个pineline图谱的各个节点

"""

def import_router(state: ImportGraphState):
    """
    根据state中的 is_pdf_read_enabled is_md_read_enabled 决定如何到达下一个节点
    Args:
        state:

    Returns:

    """
    # 1. 获取上传的文件属于pdf or md
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    if state.get("is_md_read_enabled"):
        return "md_to_img_node"
    return END


def import_graph() -> CompiledStateGraph:
    """
    1. 定义运行时图状态workflow
    2. 定义节点（入口节点、普通业务节点）
    3. 定义边（条件边、普通业务边）
    4. 返回运行时状态
    Returns:

    """
    # 1. 定义运行时图状态workflow
    work_flow = StateGraph(ImportGraphState)

    # 2. 定义入口节点
    work_flow.set_entry_point("entry_node")

    # 3. 定义其他节点名和节点实例的映射表
    node_name_obj = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_to_img_node": MarkdownToImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_recognition_node": ItemNameRecognitionNode(),
        "embedding_chunks_node": EmbeddingChunksNode(),
        "import_milvus_node": MilvusImportNode(),
    }

    # 4. 遍历映射表添加
    for node_name, node_obj in node_name_obj.items():
        work_flow.add_node(node_name, node_obj)

    # 5. 定义边
    # 5.1 定义条件边 source path path_map
    work_flow.add_conditional_edges("entry_node",import_router, {
        "pdf_to_md_node": "pdf_to_md_node",
        "md_to_img_node": "md_to_img_node",
        END: END
    })

    # 5.2  定义业务边
    work_flow.add_edge("pdf_to_md_node", "md_to_img_node")
    work_flow.add_edge("md_to_img_node", "document_split_node")
    work_flow.add_edge("document_split_node", "item_name_recognition_node")
    work_flow.add_edge("item_name_recognition_node", "embedding_chunks_node")
    work_flow.add_edge("embedding_chunks_node", "import_milvus_node")
    work_flow.add_edge("import_milvus_node", END)

    # 5.3 编译
    compiled_state_graph = work_flow.compile()
    return compiled_state_graph


import_app  = import_graph()

def run_import_graph():

    graph_state = {
        "import_file_path": r"D:\Python project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用.pdf",
        "file_dir": r"D:\Python project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }
    final_state = {}
    for event in import_app.stream(graph_state):
        for key, value in event.items():
            print(f"当前执行的节点：{key}")
            print(f"当前正在执行的节点状态：{value}")
            final_state = value
    return final_state

if __name__ == '__main__':
    setup_logging()
    final_state = run_import_graph()
    print(json.dumps(final_state, ensure_ascii=False,indent=4))
    # 整个状态图
    print(import_app.get_graph().print_ascii())