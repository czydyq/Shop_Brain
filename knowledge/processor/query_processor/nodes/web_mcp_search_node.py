import asyncio
import json
from json import JSONDecodeError
from typing import Tuple, List, Dict, Any,Union

import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError


class WebMcpSearchNode(BaseNode):
    name = "web_mcp_search_node"

    # search_pro 只接受 query 一个参数，返回条数固定约10条，这里按原意只取前3条
    _WEB_SEARCH_MAX_RESULTS = 3

    def process(self, state: QueryGraphState) ->Union[QueryGraphState,Dict[str, Any]] :

        # 1. 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 2. 定义并且执行mcp的调用
        # 调用方调用一个async修饰的方法，有且只有两种方式：方式一：继续添加await  方式二：将这个方法放到异步环境中(调用方是同步)
        web_search_results = asyncio.run(self._execute_mcp_server(rewritten_query))

        # 3. 判断
        if not web_search_results:
            return state

        # 4. 返回
        return {"web_search_docs": web_search_results}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        # 1. 用户的问题（LLM重写后的）
        rewritten_query = state.get('rewritten_query')

        # 2. 获取商品名列表
        item_names = state.get('item_names')

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewritten_query, item_names

    async def _execute_mcp_server(self, rewritten_query: str) -> List[Dict[str, Any]]:
        """
        执行MCP服务
        注意：一个MCP服务下可能有多个工具（工具:就是函数）
        Args:
            rewritten_query:

        Returns:

        """

        # 1. 定义MCP客户端(StreamableHttp方式)
        #    注意：mcp 2.x 的HTTP客户端是 httpx2，不是 httpx
        async with httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {self.config.openai_api_key}"},
                timeout=60,  # 超时时间
        ) as http_client:
            async with streamable_http_client(
                    self.config.mcp_dashscope_base_url,
                    http_client=http_client,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:

                    # 2. 握手：必须显式调用initialize
                    #    mcp 2.x 默认先发 server/discover 做协议代际探测，而DashScope的MCP服务
                    #    只支持经典握手、对 server/discover 直接返回500，会导致连接失败
                    await session.initialize()

                    # 3. 调用工具
                    #    search_pro 只支持query参数，多传count会返回PARAM_INVALID
                    web_search_result = await session.call_tool(
                        "search_pro", arguments={"query": rewritten_query})

                    # 4. 工具自身报错（HTTP状态码是200，错误在返回报文里）
                    if web_search_result.is_error:
                        self.logger.error(f"web_search检索失败 失败信息：{self._get_first_text(web_search_result)}")
                        return []

                    # 5. 解析数据
                    # 5.1 获取文本内容块对象的内容
                    text_content_text = self._get_first_text(web_search_result)
                    if not text_content_text:
                        return []

                    # 5.2 反序列化
                    try:
                        text_content_obj: Dict[str, Any] = json.loads(text_content_text)
                    except JSONDecodeError as e:
                        self.logger.error(f"web_search检索失败 失败信息：{e.msg} 失败的内容:{e.doc} 失败的位置：{e.pos}")
                        return []

                    # 5.3 获取真正的网页内容
                    pages = text_content_obj.get('pages', [])
                    if not pages:
                        return []

                    # 5.4 遍历
                    web_search_results = []
                    for page in pages[:self._WEB_SEARCH_MAX_RESULTS]:
                        web_search_results.append({
                            "snippet": page.get('snippet', '').strip(),
                            "title": page.get('title', '').strip(),
                            "url": page.get('url', '').strip(),
                        })
                    return web_search_results

    @staticmethod
    def _get_first_text(tool_result) -> str:
        """取出工具返回的第一个文本内容块的内容"""
        if not tool_result.content:
            return ""
        return tool_result.content[0].text or ""


if __name__ == '__main__':
    web_search_node = WebMcpSearchNode()

    mock_state = {
        "rewritten_query": "今天的天气怎么样？",
        "item_names": ["RS-12 数字万用表"],
    }

    search_result = web_search_node.process(mock_state)
    search_docs = search_result.get('web_search_docs', []) if isinstance(search_result, dict) else []

    print(f"检索到 {len(search_docs)} 条网络结果")
    for index, doc in enumerate(search_docs, 1):
        print(f"  {index}. {doc['title']}  {doc['url']}")
        print(f"     {doc['snippet'][:80]}...")
