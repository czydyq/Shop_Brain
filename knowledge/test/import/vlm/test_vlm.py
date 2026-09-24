from openai import OpenAI
import os

# 初始化OpenAI客户端
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url="https://[workspace-id].cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen3-vl-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg"
                    },
                },
                {"type": "text", "text": "这道题怎么解答？"},
            ],
        },
    ],
    stream=True,
    # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
    extra_body={
        'enable_thinking': True,
        "thinking_budget": 81920},

    # 解除以下注释会在最后一个chunk返回Token使用量
    # stream_options={
    #     "include_usage": True
    # }
)

for chunk in completion:
    # 如果chunk.choices为空，则打印usage
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content

# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(reasoning_content)
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(answer_content)