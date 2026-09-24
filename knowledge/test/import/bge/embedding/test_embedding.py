from pymilvus.model.hybrid import BGEM3EmbeddingFunction

beg_m3_ef = BGEM3EmbeddingFunction(
    model_name=r"D:\ai_models\modelscope_cache\models\bge-m3",
    device='gpu',
    use_fp16=True
)

print(beg_m3_ef)