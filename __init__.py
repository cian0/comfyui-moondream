from .nodes.MoondreamNode import MoondreamNode, LoadMoondreamModel#, LoadTokenizer

NODE_CLASS_MAPPINGS = {
    "Moondream": MoondreamNode,
    "MoondreamLoader": LoadMoondreamModel,
    # "MoondreamTokenizerLoader": LoadTokenizer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Moondream": "Moondream:a tiny vision language model",
    "MoondreamLoader": "Moondream Loader",
    # "MoondreamTokenizerLoader": "Moondream Tokenizer Loader",
}