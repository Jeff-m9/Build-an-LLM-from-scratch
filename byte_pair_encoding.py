import tiktoken

enc = tiktoken.get_encoding("o200k_base")

text = "Damnnnn!!! <|endoftext|> Hallellujah "
tokens = enc.encode(text, allowed_special={"<|endoftext|>"})

print(tokens)

print(enc.decode(tokens))
