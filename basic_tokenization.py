# import the regular expression module
import re

# read the text that will serve as raw data
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# split the entire text into individual words and symbols known as tokens while removing any whitespace
preprocessed = re.split(r'([,.;:?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token: integer for integer, token in enumerate(all_tokens)}

class simpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: t for t, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.;:?_!"()\']|--|\s)', text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        tokens = [self.str_to_int[s] for s in preprocessed]
        return tokens

    def decode(self, tokens):
        text = " ".join([self.int_to_str[i] for i in tokens])

        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text
    

# tokenizer = simpleTokenizer(vocab)
# text1 = "I am a good boy who is aiming to become an ai engineer or ai infrastructure engineer"
# text2 = "He stood up and laid his hand on my shoulder with a laugh. 'Only the irony of it is that I _am_ still painting--since Grindle's doing it for me! The Strouds stand alone, and happen once--but there's no exterminating our kind of art.'"

# text = " <|endoftext|> ".join((text1,text2))

# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))