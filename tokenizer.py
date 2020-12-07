
from nltk import TreebankWordTokenizer

class Tokenizer:

    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()

    def tokenize(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        return tokens


if __name__ == '__main__':
    sentence = 'معارج القدس محمد بن محمد بن محمد الغزالي ابو حامد مقدمه في معاني الالفاظ المترادفه علي النفس وهي اربعه النفس والقلب والروح والعقل اما النفس فتطلق بمعنين احدهما ان يطلق ويراد به المعني الجامع لصفات المذمومه وهي القوي الحيوانيه المضاده لقوي العقليه وهو'
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(sentence)
    print(tokens)