from datetime import datetime
import logging
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import torch

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

model_name = "./translateEnSk/saved/"

output_layer = 'loss:0'
input_node = 'Placeholder:0'

tokenizer = None
model = None


def _initialize():
    nltk.download('punkt')

    global tokenizer
    global model
    if tokenizer is None or model is None:
        
        # model_name = "Helsinki-NLP/opus-mt-en-sk"
        # need_save = True
        # if os.path.isdir("./translateEnSk/saved/"):
        #     model_name = "./translateEnSk/saved/"
        #     need_save = False

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # if need_save:
        #     model.save_pretrained("./translateEnSk/saved/")
        #     tokenizer.save_pretrained("./translateEnSk/saved/")

        

        # dynamic quantization for faster CPU inference
        model.to('cpu')
        # torch.backends.quantized.engine = 'qnnpack'
        torch.backends.quantized.engine = 'fbgemm'
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)


def _log_msg(msg):
    logging.info("{}: {}".format(datetime.now(),msg))

def preprocess(text):
    return nltk.sent_tokenize(text)


def translate(text: str):

    _initialize()

    sentences = preprocess(text=text)

    print(sentences)

    tok = tokenizer(sentences, return_tensors="pt",padding=True)
    translated = model.generate(**tok)

    translated = " ".join([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    print(translated)

    response = {
                'created': datetime.utcnow().isoformat(),
                'translations': translated 
            }

    print(response)

    _log_msg("Results: " + str(response))
    return response


if __name__ == "__main__":

    translate("My name is Sarah and I live in London. It is a very nice city.")