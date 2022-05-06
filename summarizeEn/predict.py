from datetime import datetime
import logging
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import torch

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

# model_name ="cnicu/t5-small-booksum"
# model = AutoModelForSeq2SeqLM.from_pretrained("Miranda/t5-small-train")
# model_name ="sshleifer/distilbart-xsum-12-1"
model_name ="summarizeEn/saved"
# model_name = "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization"

output_layer = 'loss:0'
input_node = 'Placeholder:0'

tokenizer = None
model = None


def _initialize():
    nltk.download('punkt')

    global tokenizer
    global model
    if tokenizer is None or model is None:
        
        _log_msg("Initializing model and tokenizer.")

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        
        # model.save_pretrained("./summarizeEn/saved/")
        # tokenizer.save_pretrained("./summarizeEn/saved/")
            

        
        _log_msg("Dynamic quantization of model.")
        # dynamic quantization for faster CPU inference
        model.to('cpu')
        # torch.backends.quantized.engine = 'qnnpack' # ARM
        torch.backends.quantized.engine = 'fbgemm' # x86
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)

        _log_msg("Model ready!")


def _log_msg(msg, debug=False):
    message = "{}: {}".format(datetime.now(),msg)

    if debug:
       logging.debug(message) 
       return

    logging.info(message)

def preprocess(text):

    paragraph_list = [text]
    # if len(text) > model.config.encoder.max_position_embeddings:
    if len(text) > 512:
        paragraph_list = []
        
        paragraph = ""
        sentences = nltk.sent_tokenize(text)
        
        for i in range(len(sentences)):

            new_len = len(paragraph.split()) + len(sentences[i].split())

            if new_len < 512:
                paragraph += " " + sentences[i]
            else:
                paragraph_list.append(paragraph)
                paragraph = ""
        

    return paragraph_list


def translate(text: str):

    _initialize()

    _log_msg("Text length:" + str(len(text)), True)
    
    print(model.config)

    sentences = preprocess(text=text)
    # sentences = text

    
    # print(sentences)

    output = []
    for sentence in sentences:
        tok = tokenizer(sentence, return_tensors="pt",padding=True)
        translated  = model.generate(**tok)

        res = " ".join([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
        output.append(res)
    # print(translated)
    
    output = " ".join(output)

    _log_msg("Summarized length: " + str(len(output)), True)
    # print("Translated length",len(output))

    response = {
                'created': datetime.utcnow().isoformat(),
                'output': output 
            }

    print(response)

    _log_msg("Results: " + str(response), True)
    return response


if __name__ == "__main__":

    text = """In the beginning God created the heavens and the earth.
Now the earth was formless and empty. Darkness was on the surface of the deep. God's Spirit was hovering over the surface of the waters.
God said, "Let there be light," and there was light.
God saw the light, and saw that it was good. God divided the light from the darkness.
God called the light "day," and the darkness he called "night." There was evening and there was morning, one day.
God said, "Let there be an expanse in the middle of the waters, and let it divide the waters from the waters."
God made the expanse, and divided the waters which were under the expanse from the waters which were above the expanse; and it was so.
God called the expanse "sky." There was evening and there was morning, a second day.
God said, "Let the waters under the sky be gathered together to one place, and let the dry land appear"; and it was so.
God called the dry land "earth," and the gathering together of the waters he called "seas." God saw that it was good.
God said, "Let the earth yield grass, herbs yielding seed, and fruit trees bearing fruit after their kind, with its seed in it, on the earth"; and it was so.
The earth yielded grass, herbs yielding seed after their kind, and trees bearing fruit, with its seed in it, after their kind; and God saw that it was good.
There was evening and there was morning, a third day.
God said, "Let there be lights in the expanse of sky to divide the day from the night; and let them be for signs, and for seasons, and for days and years;
and let them be for lights in the expanse of sky to give light on the earth"; and it was so.
God made the two great lights: the greater light to rule the day, and the lesser light to rule the night. He also made the stars.
God set them in the expanse of sky to give light to the earth,
and to rule over the day and over the night, and to divide the light from the darkness. God saw that it was good.
There was evening and there was morning, a fourth day.
God said, "Let the waters swarm with swarms of living creatures, and let birds fly above the earth in the open expanse of sky."
God created the large sea creatures, and every living creature that moves, with which the waters swarmed, after their kind, and every winged bird after its kind. God saw that it was good.
God blessed them, saying, "Be fruitful, and multiply, and fill the waters in the seas, and let birds multiply on the earth."
There was evening and there was morning, a fifth day.
God said, "Let the earth produce living creatures after their kind, livestock, creeping things, and animals of the earth after their kind"; and it was so.
God made the animals of the earth after their kind, and the livestock after their kind, and everything that creeps on the ground after its kind. God saw that it was good.
God said, "Let us make man in our image, after our likeness: and let them have dominion over the fish of the sea, and over the birds of the sky, and over the livestock, and over all the earth, and over every creeping thing that creeps on the earth."
God created man in his own image. In God's image he created him; male and female he created them.
God blessed them. God said to them, "Be fruitful, multiply, fill the earth, and subdue it. Have dominion over the fish of the sea, over the birds of the sky, and over every living thing that moves on the earth."
God said, "Behold, I have given you every herb yielding seed, which is on the surface of all the earth, and every tree, which bears fruit yielding seed. It will be your food.
To every animal of the earth, and to every bird of the sky, and to everything that creeps on the earth, in which there is life, I have given every green herb for food"; and it was so.
God saw everything that he had made, and, behold, it was very good. There was evening and there was morning, a sixth day.
The heavens and the earth were finished, and all their vast array.
On the seventh day God finished his work which he had made; and he rested on the seventh day from all his work which he had made.
God blessed the seventh day, and made it holy, because he rested in it from all his work which he had created and made.
This is the history of the generations of the heavens and of the earth when they were created, in the day that Yahweh God made the earth and the heavens.
No plant of the field was yet in the earth, and no herb of the field had yet sprung up; for Yahweh God had not caused it to rain on the earth. There was not a man to till the ground,
but a mist went up from the earth, and watered the whole surface of the ground.
Yahweh God formed man from the dust of the ground, and breathed into his nostrils the breath of life; and man became a living soul."""
    # text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    translate(text)