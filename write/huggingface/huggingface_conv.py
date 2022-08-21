import sys
import os

import torch
import pandas as pd
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelWithLMHead, TrainingArguments, Trainer

# print(torch.cuda.is_available())

class Robot:
    def __init__(self, type_) -> None:
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        self.paths = {
            "close":self.base_path+"/data/close/Mental_Health_FAQ.csv",
            "open":self.base_path+"/data/open/Ubuntu-dialogue-corpus/dialogueText.csv"}
        self.data_encodings = {
            "close":self.close_encodings,
            "open":self.open_encodings
        }
        self.type = type_
    
    def open_encodings(self):
        from datasets import load_dataset
        self.dataset = load_dataset("conv_ai_3")
        print(self.dataset["train"][0])
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

    def close_encodings(self):
        self.X = self.faq["Questions"]
        self.Y = self.faq["Answers"]
        print(self.X[0], "\n\n", self.Y[0])
        print(self.tokenizer(self.X[0]), "\n\n", self.tokenizer(self.Y[0]))

    def load_data(self):
        self.path = self.paths[self.type]
        # self.dataset
        self.data_encodings[self.type]()
        self.data_collator = DefaultDataCollator()

    def load_model(self):
        self.model = AutoModelWithLMHead.from_pretrained('microsoft/DialoGPT-medium')
    
    def train(self):
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        self.trainer.train()
    
    def inference(self, question):
        pass

    def test(self):
        # Let's chat for 5 lines
        for step in range(100):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = self.tokenizer.encode(input(">> User:") + self.tokenizer.eos_token, return_tensors='pt')
            # print(new_user_input_ids)

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens, 
            chat_history_ids = self.model.generate(
                bot_input_ids, max_length=500,
                pad_token_id=self.tokenizer.eos_token_id,  
                no_repeat_ngram_size=3,       
                do_sample=True, 
                top_k=100, 
                top_p=0.7,
                temperature = 0.8
            )
    
            # pretty print last ouput tokens from bot
            print("AI: {}".format(self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


if __name__ == '__main__' :
    type_ = "open"
    if len(sys.argv) > 1 and sys.argv[1]:
        type_ = sys.argv[1]
    robot = Robot(type_)
    robot.load_data()
    robot.load_model()
    robot.test()
    # robot.train()
    # from datasets import load_dataset
    # ubuntu_dialogs_corpus = load_dataset("ubuntu_dialogs_corpus", 'train')
    # print(ubuntu_dialogs_corpus[0])
    