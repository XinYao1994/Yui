import sys
import os

import torch
import pandas as pd
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

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
        squad = load_dataset("squad")
        print(squad["train"][0])
        def preprocess_function(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = self.tokenizer(
                questions,
                examples["context"],
                max_length=384,
                truncation="only_second",
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs.pop("offset_mapping")
            answers = examples["answers"]
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                answer = answers[i]
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label it (0, 0)
                if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs
        self.datasets = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
        # print(self.datasets["train"][0])
        # print(len(self.datasets["train"][0]["input_ids"]))
        # print(sum(self.datasets["train"][0]["attention_mask"]))


    def close_encodings(self):
        self.X = self.faq["Questions"]
        self.Y = self.faq["Answers"]
        print(self.X[0], "\n\n", self.Y[0])
        print(self.tokenizer(self.X[0]), "\n\n", self.tokenizer(self.Y[0]))

    def load_data(self):
        self.path = self.paths[self.type]
        self.faq = pd.read_csv(self.path, encoding='utf-8')
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # self.dataset
        self.data_encodings[self.type]()
        self.data_collator = DefaultDataCollator()

    def load_model(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    
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
        pass

if __name__ == '__main__' :
    type_ = "close"
    if len(sys.argv) > 1 and sys.argv[1]:
        type_ = sys.argv[1]
    robot = Robot(type_)
    robot.load_data()
    # robot.load_model()
    # robot.train()
    # from datasets import load_dataset
    # ubuntu_dialogs_corpus = load_dataset("ubuntu_dialogs_corpus", 'train')
    # print(ubuntu_dialogs_corpus[0])
    