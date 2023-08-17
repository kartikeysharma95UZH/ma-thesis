import datetime
import os
import argparse
import numpy as np
import torch
import datasets
import pandas as pd
import editdistance
import transformers
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from accelerate import Accelerator
from sklearn.model_selection import StratifiedShuffleSplit


class ModelTrainer_notebook:
    def __init__(
        self,
        data_path,
        model_checkpoint,
        model,
        tokenizer,
        metric,
        prefix,
        epoch,
        batch,
        learning_rate,
        weight_decay,
        softskill_flag,
    ):
        self.model_checkpoint = model_checkpoint
        # self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = model
        self.tokenizer = tokenizer
        self.data_path = data_path
        # self.train_dataset = train_data
        # self.val_dataset = validation_data
        # self.test_dataset = test_data
        self.softskill_flag = softskill_flag
        if self.softskill_flag == 0:
            self.train_dataset, self.val_dataset, self.test_dataset = self.load_data()
        else:
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = self.load_data_softskill()
        self.metric = load_metric(metric)
        self.prefix = prefix
        self.max_input_length = 512
        self.max_target_length = 512
        self.batch_size = batch
        self.num_epochs = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_name = self.model_checkpoint.split("/")[-1]
        self.run_name = f"{self.model_name}-epoch{self.num_epochs}"
        self.output_dir = f"/home/user/ksharma/ks_thesis/saved_models/automated_model_saves/{self.run_name}"
        self.raw_datasets = self.get_raw_datasets()
        self.tokenized_datasets = self.raw_datasets.map(
            self.preprocess_function, batched=True
        )
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(
            ["input", "output"]
        )
        # self.tokenized_datasets = self.tokenized_datasets.remove_columns(["input", "output", "__index_level_0__"])

    def load_data(self):
        data = pd.read_csv(self.data_path)
        valid_rows = data[data["valids"] == 1.0]
        valid_data_df = valid_rows.copy()
        fine_tuning_data = valid_data_df[["input", "output"]]
        train_data = fine_tuning_data[:400]
        test_data = fine_tuning_data.loc[415:520]
        validation_data, test_data = train_test_split(
            test_data, test_size=0.5, random_state=42
        )
        return train_data, validation_data, test_data

    def load_data_softskill(self):
        # data = pd.read_csv(self.data_path)
        # valid_rows = data[data['valids'] == 0]
        # valid_data_df = valid_rows.copy()
        # fine_tuning_data = valid_data_df[['input', 'output']]
        # type_data = valid_data_df['Problem type']

        # train_ratio = 0.8
        # val_ratio = 0.1
        # test_ratio = 0.1

        # sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio+test_ratio, random_state=42)

        # # Split the data into train, test, and validation sets while maintaining the type distribution
        # for train_index, test_val_index in sss.split(fine_tuning_data, type_data):
        #     # Reset the indices of input_data and type_data
        #     reset_input_data = fine_tuning_data.reset_index(drop=True)
        #     reset_type_data = type_data.reset_index(drop=True)

        #     # Reset the indices of reset_input_data and reset_type_data
        #     reset_input_data = reset_input_data.iloc[test_val_index].reset_index(drop=True)
        #     reset_type_data = reset_type_data.iloc[test_val_index].reset_index(drop=True)

        #     # Perform a second stratified shuffle split for test and validation sets
        #     sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
        #     for test_index, val_index in sss_inner.split(reset_input_data, reset_type_data):
        #         # Get the indices for test and validation sets
        #         test_indices = test_val_index[test_index]
        #         val_indices = test_val_index[val_index]
        #         train_indices = train_index

        # # Reset the indices of the train, test, and validation sets
        # train_data = fine_tuning_data.iloc[train_indices].reset_index(drop=True)
        # test_data = fine_tuning_data.iloc[test_indices].reset_index(drop=True)
        # validation_data = fine_tuning_data.iloc[val_indices].reset_index(drop=True)

        data = pd.read_csv(self.data_path)
        valid_rows = data[data["valids"] == 0]
        valid_data_df = valid_rows.copy()
        new_dataset = valid_data_df[valid_data_df["Problem type"] == "OWA"]
        fine_tuning_data = new_dataset[["input", "output"]]
        train_data, test_data = train_test_split(
            fine_tuning_data, test_size=0.2, random_state=42
        )
        validation_data, test_data = train_test_split(
            test_data, test_size=0.5, random_state=42
        )

        return train_data, validation_data, test_data

    def get_raw_datasets(self):
        # Convert the dataframes to datasets
        train_dataset = datasets.Dataset.from_pandas(self.train_dataset)
        val_dataset = datasets.Dataset.from_pandas(self.val_dataset)
        test_dataset = datasets.Dataset.from_pandas(self.test_dataset)

        # Create the DatasetDict
        raw_datasets = datasets.DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )
        return raw_datasets

    def preprocess_function(self, data):
        inputs = [self.prefix + noun_ellipsis for noun_ellipsis in data["input"]]
        model_inputs = self.tokenizer(inputs, truncation=True, padding=True)

        # Setup the tokenizer for targets
        labels = self.tokenizer(
            data["output"], truncation=True, text_pair=data["input"], padding=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        print(labels.dtype)
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=False
        )
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        print(
            f"################################################ {self.run_name} ################################################"
        )
        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            run_name=self.run_name,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=self.learning_rate,
            logging_steps=10,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            save_total_limit=3,
            num_train_epochs=self.num_epochs,
            predict_with_generate=True,
            fp16=False,
            push_to_hub=False,
            report_to="wandb",
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        accelerator = Accelerator()
        self.tokenized_datasets, trainer = accelerator.prepare(
            self.tokenized_datasets, trainer
        )

        trainer.train()

    def save(self):
        # Generate the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # self.model.save_pretrained(f'/home/user/ksharma/ks_thesis/saved_models/automated_model_saves/{self.run_name}_{timestamp}')
        # self.tokenizer.save_pretrained(f'/home/user/ksharma/ks_thesis/saved_models/automated_model_saves/{self.run_name}_{timestamp}')

    def quick_test(self, input_text):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(device)
        # Tokenize the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(device)

        # Generate the output text
        output_ids = self.model.generate(input_ids)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    def test(self):
        # Generate the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        predicted_values = []
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        for text in self.test_dataset["input"]:
            # inputs = ["transform: " + text]
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            input_ids = input_ids.to(device)
            # Generate the output text
            output_ids = self.model.generate(input_ids)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predicted_values.append(output_text)

        testing_df = self.test_dataset.copy()
        testing_df["prediction"] = predicted_values
        testing_df["result"] = (
            testing_df["output"] == testing_df["prediction"]
        ).astype(int)
        testing_df["Levenshtein_dist"] = testing_df.apply(
            lambda row: editdistance.eval(row["output"], row["prediction"]), axis=1
        )
        accuracy = (
            len(testing_df[testing_df["Levenshtein_dist"] < 30]) * 100 / len(testing_df)
        )
        testing_df.to_csv(
            f"/home/user/ksharma/ks_thesis/saved_data_files/{self.run_name}_{timestamp}.csv",
            index=True,
        )
        return testing_df, accuracy

    def test_some_model(self, model, tokenizer):
        predicted_values = []
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)
        for text in self.test_dataset["input"]:
            # inputs = ["transform: " + text]
            input_ids = tokenizer.encode(text, return_tensors="pt")
            input_ids = input_ids.to(device)
            # Generate the output text
            output_ids = model.generate(input_ids)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predicted_values.append(output_text)

        testing_df = self.test_dataset.copy()
        testing_df["prediction"] = predicted_values
        testing_df["result"] = (
            testing_df["output"] == testing_df["prediction"]
        ).astype(int)
        testing_df["Levenshtein_dist"] = testing_df.apply(
            lambda row: editdistance.eval(row["output"], row["prediction"]), axis=1
        )
        accuracy = (
            len(testing_df[testing_df["Levenshtein_dist"] < 3]) * 100 / len(testing_df)
        )
        return testing_df, accuracy


def main(
    model_checkpoint,
    data_path,
    metric,
    prefix,
    epoch,
    batch,
    learning_rate,
    weight_decay,
    softskill_flag,
):

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_checkpoint, model_max_length=512
    )
    # Create instance of class
    my_instance = ModelTrainer_notebook(
        data_path=data_path,
        model_checkpoint=model_checkpoint,
        model=model,
        tokenizer=tokenizer,
        metric=metric,
        prefix=prefix,
        epoch=epoch,
        batch=batch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        softskill_flag=softskill_flag,
    )
    # Perform training and evaluation
    my_instance.train()
    my_instance.save()
    my_instance.test()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, help="Model checkpoint name")
    parser.add_argument("--data_path", type=str, help="Path to data")
    parser.add_argument("--metric", type=str, help="Metric")
    parser.add_argument("--prefix", type=str, help="Prefix")
    parser.add_argument("--epoch", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--softskill_flag", type=int, help="Softskill flag")
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(
        model_checkpoint=args.model_checkpoint,
        data_path=args.data_path,
        metric=args.metric,
        prefix=args.prefix,
        epoch=args.epoch,
        batch=args.batch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        softskill_flag=args.softskill_flag,
    )
    # from my_class import ModelTrainer_1
