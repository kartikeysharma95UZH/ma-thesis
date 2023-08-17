import spacy
import de_soski_ner_model
from spacy.language import Language
import logging

log = logging.getLogger(__name__)
import sys
from smart_open import open
import json
import re

import os, glob
from tqdm import tqdm
import pandas as pd
import random

folder_path = "/mnt/home/kartikeysharma/sentence_extraction/new_quarterly_datasets"
# dataset = 'softskill-completion/data/quarterly_data_1April/ads_raw-2014-09.jsonl.bz2'


@Language.component("paragrapher")
def set_sentence_starts(doc):
    for i, token in enumerate(doc):
        if i > 0 and doc[i - 1].text.count("\n") > 1:
            doc[i].is_sent_start = True
            # log.error(f"SENTENCE BOUNDARY: {i}")
    return doc


class MainApplication(object):
    def __init__(self, num, lang):
        # self.args = args
        # self.data = dataset
        self.num = num
        self.lang = lang
        self.nlp = de_soski_ner_model.load()

        senter = self.nlp.create_pipe("sentencizer")
        self.nlp.add_pipe("sentencizer")
        self.nlp.add_pipe("paragrapher", after="transformer")

        # Get the names of all components in the pipeline
        pipeline_names = self.nlp.pipe_names
        # print("Pipeline component names:", pipeline_names)

        # # Print detailed information about each pipeline component
        # for name in pipeline_names:
        #     pipe_component = self.nlp.get_pipe(name)
        #     print(f"\nComponent name: {name}")
        #     print(f"Component type: {type(pipe_component)}")
        #     if hasattr(pipe_component, "labels"):
        #         print(f"Labels: {pipe_component.labels}")
        #     if hasattr(pipe_component, "vocab"):
        #         print(f"Vocab: {pipe_component.vocab}")
        #     if hasattr(pipe_component, "model"):
        #         print(f"Model: {pipe_component.model}")

    def prepare_prompt(self, sentence):
        text = sentence.text
        words = text.split()
        num_tokens = len(words)

        for ent in sentence.ents:
            if ent.label_ == "SoftSkill_C":
                text = text.replace(ent.text, f"<SoftSkill_C>{ent.text}</SoftSkill_C>")
            if ent.label_ == "SoftSkill":
                text = text.replace(ent.text, f"<SoftSkill>{ent.text}</SoftSkill>")
        return text, num_tokens

    def has_softskill_C(self, sentence):
        for token in sentence:
            if token.ent_type_.endswith("_C"):
                return True

    def run(self, JSON_file):
        # log.warning(self.args)

        entity_sent = []
        job_ids_list = []
        year_list = []
        quarter_list = []
        # creation_date_list = []
        text_valid_list = []
        length_list = []
        language_list = []
        text_list = []
        tokens_in_sent_list = []
        pipeline_version_list = []
        actual_sentence_list = []

        with open(JSON_file) as infile:
            json_lines = infile.readlines()

            for i, l in enumerate(tqdm(json_lines)):
                if i > self.num - 1:
                    break
                data = json.loads(l)
                if (
                    data["language"] != self.lang
                    or data["text_valid"] != 1
                    or "adve_text_copy" not in data
                ):
                    continue

                doc = self.nlp(data["adve_text_copy"])

                job_ids = data["adve_iden_adve"]
                year = data["adve_time_year"]
                quarter = data["adve_time_quar"]
                # creation_date = data["creation_date"]
                text_valid = data["text_valid"]
                length = data["length"]
                language = data["language"]
                text = data["adve_text_copy"]
                pipeline_version = data["pipeline_version"]

                if doc.ents:
                    for j, sent in enumerate(doc.sents):
                        if not self.has_softskill_C(sent):
                            continue

                        flag_softskill = 0
                        flag_softskill_C = 0
                        for token in sent:
                            if token.ent_type_ == "SoftSkill":
                                flag_softskill = 1
                            if token.ent_type_ == "SoftSkill_C":
                                flag_softskill_C = 1
                        if flag_softskill == 1 and flag_softskill_C == 1:
                            prompt_sent, num_tokens = self.prepare_prompt(sent)
                            actual_sentence = sent.text

                            actual_sentence_list.append(actual_sentence)
                            tokens_in_sent_list.append(num_tokens)
                            entity_sent.append(prompt_sent)
                            job_ids_list.append(job_ids)
                            year_list.append(year)
                            quarter_list.append(quarter)
                            # creation_date_list.append(creation_date)
                            text_valid_list.append(text_valid)
                            length_list.append(length)
                            language_list.append(language)
                            text_list.append(text)
                            pipeline_version_list.append(pipeline_version)

                            continue

        zipped_list = list(
            zip(
                job_ids_list,
                year_list,
                quarter_list,
                entity_sent,
                actual_sentence_list,
                tokens_in_sent_list,
                text_valid_list,
                length_list,
                language_list,
                pipeline_version_list,
                text_list,
            )
        )
        df = pd.DataFrame(
            zipped_list,
            columns=[
                "job_ids_list",
                "year_list",
                "quarter_list",
                "entity_sent",
                "actual_sentence_list",
                "tokens_in_sent_list",
                "text_valid_list",
                "length_list",
                "language_list",
                "pipeline_version_list",
                "text_list",
            ],
        )
        return df

    def get_all_jsonl_job_ad_files(self, folder_path):
        # folder_path = 'ads_annotated/'
        file_names_jsonl = []
        for filename in glob.glob(os.path.join(folder_path, "*.jsonl.bz2")):
            with open(filename, "r") as f:
                file_names_jsonl.append(filename)

        date_regex = re.compile(r"\d{4}-\d{2}")

        # Extract the dates from the file names and create a list of tuples with the date and file name
        date_file_tuples = [
            (date_regex.search(name).group(0), name) for name in file_names_jsonl
        ]

        # Sort the list of tuples based on the extracted dates
        sorted_tuples = sorted(date_file_tuples)

        # Create a list of sorted file names from the sorted tuples
        sorted_file_names = [tup[1] for tup in sorted_tuples]
        return sorted_file_names

    def get_complete_df(self, num, lang):
        batch_size = 5
        list_file_names = self.get_all_jsonl_job_ad_files(folder_path)
        batches = [
            list_file_names[i : i + batch_size]
            for i in range(0, len(list_file_names), batch_size)
        ]

        for i, batch in enumerate(batches):
            master_GS = []
            for jsonl_file in batch:
                print(jsonl_file)
                child_GS_df = self.run(jsonl_file)

                master_GS.append(child_GS_df)

            master_df = pd.concat(master_GS)
            # Save dataframe as a CSV file
            csv_file_name = f"batch_new_{i+1}.csv"
            master_df.to_csv(os.path.join(folder_path, csv_file_name), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process job ads dataset")
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="an integer for the number of ads to process",
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="a string for the language of the job ads",
    )
    args = parser.parse_args()
    # num = 1
    # lang = 'de'
    # launching application ...
    folder_path = "/mnt/home/kartikeysharma/sentence_extraction/new_quarterly_datasets"
    app = MainApplication(args.num, args.lang)
    app.get_complete_df(args.num, args.lang)
