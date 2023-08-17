# !/usr/bin/python3
# -*- coding: utf-8 -*-
# """

# """

# __appname__ = "[application name here]"
# __author__  = "AA"
# __version__ = "0.0pre0"
# __license__ = "GNU GPL 3.0 or later"

import logging

log = logging.getLogger(__name__)
import sys
import csv
import re
import random
import nilsimsa

import tsv

random.seed(42)


class MainApplication(object):
    def __init__(self, args):
        self.args = args
        self.infile = args.infile
        self.outfile = args.outfile
        self.logfile = args.logfile
        self.verbose = args.verbose

    def run(self):
        log.warning(self.args)
        self.data = self.read_csv()
        random.shuffle(self.data)
        print(self.data[0], file=sys.stderr)
        self.mark_sample()

        self.write_tsv(self.yield_sample())

    def read_csv(self):
        with open(self.infile, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]
        log.info(f"Read {len(data)} from file {self.infile}")
        return data

    def write_tsv(self, lines):
        with open(self.outfile, "w") as tsvfile:
            fieldnames = [
                "sample-id",
                "actual_sent",
                "processed_sent",
                "GS_sent",
                "token_length",
            ]
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(lines)

    def is_similar(
        self, samples_digests, candidate_digest, candidate_form, threshold=110
    ):
        for k in samples_digests:
            similarity = nilsimsa.compare_digests(
                samples_digests[k], candidate_digest, threshold=threshold
            )
            if similarity > threshold:
                log.debug(f"nilsimsa {k} <=> {candidate_form}: {similarity}")
                return True

        return False

    def mark_sample(self):
        seen = {}
        counter = 0
        excluded_counter = 0
        pattern = r"<SoftSkill>(.*?)</SoftSkill>"
        for i, d in enumerate(self.data):
            # norm_span = re.sub(r"\W+","",d['ellipsis'].lower()) %%%%%%%%%%

            try:
                norm_span = re.search(pattern, d["entity_sent"]).group(1)
            except AttributeError:
                norm_span = ""
            if norm_span == "":
                excluded_counter += 1
                continue
            if d["language_list"] != "de":
                excluded_counter += 1
                continue
            if int(d["tokens_in_sent_list"]) > 25:
                excluded_counter += 1
                continue
            if norm_span in seen:
                d["sampling"] = "multiple"
                excluded_counter += 1
            else:
                candidate_digest = nilsimsa.Nilsimsa(norm_span).hexdigest()
                if self.is_similar(seen, candidate_digest, norm_span, threshold=112):
                    d["sampling"] = "similar"
                    excluded_counter += 1
                    continue
                seen[norm_span] = candidate_digest
                d["sampling"] = True

                counter += 1

            if counter > 2000:
                break

        print(f"Sampled: {counter} ads (while excluding {excluded_counter})")
        log.info(f"Sampled: {counter} ads (while excluding {excluded_counter})")

    def yield_sample(self):
        for d in self.data:
            if d.get("sampling") is True:
                r = {
                    "sample-id": d["job_ids_list"],
                    "actual_sent": d["actual_sentence_list"],
                    "processed_sent": d["entity_sent"],
                    "GS_sent": d["entity_sent"],
                    "token_length": d["tokens_in_sent_list"],
                }
                yield r


if __name__ == "__main__":
    import argparse

    description = ""
    epilog = ""
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "-l", "--logfile", dest="logfile", help="write log to FILE", metavar="FILE"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=4,
        type=int,
        metavar="LEVEL",
        help="set verbosity level: 0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO 4=DEBUG (default %(default)s)",
    )
    parser.add_argument(
        "infile",
        metavar="INPUT",
        help="Input file",
    )
    parser.add_argument(
        "outfile",
        metavar="OUTPUT",
        help="Output file",
    )

    arguments = parser.parse_args()

    log_levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ]
    logging.basicConfig(
        level=log_levels[arguments.verbose],
        format="%(asctime)-15s %(levelname)s: %(message)s",
    )

    MainApplication(arguments).run()
