#!/usr/bin/env python
import argparse
import logging
import os

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info("Creating run in project exercise_5")
    run = wandb.init(project="exercise_5", job_type="process_data")

    logger.info("Getting artifact")
    artifact = run.use_artifact(args.input_artifact)

    logger.info("Read artifact")
    df = pd.read_parquet(artifact.file())

    # Pre-processing
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    logger.info("Dropping missing values")
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']


    outfile = "preprocessed_data.csv"
    df.to_csv(outfile)

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )

    artifact.add_file(outfile)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)