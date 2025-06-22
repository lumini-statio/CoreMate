import polars as pl
import yaml
from chatterbot.trainers import ListTrainer


def train_csv(bot, path, file_type: str):
    
    if file_type == "parquet":
        df = pl.read_parquet(path)
        csv_path = path.replace(".parquet", ".csv")
        df = df.write_csv(csv_path)
        df = pl.read_csv(csv_path)
    else:
        df = pl.read_csv(path)

    if "answer_score" in df.columns:
        df = (
            df.sort("answer_score", descending=True)
            .group_by("question")
            .agg(pl.col("answer").first())
        )
    elif "Referencia" in df.columns:
        df = (
            df.sort("answer_score", descending=True)
            .group_by("question")
            .agg(pl.col("answer").first())
        )
    else:
        df = df.unique(subset=["question"])

    pares = []
    for row in df.iter_rows(named=True):
        pares.append(str(row["question"]).strip())
        pares.append(str(row["answer"]).strip())

    ListTrainer(bot).train(pares)

