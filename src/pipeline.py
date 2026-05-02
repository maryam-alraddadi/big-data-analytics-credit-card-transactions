"""Spark data preparation pipeline for credit-card transactions."""

from __future__ import annotations

import argparse
import re
from typing import Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def create_spark_session(app_name: str = "CreditCardDataPreparation") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.sql.shuffle.partitions", "50")
        # JDK 21+ compatibility guard for some local macOS setups.
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalize_column_names(df: DataFrame) -> DataFrame:
    for old_name in df.columns:
        new_name = _normalize_name(old_name)
        if old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)
    return df


def standardize_schema(df: DataFrame) -> DataFrame:
    alias_map: Dict[str, str] = {
        "trans_date_trans_time": "transaction_ts",
        "timestamp": "transaction_ts",
        "datetime": "transaction_ts",
        "date": "transaction_ts",
        "time": "time",
        "class": "is_fraud",
        "fraud": "is_fraud",
        "label": "is_fraud",
        "amt": "amount",
        "trans_num": "transaction_id",
        "cc_num": "card_id",
        "merchant_name": "merchant",
    }
    for source, target in alias_map.items():
        if source in df.columns and target not in df.columns:
            df = df.withColumnRenamed(source, target)

    if "transaction_ts" not in df.columns and "time" in df.columns:
        # Handle datasets with only relative seconds since start.
        df = df.withColumn(
            "transaction_ts",
            F.to_timestamp(F.from_unixtime(F.col("time").cast("double"))),
        )

    if "transaction_ts" in df.columns:
        df = df.withColumn(
            "transaction_ts",
            F.coalesce(
                F.to_timestamp(F.col("transaction_ts")),
                F.to_timestamp(F.col("transaction_ts"), "yyyy-MM-dd HH:mm:ss"),
                F.to_timestamp(F.col("transaction_ts"), "MM/dd/yyyy HH:mm"),
                F.to_timestamp(F.col("transaction_ts"), "dd-MM-yyyy HH:mm:ss"),
            ),
        )

    if "amount" in df.columns:
        df = df.withColumn("amount", F.col("amount").cast("double"))

    if "is_fraud" in df.columns:
        df = df.withColumn("is_fraud", F.col("is_fraud").cast("int"))

    if "transaction_id" not in df.columns:
        concat_cols = [F.coalesce(F.col(c).cast("string"), F.lit("")) for c in sorted(df.columns)]
        df = df.withColumn("transaction_id", F.sha2(F.concat_ws("||", *concat_cols), 256))

    ordered_cols = ["transaction_id"]
    for preferred in ["transaction_ts", "card_id", "merchant", "amount", "is_fraud"]:
        if preferred in df.columns and preferred not in ordered_cols:
            ordered_cols.append(preferred)
    for col_name in df.columns:
        if col_name not in ordered_cols:
            ordered_cols.append(col_name)
    return df.select(*ordered_cols)


def clean_transactions(df: DataFrame) -> DataFrame:
    if "amount" in df.columns:
        df = df.filter(F.col("amount").isNotNull()).filter(F.col("amount") >= F.lit(0.0))

    if "is_fraud" in df.columns:
        df = df.filter(F.col("is_fraud").isin(0, 1))

    df = df.dropDuplicates(["transaction_id"]) if "transaction_id" in df.columns else df.dropDuplicates()
    return df


def ingest_and_prepare(spark: SparkSession, input_path: str) -> DataFrame:
    raw_df = (
        spark.read.format("csv")
        .option("header", True)
        .option("mode", "DROPMALFORMED")
        .option("multiLine", False)
        .load(input_path)
    )
    normalized_df = normalize_column_names(raw_df)
    standardized_df = standardize_schema(normalized_df)
    cleaned_df = clean_transactions(standardized_df)
    return cleaned_df


def write_parquet(df: DataFrame, output_path: str, mode: str = "overwrite") -> None:
    df.write.mode(mode).parquet(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare credit-card transaction data with Spark.")
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--output", required=True, help="Path to output Parquet directory.")
    parser.add_argument(
        "--mode",
        default="overwrite",
        choices=["overwrite", "append", "error", "ignore"],
        help="Parquet write mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = create_spark_session()
    try:
        prepared_df = ingest_and_prepare(spark, args.input)
        write_parquet(prepared_df, args.output, mode=args.mode)
        print(f"Prepared rows: {prepared_df.count()}")
        print(f"Wrote Parquet to: {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
