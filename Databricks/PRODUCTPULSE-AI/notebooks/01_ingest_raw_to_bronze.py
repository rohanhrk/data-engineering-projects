# Databricks notebook source
# ProductPulse AI - 01 ingest raw files to bronze Delta tables
#
# Purpose:
# Load structured retail feeds and unstructured PDF documents from the Unity
# Catalog volume into bronze Delta tables. The PDF path is intentionally real:
# internal manuals, policies, SOPs, and QA guides live as PDF files, while a
# small manifest supplies business metadata.

from io import BytesIO

from pypdf import PdfReader
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

dbutils.widgets.text("catalog", "adidas_retail_ai", "Unity Catalog catalog")
dbutils.widgets.text("schema", "bronze", "Unity Catalog schema")

# The raw files are expected under the Unity Catalog volume created by notebook
# 00. The local folder `data/raw` is uploaded there by the runbook/CLI step.
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
raw_root = f"/Volumes/{catalog}/{schema}/raw_data/raw"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

def extract_pdf_text(content: bytes) -> str:
    """Extract readable text from one PDF binary payload.

    Databricks reads PDFs as binary files. This UDF converts each file's bytes
    into text so the next notebook can chunk it for RAG. The function returns an
    empty string rather than failing the whole pipeline if a single PDF page is
    unreadable.
    """
    try:
        reader = PdfReader(BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        full_text = "\n\n".join(page.strip() for page in pages if page.strip())
        return full_text
    except Exception:
        return ""

# Register as PySpark UDF
extract_pdf_text_udf = F.udf(extract_pdf_text, StringType())

# COMMAND ----------

# Product master is a structured dimension feed.
# It is loaded from CSV because product attributes are maintained as one compact
# reference file in this demo.
products_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{raw_root}/products.csv")
    .withColumn("launch_date", F.to_date("launch_date"))
    .withColumn("ingestion_ts", F.current_timestamp())
)

# Orders, returns, and inventory are daily fact/snapshot feeds, so they are
# stored in partition-style folders just like reviews and support tickets.
# Spark reads every date partition under `orders/*/*.jsonl`, which keeps the
# code simple while still giving learners realistic date-partitioned raw files.
orders_df = (
    spark.read
    .json(f"{raw_root}/orders/*/*.jsonl")
    .withColumn("event_date", F.to_date("event_date"))
    .withColumn("quantity", F.col("quantity").cast("int"))
    .withColumn("unit_price_usd", F.col("unit_price_usd").cast("double"))
    .withColumn("ingestion_ts", F.current_timestamp())
)

# Returns arrive as event records rather than daily aggregates. Keeping every
# return event in Bronze lets Gold calculate both counts and refund impact later.
returns_df = (
    spark.read
    .json(f"{raw_root}/returns/*/*.jsonl")
    .withColumn("event_date", F.to_date("event_date"))
    .withColumn("refund_amount_usd", F.col("refund_amount_usd").cast("double"))
    .withColumn("is_exchange", F.col("is_exchange").cast("boolean"))
    .withColumn("ingestion_ts", F.current_timestamp())
)

# Inventory is a snapshot feed. The same product appears once per warehouse per
# date, and Gold later rolls those rows up to product/day availability.
inventory_snapshots_df = (
    spark.read
    .json(f"{raw_root}/inventory_snapshots/*/*.jsonl")
    .withColumn("snapshot_date", F.to_date("snapshot_date"))
    .withColumn("on_hand_units", F.col("on_hand_units").cast("int"))
    .withColumn("reserved_units", F.col("reserved_units").cast("int"))
    .withColumn("reorder_point", F.col("reorder_point").cast("int"))
    .withColumn("ingestion_ts", F.current_timestamp())
)

# Customer reviews and support tickets are semi-structured event feeds.
# They are loaded to Bronze first because their structured fields support
# metrics, while their text fields later become RAG evidence.
reviews_df = (
    spark.read
    .json(f"{raw_root}/reviews/*/*.jsonl")
    .withColumn("event_date", F.to_date("event_date"))
    .withColumn("rating", F.col("rating").cast("int"))
    .withColumn("verified_purchase", F.col("verified_purchase").cast("boolean"))
    .withColumn("ingestion_ts", F.current_timestamp())
)

# Tickets contain operational support language that is often more precise than
# reviews: requested outcome, priority, status, and customer channel.
tickets_df = (
    spark.read
    .json(f"{raw_root}/support_tickets/*/*.jsonl")
    .withColumn("event_date", F.to_date("event_date"))
    .withColumn("ingestion_ts", F.current_timestamp())
)

# PDF manifest provides business metadata. The PDF binary reader provides file
# content. Joining both gives a governed bronze document table.
# The manifest is important because a PDF file name alone is not enough for RAG
# filtering. We need doc_id, source_type, product_id, title, and effective date.
pdf_manifest_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{raw_root}/knowledge_pdf_manifest.csv")
    .withColumn("effective_date", F.to_date("effective_date"))
    .withColumn("page_count", F.col("page_count").cast("int"))
)

# Databricks `binaryFile` lets Spark read each PDF as one row with bytes in the
# `content` column. The UDF extracts text so downstream notebooks do not need to
# touch PDF binaries.
pdf_files_df = (
    spark.read
    .format("binaryFile")
    .load(f"{raw_root}/knowledge_pdfs/*.pdf")
    .withColumn("file_name", F.element_at(F.split(F.col("path"), "/"), -1))
    .withColumn("body", extract_pdf_text_udf(F.col("content")))
    .select("file_name", "body")
)

# Joining manifest + extracted text creates the Bronze document table. If a PDF
# cannot be extracted, the body will be blank and notebook 02 will skip it.
knowledge_docs_df = (
    pdf_manifest_df
    .join(
        pdf_files_df,
        "file_name",
        "left"
    ).select(
        "doc_id",
        "source_type",
        F.when(
            F.col("product_id") == "",
            F.lit("None")
        ).otherwise(
            F.col("product_id")
        ).alias("product_id"),
        "title",
        "body",
        "effective_date",
        "file_name",
        "page_count",
        F.current_timestamp().alias("ingestion_ts"),
    )
)

# COMMAND ----------

# This demo uses overwrite mode for clarity and repeatability. In a production
# incremental pipeline, each feed would usually merge by business key/date and
# use Auto Loader or another ingestion mechanism for new files.
products_df.write.mode("overwrite").saveAsTable(f"{catalog}.bronze.products")
orders_df.write.mode("overwrite").partitionBy("event_date").saveAsTable(f"{catalog}.bronze.orders")
returns_df.write.mode("overwrite").partitionBy("event_date").saveAsTable(f"{catalog}.bronze.returns")
inventory_snapshots_df.write.mode("overwrite").partitionBy("snapshot_date").saveAsTable(f"{catalog}.bronze.inventory_snapshots")
reviews_df.write.mode("overwrite").partitionBy("event_date").saveAsTable(f"{catalog}.bronze.reviews")
tickets_df.write.mode("overwrite").partitionBy("event_date").saveAsTable(f"{catalog}.bronze.support_tickets")
knowledge_docs_df.write.mode("overwrite").saveAsTable(f"{catalog}.bronze.knowledge_docs")

print("Bronze load complete")
print(f"products={products_df.count()}")
print(f"orders={orders_df.count()}")
print(f"returns={returns_df.count()}")
print(f"inventory_snapshots={inventory_snapshots_df.count()}")
print(f"reviews={reviews_df.count()}")
print(f"tickets={tickets_df.count()}")
print(f"pdf_docs={knowledge_docs_df.count()}")
