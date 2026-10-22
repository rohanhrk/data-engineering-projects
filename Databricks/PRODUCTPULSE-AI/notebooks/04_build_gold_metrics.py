# Databricks notebook source
# ProductPulse AI - 04 build gold product and issue metrics
#
# Purpose:
# Create metric tables the agent can query without asking the LLM to write SQL.
# The gold layer answers trend/count questions and provides operational context
# around the retrieved textual evidence.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "adidas_retail_ai", "Unity Catalog catalog")
dbutils.widgets.text("schema", "gold", "Unity Catalog schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# Load Bronze tables created by notebook 01. Notice that Gold is built directly
# from Bronze in this project. Silver is reserved for RAG text preparation, not
# for structured metric modeling.
products_df = spark.table(f"{catalog}.bronze.products")
reviews_df = spark.table(f"{catalog}.bronze.reviews")
tickets_df = spark.table(f"{catalog}.bronze.support_tickets")
orders_df = spark.table(f"{catalog}.bronze.orders")
returns_df = spark.table(f"{catalog}.bronze.returns")
inventory_snapshots_df = spark.table(f"{catalog}.bronze.inventory_snapshots")

# Daily review aggregate. Reviews contribute both engagement volume and product
# sentiment/issue signal.
review_daily_df = (
    reviews_df.groupBy(
        "event_date",
        "product_id"
    ).agg(
        F.count("*").alias("review_count"),
        F.round(F.avg("rating"), 2).alias("avg_rating"),
        F.sum(F.when(F.col("sentiment") == "negative", 1).otherwise(0)).alias("negative_sentiment_count"),
        F.sum(F.when(F.col("issue_tag").contains("return"), 1).otherwise(0)).alias("review_return_related_count")
    )
)

# Daily support workload aggregate. The reason field helps connect support
# demand to return/exchange/warranty themes.
ticket_daily_df = (
    tickets_df.groupBy(
        "event_date",
        "product_id"
    ).agg(
        F.count("*").alias("ticket_count"),
        F.sum(F.when(F.col("reason").isin("return", "exchange", "warranty"), 1).otherwise(0)).alias("ticket_return_related_count")
    )
)

# Orders provide the denominator for return rates and the commercial demand
# context for issues. A product with many complaints but huge order volume may
# need a different action than a low-volume product with severe issues.
order_daily_df = (
    orders_df.groupBy(
        "event_date",
        "product_id"
    ).agg(
        F.countDistinct("order_id").alias("order_count"),
        F.sum("quantity").alias("units_sold")
    )
)

# Returns are counted separately from tickets because a ticket may ask for help
# without becoming a return, while a return can be completed without a ticket.
return_daily_df = (
    returns_df.groupBy(
        "event_date",
        "product_id"
    ).agg(
        F.count("*").alias("return_count"),
        F.sum(F.when(F.col("is_exchange"), 1).otherwise(0)).alias("exchange_count")
    )
)

# Inventory gives operational feasibility. Support should not recommend a size
# exchange if available stock is too low.
inventory_daily_df = (
    inventory_snapshots_df.groupBy(
        F.col("snapshot_date").alias("event_date"),
        F.col("product_id")
    ).agg(
        F.sum("on_hand_units").alias("on_hand_units"),
        F.max("stock_status").alias("stock_status")
    )
)

# Pick the most common review issue for each product/day. This is a simple
# readable signal for dashboards and for the SQL metric tool.
issue_counts_daily_df = (
    reviews_df.groupBy(
        "event_date",
        "product_id",
        "issue_tag"
    ).agg(
        F.count("*").alias("issue_count")
    ).withColumn(
        "issue_rank",
        F.row_number().over(
            Window.partitionBy("event_date", "product_id")
            .orderBy(
                F.col("issue_count").desc(),
                F.col("issue_tag").asc()
            )
        )
    ).filter(
        F.col("issue_rank") == 1
    ).select(
        "event_date",
        "product_id",
        F.col("issue_tag").alias("top_issue")
    )
)

# Product daily is the main metric serving table. It combines demand, feedback,
# returns, inventory, and product attributes at one grain:
# one row per product per event_date.
product_daily_df = (
    review_daily_df.join(
        ticket_daily_df,
        ["event_date", "product_id"],
        "full"
    ).join(
        order_daily_df,
        ["event_date", "product_id"],
        "full"
    ).join(
        return_daily_df,
        ["event_date", "product_id"],
        "full"
    ).join(
        inventory_daily_df,
        ["event_date", "product_id"],
        "left"
    ).join(
        issue_counts_daily_df,
        ["event_date", "product_id"],
        "left"
    ).join(
        products_df.select(
            "product_id",
            "product_name",
            "category"
        ),
        "product_id",
        "left"
    ).fillna(
        {
            "review_count": 0,
            "ticket_count": 0,
            "order_count": 0,
            "units_sold": 0,
            "return_count": 0,
            "exchange_count": 0,
            "negative_sentiment_count": 0,
            "review_return_related_count": 0,
            "ticket_return_related_count": 0,
            "on_hand_units": 0,
            "stock_status": "unknown",
            "top_issue": "none",
        }
    ).withColumn(
        "return_related_count",
        F.col("review_return_related_count") + F.col("ticket_return_related_count") + F.col("return_count")
    ).withColumn(
        "return_rate",
        F.when(
            F.col("order_count") > 0,
            F.round(F.col("return_count") / F.col("order_count"), 4)
        ).otherwise(F.lit(0.0))
    ).select(
        "event_date",
        "product_id",
        "product_name",
        "category",
        "review_count",
        "ticket_count",
        "order_count",
        "units_sold",
        "return_count",
        "return_rate",
        "avg_rating",
        "negative_sentiment_count",
        "return_related_count",
        "on_hand_units",
        "stock_status",
        "top_issue",
        F.current_timestamp().alias("updated_at"),
    )
)

product_daily_df.write.mode("overwrite").saveAsTable("product_daily")

# COMMAND ----------

# Gold issue daily uses a different grain:
# one row per product, date, and issue_tag. It is useful when learners want to
# extend the agent with issue-ranking or issue-trend SQL templates.
review_issues_df = (
    reviews_df.select(
        "event_date",
        "product_id",
        "issue_tag",
        F.lit(1).alias("feedback_count"),
        F.when(F.col("sentiment") == "negative", 1).otherwise(0).alias("negative_count"),
        F.lit(0).alias("high_priority_ticket_count")
    )
)

# Ticket reasons become issue tags so support workload can be compared against
# review issue tags and return reasons in one table.
ticket_issues_df = (
    tickets_df.select(
        "event_date",
        "product_id",
        F.col("reason").alias("issue_tag"),
        F.lit(1).alias("feedback_count"),
        F.when(F.col("priority") == "high", 1).otherwise(0).alias("negative_count"),
        F.when(F.col("priority") == "high", 1).otherwise(0).alias("high_priority_ticket_count")
    )
)

# Return reasons are treated as negative issue signals by definition because a
# return means the customer did not keep the product.
return_issues_df = (
    returns_df.select(
        "event_date",
        "product_id",
        F.col("return_reason").alias("issue_tag"),
        F.lit(1).alias("feedback_count"),
        F.lit(1).alias("negative_count"),
        F.lit(0).alias("high_priority_ticket_count")
    )
)

# Union all issue-like signals and aggregate them into the issue serving table.
issue_daily_df = (
    review_issues_df
    .unionByName(ticket_issues_df)
    .unionByName(return_issues_df)
    .groupBy(
        "event_date",
        "product_id",
        "issue_tag"
    ).agg(
        F.sum("feedback_count").alias("feedback_count"),
        F.sum("negative_count").alias("negative_count"),
        F.sum("high_priority_ticket_count").alias("high_priority_ticket_count")
    ).withColumn(
        "updated_at",
        F.current_timestamp()
    )
)

issue_daily_df.write.mode("overwrite").saveAsTable("issue_daily")

print("Gold metrics build complete")