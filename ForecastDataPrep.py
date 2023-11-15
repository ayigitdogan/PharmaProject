
from pyspark.sql import SparkSession
import pandas as pd 
from pyspark.sql import functions as F
import numpy as np
import os

"""This module is used to prepare the data for the model. The data is enriched by promotion labels"""

class ForecastDataPreparation:
    """Raw data is provided in the form of parquet files. This class is used to prepare the data for the model."""

    def __init__(self):

        self.data = pd.read_parquet("Data/sale_data.parquet")
        self.promo_data = pd.read_csv("Data/promo_data.csv")
        self.spark = SparkSession.builder.appName("Forecasting")\
                .config('spark.sql.session.timeZone', 'UTC') \
                .config('spark.driver.memory','600M') \
                .config('spark.ui.showConsoleProgress', True) \
                .config('spark.sql.repl.eagerEval.enabled', True) \
                .getOrCreate()

    def PromotionsPrep(self):

        print("PromotionsPrep Started")
        self.promo_data["EndDate"] = pd.to_datetime(self.promo_data.EndDate)+pd.Timedelta(value=30,unit="min")
        self.promo_data["BeginDate"] = pd.to_datetime(self.promo_data.BeginDate)

        promotions = self.promo_data.loc[:,["MProductId","BeginDate","EndDate","PaidQty","FreeQty"]].\
            groupby(by=["MProductId","PaidQty","FreeQty","BeginDate"]).agg({"EndDate":"max"}).reset_index()
        
        promotions["Promo"]=promotions.PaidQty.astype(str) + "-" + promotions.FreeQty.astype(str)
        promotions["Discount"] = promotions.FreeQty/(promotions.PaidQty+promotions.FreeQty)
        promotions = promotions.rename(columns={"MProductId":"ProductId"})

        print("PromotionsPrep Ended")
        return promotions
    
    def SparkOpetaions(self):

        print("SparkOpetaions Started")
        SparkDF = self.spark.read.parquet("Data/MetaData1.parquet")

        SparkDF = SparkDF.drop("__index_level_0__")
        SparkDF = SparkDF.withColumn("Date",F.to_date("Date"))
        DF = SparkDF.withColumn("Discount",SparkDF.FreeQty/(SparkDF.FreeQty+SparkDF.PaidQty))

        sparkPromo = self.spark.createDataFrame(self.promo_data)
        sparkPromo = sparkPromo.withColumn("BeginDate",F.to_date("BeginDate"))
        DF2 = sparkPromo.withColumn("EndDate",F.to_date("EndDate"))
        
        condition = [DF.ProductId == DF2.ProductId,
                     F.abs(DF2.Discount - DF.Discount)<0.02,
                     DF.Date>=DF2.BeginDate,
                     DF.Date<DF2.EndDate]
        merged = DF.join(DF2,on = condition,how="left").\
            select(DF["*"],DF2.BeginDate,DF2.EndDate,
                   DF2.Promo,DF2.Discount.alias("DStep"))
        
        
        """ merged1 = merged.filter((F.col("Date")>=F.col("BeginDate")) & 
              (F.col("Date")<F.col("EndDate"))&
              (F.abs(F.col("DStep")-F.col("Discount"))<0.02)&
              (F.col("isPromoted")==1))
        
        merged2 = merged.filter(F.col("BeginDate").isNull()) 

        keys = ["Date","LocationId","MainDistributorId", "DistributorId", "PackageId", "ProductId", "City", "District", "BrickId", "PaidQty", "FreeQty","isPromoted","Discount"]
        merged3 = merged.filter((F.col("isPromoted")==0))
        merged3 =merged3.dropDuplicates(keys)

        final_data = merged3.union(merged1.union(merged2)) """

        final_data = merged.withColumn("DStep",F.when(F.col("DStep").isNull(),
                                     F.lit(0)).\
                      otherwise(F.col("DStep")))
        
        print("SparkOpetaions Ended")
        return final_data


    def PreProcess(self):
        """This function is used to preprocess the data"""
        # Converting the date column to datetime format

        print("PreProcess Started")
        self.promo_data = self.PromotionsPrep()
        self.data["isPromoted"] = self.data["FreeQty"].apply(lambda x:  0 if x == 0 else 1)
        self.data['Date'] = self.data['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        self.data.to_parquet("Data/MetaData1.parquet")
        print("PreProcess Ended")

    def Main(self):

        print("Main Started")
        final_data = self.SparkOpetaions()
        print("write started")
        final_data.repartition(1).write.parquet("Data/MetaData2",mode="overwrite")
        print("write ended")
        final_data_v2 = pd.read_parquet("Data/MetaData2/")
        final_data_v2['Date'] = pd.to_datetime(final_data_v2['Date'])
        final_data_v2['BeginDate'] = pd.to_datetime(final_data_v2['BeginDate'])
        final_data_v2['EndDate'] = pd.to_datetime(final_data_v2['EndDate'])
        keys = ["Date","LocationId","MainDistributorId", "DistributorId", "PackageId", "ProductId", "City", "District", "BrickId", "PaidQty", "FreeQty","isPromoted","Discount"]
        final_data_v2 = final_data_v2.drop_duplicates(keys,ignore_index=True)
        final_data_v2['WeekStartDate'] = pd.to_datetime(final_data_v2['Date'].dt.year.astype(str) \
                                                        + '-' + final_data_v2.Date.dt.isocalendar().week.astype(str) + '-' +'1', format='%Y-%W-%w')
        
        self.data = final_data_v2

        print("Main Ended")
    
    def Run(self):

        print("Data Preparation Started")
        self.PreProcess()
        self.Main()
        self.data.to_parquet("Data/Sales_Data_v2.parquet")

        """if os.path.exists("Data/MetaData1.parquet"):
            os.remove("Data/MetaData1.parquet")

        if os.path.exists("Data/MetaData2"):
            os.remove("Data/MetaData2")"""

        print("Data Preparation Completed Successfully")


