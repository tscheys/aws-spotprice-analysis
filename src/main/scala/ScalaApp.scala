// import spark dependencies
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive
import org.apache.spark.sql.hive._
import org.apache.spark.ml
import org.apache.spark.ml._
// import jodatime
import com.github.nscala_time.time.Imports._
import org.apache.spark.rdd
import org.apache.spark.rdd._
// ml deps
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Binarizer

import org.apache.spark.ml.classification.LogisticRegression

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}

// import rf regression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

// main class
object ScalaApp {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SpotPriceAnalysis").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
    var df = sqlContext.read.json("/Users/tscheys/ScalaApp/aws.json")

    // inspect data and schema
    df.show()
    df.printSchema()

    // apply correct types to columns
    // rename other columns to camelCase
    df = df
      .withColumn("spotPrice", col("SpotPrice").cast("Double"))
      .withColumnRenamed("AvailabilityZone", "availabilityZone")
      .withColumnRenamed("InstanceType", "instanceType")
      .withColumnRenamed("ProductDescription", "productDescription")
      .withColumnRenamed("Timestamp", "timeStamp")

       // create binary for day/night time
    def dayTime = udf((hour: Integer) => {
      if(hour >= 18 || hour <= 6) 0
      else 1
    })

    // create binary for weekday/weekend
    def isWeekDay = udf((date: String) => {
      val fmt = DateTimeFormat.forPattern("yy-MM-dd")
      val dt = fmt.parseDateTime(date)
      if(dt.getDayOfWeek < 6) {1} else {0}
    })

    // get day of week
    def dayOfWeek = udf((date: String) => {
      val fmt = DateTimeFormat.forPattern("yy-MM-dd")
      val dt = fmt.parseDateTime(date)
      dt.getDayOfWeek
    })

    def getSeconds = udf((hours: Integer, minutes: Integer, seconds: Integer) => {
      hours * 3600 + minutes * 60 + seconds
    })

    def combine = udf((a:String, b: Int, c: Int) => {
      // this function is really bad TODO rewrite!
      var a1 = a.replace("-","")
      var d = 0

      if(b < 10) {
        a1 = a1.concat("0")
      }
      if(c < 15) {
        d = 1
      } else if(c >= 15 && c < 30) {
        d = 2
      } else if(c >= 30 && c < 45) {
        d = 3
      } else {
        d = 4
      }

      a1 + b + d

    })

    // create time variables
    df = df
      .withColumn("date", substring(col("timeStamp"), 0, 10))
      .withColumn("hours", substring(col("timeStamp"), 12,2).cast("Int"))
      .withColumn("minutes", substring(col("timeStamp"), 15,2).cast("Int"))

    // aggregate data (interpolation)

    df = df.withColumn("aggregation", combine(col("date"), col("hours"), col("minutes")).cast("Double"))

    // aggregation solved
    df = df
      .groupBy("availabilityZone", "instanceType","aggregation").mean("spotPrice").sort("availabilityZone", "instanceType", "aggregation")
    df = df
      .withColumnRenamed("avg(spotPrice)", "spotPrice")

    // create separate time variables
    df = df
      .withColumn("hours", substring(col("aggregation").cast("String"), 10, 2).cast("Int"))
      .withColumn("quarter", substring(col("aggregation").cast("String"), 12, 1).cast("Int"))
      .withColumn("date", concat_ws("-", substring(col("aggregation"), 4,2), substring(col("aggregation"), 6, 2), substring(col("aggregation"), 8, 2)))

    // create aggregation variable (average spot price over every 15 minutes)

    df = df
      .withColumn("isWeekDay", isWeekDay(col("date")))
      .withColumn("isDaytime", dayTime(col("hours")))

    // create variable for irrational behaviour

    def isIrrational = udf((zone: String, instance: String, price: Double) => {

      // remove subregion reference a, b, c
      val region  = zone.dropRight(1)

      // check if spot price >= on-demand price
      if(region == "eu-west-1" || region == "us-west-2") {
        if(instance == "m1.medium" && price >= 0.095) 1
        else if(instance == "c3.large" && price >= 0.12) 1
        else if(instance == "g2.2xlarge" && price >= 0.702) 1
        else 0
      } else if(region == "ap-southeast-1") {
        if(instance == "m1.medium" && price >= 0.117) 1
        else if(instance == "c3.large" && price >= 0.132) 1
        else if(instance == "g2.2xlarge" && price >= 1.00) 1
        else 0
      }
      else 0

    })
    df = df
      .withColumn("isIrrational", isIrrational(col("availabilityZone"), col("instanceType"), col("spotPrice")).cast("Integer"))
    // check frequency of irrational behaviour
    df.printSchema()
    df.show()

    // check if isIrration() worked
    println(df.stat.freqItems(Seq("isIrrational")).show())
    //df.show(100)

    // make sure changes to columns are correct
    df.show()
    df.printSchema()

    df.registerTempTable("cleanData")
    // use Spark window function to lag()
    df = sqlContext.sql("SELECT a.*, lag(a.spotPrice) OVER (PARTITION BY a.availabilityZone, a.instanceType ORDER BY a.aggregation) AS previousPrice FROM cleanData a")

    // subtract function
    def subtract = udf((price1: Double, price2: Double) => {
      price1 - price2
    })
    // TODO: simplify these 2 functions
    def hasIncrease = udf((change: Double) => {
      if(change > 0) 1
      else 0
    })

    def hasDecrease = udf((change: Double) => {
      if(change < 0) 1
      else 0
    })

    // subtract current spot price from previous spot price to get priceChange column

    df = df
      .withColumn("priceChange", subtract(col("spotPrice"), col("previousPrice")))
      .withColumn("increaseTemp", hasIncrease(col("priceChange")).cast("Double"))
      .withColumn("decrease", hasDecrease(col("priceChange")).cast("Double"))
    df.registerTempTable("labelData")
    df = sqlContext.sql("SELECT a.*, lead(a.increaseTemp) OVER (PARTITION BY a.availabilityZone, a.instanceType ORDER BY a.aggregation) AS increase, lead(a.spotPrice) OVER (PARTITION BY a.availabilityZone, a.instanceType ORDER BY a.aggregation) AS futurePrice FROM labelData a")

    // check if lag() was done correctly
    df.show(400)
    df.printSchema()

    //narrow down dataset for regression
    // test drive on asia, m1 medium

    df.registerTempTable("data")

    df = sqlContext.sql("SELECT spotPrice, priceChange, hours, quarter, isWeekDay, isDaytime, increase, futurePrice FROM data WHERE availabilityZone = 'ap-southeast-1b' AND instanceType= 'm1.medium'")

    // impute na's
    df = df.na.fill(0.0, Seq("priceChange", "increase", "futurePrice"))
  }
}
