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
      val fmt = DateTimeFormat.forPattern("yyyy-MM-dd")
      val dt = fmt.parseDateTime(date)
      if(dt.getDayOfWeek < 6) {1} else {0}
    })

    // get day of week
    def dayOfWeek = udf((date: String) => {
      val fmt = DateTimeFormat.forPattern("yyyy-MM-dd")
      val dt = fmt.parseDateTime(date)
      dt.getDayOfWeek
    })

    def getSeconds = udf((hours: Integer, minutes: Integer, seconds: Integer) => {
      hours * 3600 + minutes * 60 + seconds
    })

    // create time variables
    df = df
      .withColumn("date", substring(col("timeStamp"), 0, 10))
      .withColumn("time", substring(col("timeStamp"), 12,8))
      .withColumn("hours", substring(col("timeStamp"), 12,2).cast("Int"))
      .withColumn("minutes", substring(col("timeStamp"), 15,2).cast("Int"))
      .withColumn("seconds", substring(col("timeStamp"), 18,2).cast("Int"))
      .withColumn("unixTime", unix_timestamp(concat_ws(" ", col("date"), col("time"))))
      .withColumn("isDaytime", dayTime(col("hours")))

    df = df
      .withColumn("isWeekDay", isWeekDay(col("date")))
      .withColumn("dayOfWeek", dayOfWeek(col("date")))

    df = df
      .withColumn("SecondsDay", getSeconds(col("hours"), col("minutes"), col("seconds")))

    // make sure changes to columns are correct
    df.show()
    df.printSchema()

    // create lagged (t-1) spot price variable
    df.registerTempTable("cleanData")

    // use Spark window function to lag()
    df = sqlContext.sql("SELECT a.*, lag(a.spotPrice) OVER (PARTITION BY a.availabilityZone, a.instanceType ORDER BY a.unixTime) AS previousPrice FROM cleanData a")

    // check if lag() was done correctly
    df.show(400)
    df.printSchema()

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
      .withColumn("increase", hasIncrease(col("priceChange")))
      .withColumn("decrease", hasDecrease(col("priceChange")))

    // narrow down dataset for regression try
    df.registerTempTable("data")
    df = sqlContext.sql("SELECT unixTime, spotPrice, priceChange, increase FROM data WHERE availabilityZone = 'ap-southeast-1b' AND instanceType= 'm1.medium'")

    // do check
    df.show(400)
    df.printSchema()

    //val rdd: RDD[Vector] = df.rdd

    /*val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(df)*/

    // Train a RandomForest model.
    /*val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    println(rf)*/

    // make a simple linear regression

    // try out other techniques in the library

    val Array(train, validation, test) = df.randomSplit(Array(0.8,0.2,0.2))

    // check if split worked

    train.show()
    validation.show()
    test.show()

    val assembler = new VectorAssembler()
      .setInputCols(Array("unixTime", "spotPrice", "priceChange"))
      .setOutputCol("features")
    //val labeled = df.map(row => LabeledPoint(row.getDouble(0), row(4).asInstanceOf[Vector]))

    //val numIterations = 100
    //val model = SVMWithSGD.train(labeled, numIterations)

    // check if these 3 sets do not overlap

    // evaluate performance of model

    // adjustments to make to the data set

      // interpolate daat points/ average out over 10 minutes ?
      // average out over subregions (1a, 1b,...)
      // bovenstaande taken samen nemen (aggregaten)

  }
}
