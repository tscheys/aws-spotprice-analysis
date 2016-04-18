// import spark dependencies
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive._
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.DataFrame

// joda
import org.joda.time
import org.joda.time._
import org.joda.time.DateTime
import org.joda.time.format.DateTimeFormatter
import org.joda.time.format.DateTimeFormat
import org.joda.time.format.DateTimeFormatter._

// ML
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler, Binarizer}
import org.apache.spark.ml.regression.{RandomForestRegressor, RandomForestRegressionModel}

//spark submit command:
// spark-submit --class "basetable" --master "local[2]" --packages "com.databricks:spark-csv_2.11:1.4.0,joda-time:joda-time:2.9.3" target/scala-2.11/sample-project_2.11-1.0.jar

// main class
object basetable {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SpotPriceAnalysis").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
    val df = sqlContext.read.json("/Users/tscheys/ScalaApp/aws.json")

    // CONSTANTS
    // prices for on demand instances
    val M1_EU_US = 0.095
    val C3_EU_US = 0.12
    val G2_EU_US = 0.702
    val M1_AP = 0.117
    val C3_AP = 0.132
    val G2_AP = 1.00

    val INTERVALS = Seq(60)

    // HELPER FUNCTIONS

    // create binary for weekday/weekend
    def isWeekDay = udf((date: String) => {
      var formatter: DateTimeFormatter  = DateTimeFormat.forPattern("yyyy-MM-dd")
      formatter.parseDateTime(date).dayOfWeek().get
    })

    def aggregate(split: Int) = udf((date:String, hours: Int, minutes: Int) => {
      // aggregate datapoints on 15, 30 or 60 minute intervals

      // initialize variable to be re-assigned a value in pattern matching
      var group = 0

      split match {
        // split the data every 15, 30 or 60 minutes
        case 15 => minutes match {
          // reassign all minutes to one specific minute in one of 4 quarters, so they can later be grouped
          case x if x < 15 => group = 1
          case x if x >= 15 && x < 30 => group = 2
          case x if x >= 30 && x < 45 => group = 3
          case x => group = 4
          }
        case 30 => minutes match {
          case x if x < 30 => group = 1
          case x if x >= 30 => group = 2
        }
        case 60 => group = 1
      }

      // create string ready for unix_TimeStamp conversion
      // minutes in this unix are meaningless, since we use the group variable to perform a .groupby().mean("spotPrice") on the aggregation column
      date + " " + hours + ":" + group + ":" + "00"

    })
    def isIrrational = udf((zone: String, instance: String, price: Double) => {
      // remove subregion reference a, b, c
      val region  = zone.dropRight(1)

      // check if spot price >= on-demand price
      region match {
        case x if x == "eu-west-1" || x == "us-west-2" => instance match {
          case x if x == "m1.medium" => price >= M1_EU_US
          case x if x == "c3.large" => price >= C3_EU_US
          case x if x == "g2.2xlarge" => price >= G2_EU_US
        }
        case x if x == "ap-southeast-1" => instance match {
          case x if x == "m1.medium" => price >= M1_AP
          case x if x == "c3.large" => price >= C3_AP
          case x if x == "g2.2xlarge" => price >= G2_AP
        }
      }
    })

    // makes basetable for different time aggregation intervals
    // we will call this function 3 times: for 15, 30 and 60 minutes
    def basetableMaker = (data: DataFrame, interval: Int) => {
      var df = data
        .withColumn("SpotPrice", col("SpotPrice").cast("Double"))

      // create time variables
      df = df
        .withColumn("date", substring(col("TimeStamp"), 0, 10))
        .withColumn("hours", substring(col("TimeStamp"), 12,2).cast("Int"))
        .withColumn("minutes", substring(col("TimeStamp"), 15,2).cast("Int"))

      // aggregate data (interpolation)

      df = df.withColumn("aggregation", unix_timestamp(aggregate(interval)(col("date"), col("hours"), col("minutes"))))

      // do quick check if aggregation is properly able to form groups
      df.orderBy("AvailabilityZone", "InstanceType", "aggregation").show()

      // take mean over fixed time interval chosen in aggregate() function
      df = df
        .groupBy("AvailabilityZone", "InstanceType","aggregation").mean("spotPrice").sort("AvailabilityZone", "InstanceType", "aggregation")
      df = df
        .withColumnRenamed("avg(spotPrice)", "spotPrice")

      // create separate time variables
      df = df
        .withColumn("TimeStamp", from_unixtime(col("aggregation")))
        .withColumn("hours", substring(col("TimeStamp"), 12, 2).cast("Int"))
        .withColumn("quarter", substring(col("TimeStamp"), 16, 1).cast("Int"))
        .withColumn("date", substring(col("TimeStamp"), 1, 10))
        .withColumn("isWeekDay", (isWeekDay(col("date")) <= 5).cast("Int"))
        .withColumn("dayOfWeek", isWeekDay(col("date")))
        .withColumn("isDaytime", (col("hours") >= 6 && col("hours") <= 18).cast("Int"))
        .withColumn("isWorktime", (col("hours") >= 9 && col("hours") <= 17).cast("Int"))
        .withColumn("isNight", (col("hours") <= 6).cast("Int"))
        .withColumn("isWorktime2", (col("hours") >= 8 && col("hours") <= 18).cast("Int"))
        .withColumn("isIrrational", isIrrational(col("AvailabilityZone"), col("InstanceType"), col("spotPrice")).cast("Integer"))

      // make sure changes to columns are correct
      df.show()
      df.printSchema()

      df.registerTempTable("cleanData")
      // use Spark window function to lag()
      df = sqlContext.sql("""SELECT a.*,
          lag(a.spotPrice) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS t1,
          lag(a.spotPrice, 2) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS t2,
          lag(a.spotPrice, 3) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS t3
          FROM cleanData a""")

      // some rows contain null values because we have shifted cols with window function
      df = df.na.drop()

      // subtract current spot price from previous spot price to get priceChange column

      df = df
        .withColumn("priceChange", col("spotPrice") - col("t1"))
        .withColumn("priceChangeLag1", col("t1") - col("t2"))
        .withColumn("priceChangeLag2", col("t2") - col("t3"))
        .withColumn("increaseTemp", (col("priceChange") > 0).cast("Double"))
        .withColumn("decreaseTemp", (col("priceChange") < 0).cast("Double"))
        .withColumn("sameTemp", (col("priceChange") === 0).cast("Double"))

      df.registerTempTable("labelData")
      df = sqlContext.sql("""SELECT a.*, lead(a.increaseTemp) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS increase,
        lead(a.decreaseTemp) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS decrease,
        lead(a.sameTemp) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS same,
        lead(a.spotPrice) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS futurePrice
        FROM labelData a""")

      // remove null rows created by performing a lead
      df.na.drop()
      // check if lag() was done correctly
      df.show(400)
      df.printSchema()

      // calculate avg, max, min, stddev of previous day
      var dailies = df.groupBy("availabilityZone", "instanceType","date").agg(avg("spotPrice" ),max("spotPrice"), min("spotPrice"), avg("priceChange"), max("priceChange"), min("priceChange"))

      // create column with date + 1 day (we want stats of 1st january to be used on 2nd of january)
      def datePlusOne = udf((date: String) => {
        var formatter: DateTimeFormatter  = DateTimeFormat.forPattern("yyyy-MM-dd")
        var nextDate = formatter.parseDateTime(date).plusDays(1)
        formatter.print(nextDate)
      })
      dailies = dailies
        .withColumn("date", datePlusOne(col("date")))
      dailies.show()
      dailies.printSchema()
      df = df
        .join(dailies, Seq("availabilityZone", "instanceType" ,"date"))
        .withColumn("diffMeanSpot", col("spotPrice") - col("avg(spotPrice)"))
        .withColumn("diffMeanChange", abs(col("priceChange") - col("avg(priceChange)")))

      var deviations = df.groupBy("availabilityZone", "instanceType", "date").agg(stddev("priceChange"))
      deviations = deviations
        .withColumnRenamed("stddev_samp(priceChange,0,0)", "stddev")

      df.show()
      //df.printSchema()
      //calculate average of stddev
      var average = deviations.na.drop().select(avg("stddev")).head()
      // fill average when deviation was NaN
      deviations = deviations.na.fill(average.getDouble(0), Seq("stddev"))
      deviations = deviations
        .withColumnRenamed("date", "date1")
      df = df
        .withColumnRenamed("date", "date1")
      println("####DEVIATIONS")
      deviations.show()
      println("####BASETABLE")
      df.show()

      // join deviations and df
      df = df
        .join(deviations, Seq("availabilityZone", "instanceType", "date1"))

      df = df
        .withColumn("isVolatile", (col("priceChange") > (col("stddev") * 2)).cast("Int"))

      // impute na's
      df = df.na.fill(0.0, Seq("priceChange", "increase", "decrease", "same" ,"futurePrice", "isVolatile"))

      // check final basetable
      df.orderBy("availabilityZone", "instanceType", "aggregation").show(400)
      df.printSchema()

      // save basetable to csv
      df.write.format("com.databricks.spark.csv").option("header", "true").mode(SaveMode.Overwrite).save("/Users/tscheys/thesis-data/basetable" + interval + ".csv")

    }

    // invoke basetableMaker() for every interval
    INTERVALS.foreach { x => basetableMaker(df, x) }

  }
}

object rfClassifier {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("SpotPriceAnalysis").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)

    //define time intervals
    val INTERVALS = Seq(60)
    val NUM_TREES = 200

    // load a basetable with a certain interval
    def loadBasetable = (interval: Int) => {
      sqlContext
        .read
        .format("com.databricks.spark.csv")
        .option("header", "true") // Use first line of all files as header
        .option("inferSchema", "true") // Automatically infer data types
        .load("../thesis-data/basetable" + interval +".csv")
    }

    val basetables = for (interval <- INTERVALS) yield loadBasetable(interval)

    //check if loaded correctly into array
    basetables(0).show()

    //START RF CLASSIFIER

    def rfClassifier = (data: DataFrame, label: String, features: Array[String], zone: String, instance: String) => {
      // subset dataset for m1.medium and us-west-2a
      //var df = data.filter("InstanceType = 'm1.medium'").filter("AvailabilityZone = 'us-west-2a'")
      var df = data
      val assembler = new VectorAssembler()
        .setInputCols(features)
        .setOutputCol("features")
      // convert increase to binary variable
      val binarizer: Binarizer = new Binarizer()
        .setInputCol(label)
        .setOutputCol("label")
        .setThreshold(0.5)

      // prepare variables for random forest
      df = assembler.transform(df)
      df = binarizer.transform(df)
      df = df.select("features", "label")

      val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel")
        .fit(df)
      // Automatically identify categorical features, and index them.
      // Set maxCategories so features with > 4 distinct values are treated as continuous.
      val featureIndexer = new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4)
        .fit(df)

      // Split the data into training and test sets (30% held out for testing)
      val Array(train, test) = df.randomSplit(Array(0.7, 0.3))

      // Train a RandomForest model.
      val rf = new RandomForestClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("indexedFeatures")
        .setNumTrees(NUM_TREES)

      // Convert indexed labels back to original labels.
      val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.labels)

      // Chain indexers and forest in a Pipeline
      val pipeline = new Pipeline()
        .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

      // Train model.  This also runs the indexers.
      val model = pipeline.fit(train)

      val predictions = model.transform(test)

      // Select example rows to display.
      predictions.select("predictedLabel", "label", "features").show(100)

      // Select (prediction, true label) and compute test error
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        .setMetricName("precision")
      val accuracy = evaluator.evaluate(predictions)

      val results  = model.stages(2)
      val importances = results.asInstanceOf[RandomForestClassificationModel].featureImportances

      //val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
      //println("Learned classification forest model:\n" + rfModel.toDebugString)
      //return accuracies
      "Test Error = " + (1.0 - accuracy) + "\n" + "Varimportances" + "\n" + importances.toJson
    }

    // define features
    val allFeatures = Array("spotPrice", "priceChange", "priceChangeLag1", "priceChangeLag2", "isIrrational", "t1", "t2", "t3", "stddev", "isVolatile" , "hours", "quarter", "isWeekDay", "isDaytime")
    val features = Array("spotPrice", "priceChange", "hours", "quarter", "isWeekDay", "isDaytime" )
    val featuresAfterImp = Array("spotPrice", "priceChange", "hours", "isWeekDay")
    // , "priceChangeLag1", "priceChangeLag2"
    val labels = Array("increase", "decrease", "same")

    val couples = Array(Array("us-west-2a", "m1.medium" ), Array("ap-southeast-1a", "c3.large"),Array("us-west-2b", "c3.large"))

    // return accuracies for each basetable
    val accuracies = for (basetable <- basetables) yield {
      // for each basetable, try out different couples
      for (couple <- couples) yield  "zone" + couple(0) + " instance " + couple(1) + ":" +  rfClassifier(basetable, labels(0), allFeatures  , couple(0), couple(1))
    }

    println("Report on Random Forest classifier (no trees: " + NUM_TREES + ")")
    println("y var: " + labels(0))
    println("for intervals")
    INTERVALS.foreach(println)
    println("Test error = 1 - accuracy")
    accuracies.foreach(x => x.foreach (println))

  }
}

object rfRegression {
  def main(args: Array[String]) {

    // LOAD CSV WITH BASETABLE
    val conf = new SparkConf().setAppName("SpotPriceAnalysis").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)

    //define time intervals
    var INTERVALS = Seq(15,30,60)

    // load a basetable with a certain interval
    def loadBasetable = (interval: Int) => {
      sqlContext
        .read
        .format("com.databricks.spark.csv")
        .option("header", "true") // Use first line of all files as header
        .option("inferSchema", "true") // Automatically infer data types
        .load("../thesis-data/basetable" + interval +".csv")
    }

    val basetables = for (interval <- INTERVALS) yield loadBasetable(interval)

    // START RF REGRESSION
    def rfRegression = (data: DataFrame) => {

      // subset dataset for m1.medium and us-west-2a
      var df = data.filter("InstanceType == m1.medium").filter("AvailabilityZone == us-west-2a")

      val assembler = new VectorAssembler()
      .setInputCols(Array("spotPrice", "priceChange", "hours", "quarter", "isWeekDay", "isDaytime"))
      .setOutputCol("features")

      // prepare variables for random forest
      df = assembler.transform(df)

      val featureIndexer = new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4)
        .fit(df)

      // Split the data into training and test sets (30% held out for testing)
      val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

      // Train a RandomForest model.
      val rf = new RandomForestRegressor()
        .setLabelCol("futurePrice")
        .setFeaturesCol("indexedFeatures")

      // Chain indexer and forest in a Pipeline
      val pipeline = new Pipeline()
        .setStages(Array(featureIndexer, rf))

      // Train model.  This also runs the indexer.
      val model = pipeline.fit(trainingData)

      // Make predictions.
      val predictions = model.transform(testData)

      // Select example rows to display.
      predictions.select("prediction", "futurePrice", "features").show(40)

      // Select (prediction, true label) and compute test error
      val evaluator = new RegressionEvaluator()
        .setLabelCol("futurePrice")
        .setPredictionCol("prediction")
        .setMetricName("rmse")
      val rmse = evaluator.evaluate(predictions)

      val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
      //println("Learned regression forest model:\n" + rfModel.toDebugString)

      "Root Mean Squared Error (RMSE) on test data = " + rmse
    }

    val RMSE = for (basetable <- basetables) yield rfRegression(basetable)

    RMSE.foreach {println}

  }
}

object statistics {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SpotPriceAnalysis").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("../thesis-data/basetable15.csv")

    val features = Array("spotPrice", "priceChange", "priceChangeLag1", "priceChangeLag2", "isIrrational", "t1", "t2", "t3", "stddev", "isVolatile" , "hours", "quarter", "isWeekDay", "isDaytime")

    df.show()
    df.printSchema()
    // Statistics

    // datapoint per availabilityZone - instanceType pair
    df.groupBy("availabilityZone", "instanceType").count.coalesce(1)
     .write.format("com.databricks.spark.csv")
     .option("header", "true")
     .mode(SaveMode.Overwrite)
     .save("../thesis-data/obsPerCouple.csv")

    df.printSchema()
    df.show()

    // calculate correlations between features and label
   var correlations = for (feature <- features) yield  feature + ": " +  df.stat.corr(feature, "increase")

   df.groupBy("availabilityZone", "instanceType").avg("priceChange").coalesce(1)
     .write.format("com.databricks.spark.csv")
     .option("header", "true")
     .mode(SaveMode.Overwrite)
     .save("../thesis-data/volatility.csv")

    // check frequency of volatility
    var volatileFreq = df.groupBy("isVolatile").count()
    var irrationalFreq = df.groupBy("isIrrational").count()
    println("number of volatile obs")
    volatileFreq.show()
    println("number of irrational obs")
    irrationalFreq.show()

    correlations.foreach (println)
  }
}
