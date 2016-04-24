// import spark dependencies
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive._
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD

// joda
import org.joda.time
import org.joda.time._
import org.joda.time.DateTime
import org.joda.time.format.DateTimeFormatter
import org.joda.time.format.DateTimeFormat
import org.joda.time.format.DateTimeFormatter._

// ML
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{ RandomForestClassificationModel, RandomForestClassifier, LogisticRegression, BinaryLogisticRegressionSummary}
import org.apache.spark.ml.evaluation.{ MulticlassClassificationEvaluator, RegressionEvaluator }
import org.apache.spark.ml.feature.{ IndexToString, StringIndexer, VectorIndexer, VectorAssembler, Binarizer }
import org.apache.spark.ml.regression.{ RandomForestRegressor, RandomForestRegressionModel, GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}

import org.apache.spark.mllib.evaluation
import org.apache.spark.mllib.evaluation._

//spark submit command:
//spark-submit --class "basetable" --master "local[2]" --packages "com.databricks:spark-csv_2.11:1.4.0,joda-time:joda-time:2.9.3" target/scala-2.11/sample-project_2.11-1.0.jar

object classifiers {
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
      val Array(trainBig, test) = df.randomSplit(Array(0.8, 0.2))
      val Array(train, validation) = trainBig.randomSplit(Array(0.7, 0.3))

      // Train a RandomForest model.
      val rf = new RandomForestClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("indexedFeatures")
      // Convert indexed labels back to original labels.
      val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.labels)
      val pipeline = new Pipeline()
        .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

      // Select (prediction, true label) and compute test error
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        .setMetricName("precision")

      // Train model.  This also runs the indexers.
      var trees = List(30,60,90,120,150,180,210,240,280,350,400, 450)
      case class Prediction(val trees: Integer, val predictions: DataFrame)
      case class AUC(auc: Double, trees: Integer)
      case class Importance(val name: String, val importance: Double)

      var predicts = for (tree <- trees) yield {
        // Chain indexers and forest in a Pipeline
        rf.setNumTrees(tree)
        val model = pipeline.fit(validation)
        val predictions = model.transform(test)
        Prediction(tree, predictions)
      }
      // Select example rows to display.
      var aucs = for (predict <- predicts) yield {
        val aucRDD = predict.predictions.select("predictedLabel", "label")

        val rdd = aucRDD.rdd.map(row => {
          (row.get(0).toString().toDouble, row.get(1).toString.toDouble)
        })

        val metrics = new BinaryClassificationMetrics(rdd)
        val auROC = metrics.areaUnderROC()
        AUC(auROC, predict.trees)
      }
      aucs = aucs.sortWith(_.auc > _.auc)
      rf.setNumTrees(aucs.head.trees)
      val model = pipeline.fit(trainBig)
      //val model = pipeline.fit(train)
      val predictions = model.transform(test)
      //val accuracy = evaluator.evaluate(predictions)
      val importances = model.stages(2).asInstanceOf[RandomForestClassificationModel].featureImportances

      println(importances)
      val indices = importances.toSparse.indices
      val zipped = for(index <- indices) yield {
        val value = importances(index)
        val name = features(index)
        Importance(name, value)
      }

      val ranking = zipped.sortWith(_.importance > _.importance)
      println(ranking.deep.toString())
      case class Report(val auc: List[AUC], val rank: Array[Importance])
      Report(aucs, ranking)
    }
  def prepare = (data: DataFrame, label: String, features: Array[String]) => {
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
      df
  }
  def neuralNet = (data: DataFrame, label: String, features: Array[String]) => {

    // Split the data into train and test
    // import data
    var df = prepare(data, label, features)

    val Array(train, test) = df.randomSplit(Array(0.6, 0.4))
    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](features.length, 30, 30, 30, 2)
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
    // train the model
    val model = trainer.fit(train)
    // compute precision on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precision")
    println("Precision:" + evaluator.evaluate(predictionAndLabels))
  }

  def logistic = (data: DataFrame, label: String, features: Array[String]) => {
    var df = prepare(data, label, features)
    val Array(training, test) = df.randomSplit(Array(0.7,0.3))
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    val trainingSummary = lrModel.summary

    // Obtain the objective per iteration.
    val objectiveHistory = trainingSummary.objectiveHistory
    objectiveHistory.foreach(loss => println(loss))

    // Obtain the metrics useful to judge performance on test data.
    // We cast the summary to a BinaryLogisticRegressionSummary since the problem is a
    // binary classification problem.
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    val roc = binarySummary.roc
    roc.show()
    println(binarySummary.areaUnderROC)
  }

  def gbt = (data: DataFrame, label: String, features: Array[String]) => {
    // Split the data into training and test sets (30% held out for testing)
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
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    // Train a GBT model.
    val gbt = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

// Chain indexers and GBT in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Train model.  This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
  }
}

object regressors {
  def rf = (data: DataFrame, label: String, features: Array[String]) => {
     // subset dataset for m1.medium and us-west-2a
     // TODO: investigate whether this can be done with one filter function and is this faster ?
      var df = data.filter("InstanceType = 'm1.medium'").filter("AvailabilityZone = 'us-west-2a'")

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
      // get stddev of label col
      var stdev = df.agg(stddev(label)).select(label).head().getDouble(0)

      "Root Mean Squared Error (RMSE) on test data = " + rmse + "\n" + "Adjusted Root Mean Squared Error (RMSE) on test data/STDDEV = " + (rmse/stdev)
  }
  def gbt = (data: DataFrame, label: String, features: Array[String]) => {
    val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(data)

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a GBT model.
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    // Chain indexer and GBT in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))

    // Train model.  This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    var stdev = data.agg(stddev(label)).select(label).head().getDouble(0)

    "Root Mean Squared Error (RMSE) on test data = " + rmse + "\n" + "Adjusted Root Mean Squared Error (RMSE) on test data/STDDEV = " + (rmse/stdev)
  }
}

object helper {
  // CONSTANTS
  // prices for on demand instances
  val M1_EU_US = 0.095
  val C3_EU_US = 0.12
  val G2_EU_US = 0.702
  val M1_AP = 0.117
  val C3_AP = 0.132
  val G2_AP = 1.00

  // HELPER FUNCTIONS FOR BASETABLE
  def isWeekDay = udf((date: String) => {
    var formatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    formatter.parseDateTime(date).dayOfWeek().get
  })

  def aggregate(split: Int) = udf((date: String, hours: Int, minutes: Int) => {
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
    val region = zone.dropRight(1)

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

  // create column with date + 1 day (we want stats of 1st january to be used on 2nd of january)
  def datePlusOne = udf((date: String) => {
    var formatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    var nextDate = formatter.parseDateTime(date).plusDays(1)
    formatter.print(nextDate)
  })

  def dailyStats = (data: DataFrame) => {
    var df = data
    var dailies = df.groupBy("availabilityZone", "instanceType", "date").agg(avg("spotPrice"), max("spotPrice"), min("spotPrice"), avg("priceChange"), max("priceChange"), min("priceChange"), stddev("priceChange"))

    dailies = dailies
      .withColumn("date", datePlusOne(col("date")))
    dailies.show()
    dailies.printSchema()
    df = df
      .join(dailies, Seq("availabilityZone", "instanceType", "date"))
      .withColumn("diffMeanSpot", col("spotPrice") - col("avg(spotPrice)"))
      .withColumn("diffMeanChange", abs(col("priceChange") - col("avg(priceChange)")))
      .withColumnRenamed("stddev_samp(priceChange,0,0)", "stddev")

    // take care of missing stddevs
    var average = df.na.drop().select(avg("stddev")).head()
    df.na.fill(average.getDouble(0), Seq("stddev"))

  }

  // load a basetable with a certain interval
  def loadBasetable = (interval: Int) => {
    getContext
      .read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("../thesis-data/basetable" + interval + ".csv")
  }

  def getContext = {
    val conf = new SparkConf().setAppName("SpotPriceAnalysis").setMaster("local[2]")
    val sc = new SparkContext(conf)
    new org.apache.spark.sql.hive.HiveContext(sc)
  }
}

// main class
object basetable {
  def main(args: Array[String]) {
    val sqlContext = helper.getContext
    val df = sqlContext.read.json("/Users/tscheys/ScalaApp/aws.json")
    val INTERVALS = Seq(15)

    // makes basetable for different time aggregation intervals
    // we will call this function 3 times: for 15, 30 and 60 minutes
    def basetableMaker = (data: DataFrame, interval: Int) => {
      var df = data
        .withColumn("SpotPrice", col("SpotPrice").cast("Double"))

      // create time variables
      df = df
        .withColumn("date", substring(col("TimeStamp"), 0, 10))
        .withColumn("hours", substring(col("TimeStamp"), 12, 2).cast("Int"))
        .withColumn("minutes", substring(col("TimeStamp"), 15, 2).cast("Int"))

      // aggregate data (interpolation)

      df = df.withColumn("aggregation", unix_timestamp(helper.aggregate(interval)(col("date"), col("hours"), col("minutes"))))

      // do quick check if aggregation is properly able to form groups
      df.orderBy("AvailabilityZone", "InstanceType", "aggregation").show()

      // take mean over fixed time interval chosen in aggregate() function
      df = df
        .groupBy("AvailabilityZone", "InstanceType", "aggregation").mean("spotPrice").sort("AvailabilityZone", "InstanceType", "aggregation")
      df.show()
      df = df
        .withColumnRenamed("avg(spotPrice)", "spotPrice")

      // create separate time variables
      df = df
        .withColumn("TimeStamp", from_unixtime(col("aggregation")))
        .withColumn("date", substring(col("TimeStamp"), 1, 10))
        .withColumn("hours", substring(col("TimeStamp"), 12, 2).cast("Int"))
        .withColumn("quarter", substring(col("TimeStamp"), 16, 1).cast("Int"))
        .withColumn("dayOfWeek", helper.isWeekDay(col("date")))
        .withColumn("isWeekDay", (helper.isWeekDay(col("date")) <= 5).cast("Int"))
        .withColumn("isWorktime1", (col("hours") >= 6 && col("hours") <= 18).cast("Int"))
        .withColumn("isWorktime2", (col("hours") >= 9 && col("hours") <= 17).cast("Int"))
        .withColumn("isWorktime3", (col("hours") >= 8 && col("hours") <= 18).cast("Int"))
        .withColumn("isNight", (col("hours") <= 6).cast("Int"))
        .withColumn("isIrrational", helper.isIrrational(col("AvailabilityZone"), col("InstanceType"), col("spotPrice")).cast("Integer"))

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
        .withColumn("increaseTemp", (col("priceChange") > 0))
        .withColumn("decreaseTemp", (col("priceChange") < 0))
        .withColumn("sameTemp", (col("priceChange") === 0))

      df.registerTempTable("labelData")
      df = sqlContext.sql("""SELECT a.*, lead(a.increaseTemp) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS increase,
        lead(a.decreaseTemp) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS decrease,
        lead(a.sameTemp) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS same,
        lead(a.spotPrice) OVER (PARTITION BY a.AvailabilityZone, a.InstanceType ORDER BY a.aggregation) AS futurePrice
        FROM labelData a""")

      def change = udf((inc: Double, decr: Double, same: Double) => {
        if(inc == 1) 3
        else if (decr == 1) 0
        else if (same == 1) 2
        2
      })
      df = df
        .withColumn("increase", col("increase").cast("Double"))
        .withColumn("decrease", col("decrease").cast("Double"))
        .withColumn("same", col("same").cast("Double"))
        .withColumn("change", change(col("increase"), col("decrease"), col("same")))

      // remove null rows created by performing a lead
      // TODO: how many rows do we omit by doing this ?
      df = df.na.drop()
      // check if lag() was done correctly

      // get statistics avg, max, min, stddev
      // calculate avg, max, min, stddev of previous day
      df = helper.dailyStats(df)
      df.show()

      df = df
        .withColumn("isVolatile", (col("priceChange") > (col("stddev") * 2)).cast("Int"))
        .na.fill(0.0, Seq("priceChange", "futurePrice"))

      // get rid of temporary cols
      df = df
        .drop("increaseTemp")
        .drop("decreaseTemp")
        .drop("sameTemp")

      // write to csv file
      df.write.format("com.databricks.spark.csv")
        .option("header", "true")
        .mode(SaveMode.Overwrite)
        .save("../thesis-data/basetable" + interval + ".csv")

      //debug
      df.printSchema()
    }

    // invoke basetableMaker() for every interval
    INTERVALS.foreach { x => basetableMaker(df, x) }

  }
}

object configClass {
  def main(args: Array[String]) {

    //define time intervals
    val INTERVALS = Seq(15)
    val basetables = for (interval <- INTERVALS) yield helper.loadBasetable(interval)

    //check if loaded correctly into array
    basetables(0).show()

    // define features
    //val allFeatures = Array("spotPrice", "priceChange", "priceChangeLag1", "priceChangeLag2", "isIrrational", "t1", "t2", "t3", "stddev", "isVolatile", "hours", "quarter", "isWeekDay", "isDaytime")
    val features = Array("spotPrice", "hours", "quarter", "diffMeanChange", "aggregation", "avg(spotPrice)", "stddev")
    val labels = Array("increase", "decrease", "same")
    val couples = Array(Array("us-west-2a", "m1.medium"))
    // CONFIG RF CLASSIFIER
    /*
    val accuracies = for (basetable <- basetables; couple <- couples) yield {
      // for each basetable, try out different couples
      classifiers.rfClassifier(basetable, labels(0), features, couple(0), couple(1))
    }
    println(accuracies(0).auc.toString())
    println(accuracies(0).rank.deep.mkString("\n").toString())
    *
    */
    // CONFIG NEURAL NET
    //classifiers.neuralNet(basetables(0).filter("instanceType='m1.medium'").filter("availabilityZone='us-west-2a'"), labels(0), features)
    // CONFIG LOGISTIC CLASSIFIER
    //classifiers.logistic(basetables(0).filter("instanceType='m1.medium'").filter("availabilityZone='us-west-2a'"), labels(0), features)
    // CONFIG GBT
    classifiers.gbt(basetables(0).filter("instanceType='m1.medium'").filter("availabilityZone='us-west-2a'"), labels(0), features)
  }
}

object configReg {
  def main(args: Array[String]) {

    //define time intervals
    var INTERVALS = Seq(60)

    val basetables = for (interval <- INTERVALS) yield helper.loadBasetable(interval)

    val RMSE = for (basetable <- basetables) yield rfRegression(basetable)

    RMSE.foreach { println }

  }
}

object statistics {
  def main(args: Array[String]) {
    // should be for all three
    val df = helper.loadBasetable(60)

    // TODO: get list of string variables dynamically
    // create val with y-var names
    // create command that gets all cols except of type string
    val corFeatures = df.columns.diff(Array("TimeStamp", "availabilityZone", "instanceType","date", "futurePrice", "increase", "decrease", "same"))

    case class Correlation(val feat1: String, val feat2: String, val corr: Double)

    val visual = df.filter("instanceType='m1.medium'").filter("availabilityZone= 'us-west-2a'").select("spotPrice", "aggregation").coalesce(1)
        .write.format("com.databricks.spark.csv")
        .option("header", "true")
        .mode(SaveMode.Overwrite)
        .save("../thesis-data/series.csv")
    val density = df.groupBy("availabilityZone", "instanceType", "date").count()
    val avg = density.groupBy("availabilityZone", "instanceType").mean("count").show()
        /*
    val corrIncrease = for (feature <- corFeatures) yield  feature + ": " +  df.stat.corr(feature, "increase")
    val corrFuture = for (feature <- corFeatures) yield  feature + ": " +  df.stat.corr(feature, "futurePrice")
    var corrs = for(i <- corFeatures; j <- corFeatures) yield {
        // put these in another dataframe for quick manipulation/sorting/...
        // create a new correlation object, round number to 2 decimals,  get absolute value
        Correlation(i, j, Math.abs(df.stat.corr(i, j)))
    }

    // check frequency of volatility
    var volatileFreq = df.groupBy("isVolatile").count()
    var irrationalFreq = df.groupBy("isIrrational").count()
    println("number of volatile obs")
    volatileFreq.show()
    println("number of irrational obs")
    irrationalFreq.show()

    println("### OBSERVATIONS PER COUPLE:")
    df.groupBy("availabilityZone", "instanceType").count.sort("count").show()
    println("### PRICE VOLATILITY PER COUPLE:")
    df.groupBy("availabilityZone", "instanceType").avg("priceChange").sort("avg(priceChange)").show()
    println("### VARIABLE CORRELATIONS (FOR LABEL 'INCREASE'):" + corrIncrease.deep.mkString("\n"))
    println("### VARIABLE CORRELATIONS (FOR Y-VAR 'FUTUREPRICE'):" + corrFuture.deep.mkString("\n"))
    println("### CORRELATIONS BETWEEN FEATURES")
    val sorted = corrs.sortWith(_.corr > _.corr)
    println(sorted.deep.mkString("\n"))
    *
    */
  }
}

object testing {
  def main(args: Array[String]) {
    val df = helper.loadBasetable(60)
    df.printSchema()

    // do reporting on asz and instances
    val azs = df.select("availabilityZone").distinct()
    val instances = df.select("instanceType").distinct()
    val couples = df.select("availabilityZone", "instanceType").distinct()

    // select certain instance in certain az on a certain date
    val averageCheck = df.filter("availabilityZone = 'us-west-2a'").filter("instanceType = 'm1.medium'").filter("date = '2016-02-12'")
    // calculate average on that date
    val ourAverage = averageCheck.select("spotPrice").agg(avg("spotPrice")).head.getDouble(0)
    // select same instance in same az on that date + 1 day
    val lookupAverage = df.filter("availabilityZone = 'us-west-2a'").filter("instanceType = 'm1.medium'").filter("date = '2016-02-13'").select("avg(spotPrice)").head.getDouble(0)
    // check if average 1 equals average 2

    // get random aggregation row
    // 1456815660
    // very inefficient
    val range = df.filter("availabilityZone = 'us-west-2a'").filter("instanceType = 'm1.medium'").filter("date = '2016-02-02'").sort("aggregation")
    // take first value of that day
    val aggregate = range.select("aggregation").head().getInt(0)
    val spot1 = range.select("spotPrice").head().getDouble(0)
    val spot3 = range.filter("aggregation = '" + (aggregate + 7200) + "'").select("spotPrice").head().getDouble(0)
    val rowNow = range.filter("aggregation = '" + (aggregate + 3600) +  "'").select("spotPrice", "futurePrice", "priceChange")
    val now = for(x <- Vector(0,1,2)) yield rowNow.head().getDouble(x)

    val random = df.sample(false, 1).select("timeStamp", "hours", "dayOfWeek")
    //visually inspect random row
    //do reporting
    println("### DATA QUALITY CHECKS")

    println("#### ALL COLUMNS HAVE CORRECT TYPE")
    println("#### WE HAVE AZs")
    println("Number of Availability Zones: " + azs.count)
    println("list: /n " + azs.show())
    println("#### WE HAVE INSTANCES")
    println("Number of Instances: /n " + instances.count)
    println("list: /n " + instances.show())
    println("#### WE HAVE 3 InstanceType in each of the 8 AZ's (24)")
    println("Number of Instance-AZ combinations: " + couples.count)
    println("list: /n " + couples.show())

    println("#### DAILY STATISTICS SHOULD CALCULATE STATISTICS FROM PREVIOUS DAY")
    println("number 1 = " + ourAverage + "/n" + "number 2 = " + lookupAverage)

    println("#### PRICECHANGE SHOULD BE DIFFERENCE BETWEEN SP at time T and SP at time T + 1")
    println("pricechange == previous price - current price: " + now(2) == now(0) - spot1 )
    println("future price == next spot price: " + now(1) == spot3)

    println("#### HOUR, DOW, SHOULD BE ENCODED CORRECTLY BASED ON TIMESTAMP")
    println(random.show())

  }
}
