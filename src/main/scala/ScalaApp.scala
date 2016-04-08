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

    // LOAD CSV WITH BASETABLE

    // START RF REGRESSION
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
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
    //println("Learned regression forest model:\n" + rfModel.toDebugString)
  }
}
