// import spark dependencies
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive
import org.apache.spark.sql.hive._

// import jodatime
import com.github.nscala_time.time.Imports._
import org.apache.spark.rdd
import org.apache.spark.rdd._
// ml deps
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.ml.feature.VectorIndexer
// filewriter
import java.io._

// main class
object ScalaApp {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SpotPriceAnalysis").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("../thesis-data/basetable.csv")
    // Statistics

    // datapoint per availabilityZone - instanceType pair
    val fw = new FileWriter("Users/tscheys/thesis-data/test.txt", true)
    df.groupBy("availabilityZone", "instanceType").count.collect.foreach(println)

  }
}
