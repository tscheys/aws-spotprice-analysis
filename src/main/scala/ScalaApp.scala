// import dependencies
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql
import org.apache.spark.sql.functions._

// main class
object ScalaApp {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Sample Application").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    var df = sqlContext.read.json("/Users/tscheys/ScalaApp/aws.json")

    // inspect data
    df.show()
    df.printSchema()

    // change data types
    //def extractDate(a:String): String = a.substring(0, 9)

    df = df
      .withColumn("Spottmp", df("SpotPrice").cast("double"))
      .drop("SpotPrice")
      .withColumnRenamed("Spottmp","SpotPrice")
      //.withColumn("Date", df("TimeStamp").split("T")[0])
      //.select("AvailabilityZone","InstanceType","SpotPrice", "Timestamp", "Date")
    // check if data type changed to double
    df.show()
    df.printSchema()

    // features engineering
    //df2.map(function )

  }
}
