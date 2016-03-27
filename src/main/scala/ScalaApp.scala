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
    def extractDate(a:String): String = a.substring(0, 9)

    val df2 = df
      .withColumn("Spottmp", df("SpotPrice").cast("double"))
      .drop("SpotPrice")
      .withColumnRenamed("Spottmp","SpotPrice")
    val df3 = df2
      .withColumn("Date", df("Timestamp"))
    //.select("AvailabilityZone","InstanceType","SpotPrice", "Timestamp", "Date")
    // check if data type changed to double
    //def split1(a:String):
    val split = (s: String) => s.substring(0,10)
    val splitudf = udf(split)

    val nieuw = df3
      .withColumn("Time", splitudf(col("Timestamp")))

    nieuw.show()
    nieuw.printSchema()

  }
}
