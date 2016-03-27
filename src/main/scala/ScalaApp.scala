// import dependencies
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql
import org.apache.spark.sql.functions._
import com.github.nscala_time.time.Imports._

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

    df = df
      .withColumn("Spottmp", df("SpotPrice").cast("double"))
      .drop("SpotPrice")
      .withColumnRenamed("Spottmp","SpotPrice")
    df = df
      .withColumn("Date", df("Timestamp"))
    //.select("AvailabilityZone","InstanceType","SpotPrice", "Timestamp", "Date")
    // check if data type changed to double
    val splitDate = udf((s: String) => s.substring(0,10))
    val splitTime = udf((s: String) => s.substring(11))
    val splitMinutes = udf((s: String) => s.substring(14,16))
    val splitHours = udf((s: String) => s.substring(11,13))
    val dayTime = udf((s: String) => {
      val hour = s.toInt
      if(hour >= 18 || hour <= 6) 0
      else 1
    })

    /*val format = new java.text.SimpleDateFormat("yyyy-MM-dd")
    val isWeekDay = udf((s: String) => {
      val date = format.parse(s)
    })*/

    df = df
      .withColumn("Date", splitDate(col("Timestamp")))
      .withColumn("Time", splitTime(col("Timestamp")))
      .withColumn("Hour", splitHours(col("Timestamp")))
      .withColumn("Minutes", splitMinutes(col("Timestamp")))

    df = df
      .withColumn("Daytime", dayTime(col("Hour")))
      //.withColumn("Weekday", isWeekDay(col("Date")))

    df.show()
    df.printSchema()

  }
}
