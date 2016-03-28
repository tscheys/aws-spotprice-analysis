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
    val splitTime = udf((s: String) => s.substring(11,19))
    val splitMinutes = udf((s: String) => s.substring(14,16))
    val splitHours = udf((s: String) => s.substring(11,13))
    val dayTime = udf((s: String) => {
      val hour = s.toInt
      if(hour >= 18 || hour <= 6) 0
      else 1
    })

    val combine = udf((a: String, b: String) => {
      a + " " + b
    })

    df = df
      .withColumn("Date", splitDate(col("Timestamp")))
      .withColumn("Time", splitTime(col("Timestamp")))
      .withColumn("Hour", splitHours(col("Timestamp")))
      .withColumn("Minutes", splitMinutes(col("Timestamp")))

    df = df
      .withColumn("Stamp", combine(col("Date"), col("Time")))

    df = df
      .withColumn("Daytime", dayTime(col("Hour")))

    // get unix timestamp from date and time information
    df = df
      .withColumn("Stamp2", unix_timestamp(col("Stamp")))

    // is date weekday or weekend?
    /*val isWeekDay = udf((date: String) => {
      val fmt = DateTimeFormat.forPattern("yyyy-MM-dd")
      val dt = fmt.parseDateTime(date)
      if(dt < 6) {1} else {0}
    })*/

    val getSeconds = udf((time: String) => {
      // split time on :
      val times = time.split(":")
      times(0).toInt * 3600 + times(1).toInt * 60 + times (2).toInt
    })

    /*df = df
      .withColumn("IsWeekDay", isWeekDay(col("Date")))*/

    df = df
      .withColumn("SecondsDay", getSeconds(col("Time")))

    // null unwanted variables from model

    df.show()
    df.printSchema()

  }
}
