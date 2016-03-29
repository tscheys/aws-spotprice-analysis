// import spark dependencies
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive
// import jodatime
import com.github.nscala_time.time.Imports._

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

    // null unwanted variables from model

    //df =  df.sort(col("AvailabilityZone"), col("InstanceType"), col("Stamp2").asc)

    //df.show()
    //df.printSchema()

    //subset

   /* val asiac3 = df.where(df("AvailabilityZone") === "ap-southeast-1a" && df("InstanceType") === "c3.large")
    asiac3.show()
    println(asiac3.count())

    // make extra column to offset by one row
    asiac3.
      withColumn("PreviousPrice", col("SpotPrice"))

    asiac3.registerTempTable("asia")
    // experiment with sql quering
    val filter = sqlContext.sql("SELECT a.AvailabilityZone,a.Date,a.Time, a.Stamp2,a.SpotPrice, lag(a.SpotPrice) OVER (PARTITION BY a.AvailabilityZone ORDER BY a.Stamp2) AS PreviousPrice FROM asia a")
    filter.show()
    filter.printSchema()  */

    // try to query a dataframe the sql way x

    // try to apply the window lag function on this query x

    // if we get stuck here, make a column with random numbers

    // then remove all unwanted variables for models

    // import mllib

    // make a simple linear regression

    // try out other techniques in the library

  }
}
