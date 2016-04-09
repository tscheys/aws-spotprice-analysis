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

// for writing to text file
import java.io._

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

    def combine(split: Int) = udf((date:String, hours: Int, minutes: Int) => {
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

      // create string ready for unix_timestamp conversion
      // minutes in this unix are meaningless, since we use the group variable to perform a .groupby().mean("spotPrice") on the aggregation column
      date + " " + hours + ":" + group + ":" + "00"

    })

    // create time variables
    df = df
      .withColumn("date", substring(col("timeStamp"), 0, 10))
      .withColumn("hours", substring(col("timeStamp"), 12,2).cast("Int"))
      .withColumn("minutes", substring(col("timeStamp"), 15,2).cast("Int"))

    // aggregate data (interpolation)

    df = df.withColumn("aggregation", unix_timestamp(combine(15)(col("date"), col("hours"), col("minutes"))))

    // do quick check if aggregation is properly able to form groups
    df.orderBy("availabilityZone", "instanceType", "aggregation").show()

    // take mean over fixed time interval chosen in combine() function
    df = df
      .groupBy("availabilityZone", "instanceType","aggregation").mean("spotPrice").sort("availabilityZone", "instanceType", "aggregation")
    df = df
      .withColumnRenamed("avg(spotPrice)", "spotPrice")

    // create separate time variables
    // 1970-01-01 00:00:00
    df = df
      .withColumn("timestamp", from_unixtime(col("aggregation")))
      .withColumn("hours", substring(col("timestamp"), 12, 2).cast("Int"))
      .withColumn("quarter", substring(col("timestamp"), 16, 1).cast("Int"))
      .withColumn("date", substring(col("timestamp"), 1, 10))
    df.show()

    // create aggregation variable (average spot price over every 15 minutes)

    df = df
      .withColumn("isWeekDay", isWeekDay(col("date")))
      .withColumn("isDaytime", dayTime(col("hours")))

    // create variable for irrational behaviour

    def isIrrational = udf((zone: String, instance: String, price: Double) => {

      // remove subregion reference a, b, c
      val region  = zone.dropRight(1)

      // check if spot price >= on-demand price
      if(region == "eu-west-1" || region == "us-west-2") {
        if(instance == "m1.medium" && price >= 0.095) 1
        else if(instance == "c3.large" && price >= 0.12) 1
        else if(instance == "g2.2xlarge" && price >= 0.702) 1
        else 0
      } else if(region == "ap-southeast-1") {
        if(instance == "m1.medium" && price >= 0.117) 1
        else if(instance == "c3.large" && price >= 0.132) 1
        else if(instance == "g2.2xlarge" && price >= 1.00) 1
        else 0
      }
      else 0

    })
    df = df
      .withColumn("isIrrational", isIrrational(col("availabilityZone"), col("instanceType"), col("spotPrice")).cast("Integer"))
    // check frequency of irrational behaviour
    df.printSchema()
    df.show()

    // check if isIrration() worked
    println(df.stat.freqItems(Seq("isIrrational")).show())
    //df.show(100)

    // make sure changes to columns are correct
    df.show()
    df.printSchema()

    df.registerTempTable("cleanData")
    // use Spark window function to lag()
    df = sqlContext.sql("""SELECT a.*,
lag(a.spotPrice) OVER (PARTITION BY a.availabilityZone, a.instanceType ORDER BY a.aggregation) AS t1,
lag(a.spotPrice, 2) OVER (PARTITION BY a.availabilityZone, a.instanceType ORDER BY a.aggregation) AS t2,
lag(a.spotPrice, 3) OVER (PARTITION BY a.availabilityZone, a.instanceType ORDER BY a.aggregation) AS t3
FROM cleanData a""")

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
      .withColumn("priceChange", subtract(col("spotPrice"), col("t1")))
      .withColumn("priceChangeLag1", subtract(col("t1"), col("t2")))
      .withColumn("priceChangeLag2", subtract(col("t2"), col("t3")))
      .withColumn("increaseTemp", hasIncrease(col("priceChange")).cast("Double"))
      .withColumn("decrease", hasDecrease(col("priceChange")).cast("Double"))
    df.registerTempTable("labelData")
    df = sqlContext.sql("SELECT a.*, lead(a.increaseTemp) OVER (PARTITION BY a.availabilityZone, a.instanceType ORDER BY a.aggregation) AS increase, lead(a.spotPrice) OVER (PARTITION BY a.availabilityZone, a.instanceType ORDER BY a.aggregation) AS futurePrice FROM labelData a")

    // check if lag() was done correctly
    df.show(400)
    df.printSchema()

    var deviations = df.groupBy("availabilityZone", "instanceType", "date").agg(stddev("priceChange"))
    deviations = deviations
      .withColumnRenamed("stddev_samp(priceChange,0,0)", "stddev")

    // calculate average of stddev
    var average = deviations.na.drop().select(avg("stddev")).head()
    println(average + " " + average.getDouble(0))
    // fill average when deviation was NaN
    deviations = deviations.na.fill(average.getDouble(0), Seq("stddev"))
    deviations.show()

    // join deviations and df
    df = df
      .join(deviations, Seq("availabilityZone", "instanceType", "date"))

    def isVolatile = udf((priceChange: Double, stddev: Double) => {
      if(priceChange > 2 * stddev) {
        1
      } else {
        0
      }
    })

    df = df
      .withColumn("isVolatile", isVolatile(col("priceChange"), col("stddev")))

    // impute na's
    df = df.na.fill(0.0, Seq("priceChange", "increase", "futurePrice"))

    // check frequency of volatility
    var volatileFreq = df.groupBy("isVolatile").count()
    volatileFreq.show()

    // check final basetable
    df.orderBy("availabilityZone", "instanceType", "aggregation").show(400)
    df.printSchema()

    // save basetable to csv
    df.write.format("com.databricks.spark.csv").save("/Users/tscheys/thesis-data/basetable.csv")

  }
}
