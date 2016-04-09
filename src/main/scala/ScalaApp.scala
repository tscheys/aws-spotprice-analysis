// import spark dependencies
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._

import org.apache.spark.sql.hive._
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.DataFrame

// import jodatime
import com.github.nscala_time.time.Imports._

// main class
object ScalaApp {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SpotPriceAnalysis").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
    var df = sqlContext.read.json("/Users/tscheys/ScalaApp/aws.json")

    // CONSTANTS
    // prices for on demand instances
    val M1_EU_US = 0.095
    val C3_EU_US = 0.12
    val G2_EU_US = 0.702
    val M1_AP = 0.117
    val C3_AP = 0.132
    val G2_AP = 1.00

    val INTERVALS = Seq(15, 30, 45)

    // HELPER FUNCTIONS

    // create binary for weekday/weekend
    def isWeekDay = udf((date: String) => {
      val fmt = DateTimeFormat.forPattern("yyyy-MM-dd")
      val dt = fmt.parseDateTime(date)
      if(dt.getDayOfWeek < 6) {1} else {0}
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
      df.show()

      // create aggregation variable (average spot price over every 15 minutes)

      df = df
        .withColumn("isWeekDay", isWeekDay(col("date")))
        .withColumn("isDaytime", (col("hours") > 6 || col("hours") < 18).cast("Int"))

      df = df
        .withColumn("isIrrational", isIrrational(col("AvailabilityZone"), col("InstanceType"), col("spotPrice")).cast("Integer"))
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
        .withColumn("increaseTemp", (col("priceChange") > 0).cast("Int"))
        .withColumn("decreaseTemp", (col("priceChange") < 0).cast("Int"))
        .withColumn("sameTemp", (col("priceChange") === 0).cast("Int"))

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

      var deviations = df.groupBy("AvailabilityZone", "InstanceType", "date").agg(stddev("priceChange"))
      deviations = deviations
        .withColumnRenamed("stddev_samp(priceChange,0,0)", "stddev")

      // calculate average of stddev
      var average = deviations.na.drop().select(avg("stddev")).head()
      // fill average when deviation was NaN
      deviations = deviations.na.fill(average.getDouble(0), Seq("stddev"))
      deviations.show()

      // join deviations and df
      df = df
        .join(deviations, Seq("AvailabilityZone", "InstanceType", "date"))

      df = df
        .withColumn("isVolatile", (col("priceChange") > (col("stddev") * 2)).cast("Int"))

      // impute na's
      df = df.na.fill(0.0, Seq("priceChange", "increase", "futurePrice", "isVolatile"))

      // check final basetable
      df.orderBy("AvailabilityZone", "InstanceType", "aggregation").show(400)
      df.printSchema()

      // save basetable to csv
      df.write.format("com.databricks.spark.csv").option("header", "true").mode(SaveMode.Overwrite).save("/Users/tscheys/thesis-data/basetable" + interval + ".csv")
    }

    // invoke basetableMaker() for every interval
    INTERVALS.foreach { x => basetableMaker(df, x) }

  }
}
