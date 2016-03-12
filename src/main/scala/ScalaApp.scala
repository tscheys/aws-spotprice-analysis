/* SampleApp.scala:
   This application simply counts the number of lines that contain "val" from itself
 */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql

 
object ScalaApp {
  def main(args: Array[String]) {
    //val txtFile = "/Users/tscheys/ScalaApp/src/main/scala/ScalaApp.scala"
    
    val conf = new SparkConf().setAppName("Sample Application").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val df = sqlContext.read.json("/Users/tscheys/second.json")
    df.show()
    
    /* val txtFileLines = sc.textFile(txtFile , 2).cache()
     * val numAs = txtFileLines .filter(line => line.contains("val")).count()
     * 
     */
    
    println("Het werkt")
  }
}
