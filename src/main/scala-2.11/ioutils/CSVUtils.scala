/**
 * @author Hang Su <hangsu@gatech.edu>.
 */
package ioutils

import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}


object CSVUtils {
  def loadCSVAsTable(sqlContext: SparkSession, path: String, tableName: String, delimiter: String = ","):  DataFrame= {
    val data = sqlContext
      .read
      .option("header","true")
      .option("delimiter", delimiter)
      .csv(path)

    data.createOrReplaceTempView(tableName)
    data
  }

  def loadCSVAsTable(sqlContext: SparkSession, path: String): DataFrame = {
    loadCSVAsTable(sqlContext, path, inferTableNameFromPath(path))
  }

  private val pattern = "(\\w+)(\\.csv)?$".r.unanchored
  def inferTableNameFromPath(path: String) = path match {
    case pattern(filename, extension) => filename
    case _ => path
  }
}
