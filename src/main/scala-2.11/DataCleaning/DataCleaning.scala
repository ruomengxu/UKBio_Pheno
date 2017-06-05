package DataCleaning

import java.lang.Double

import ioutils.{CSVUtils, UKBio_Dict}
import main.filePaths
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col

import Numeric.Implicits._
import scala.collection.mutable.ListBuffer


/**
  * Created by adamamster on 3/30/17.
  */

class DataCleaning(sparkSession: SparkSession, data_dict: RDD[UKBio_Dict], data: DataFrame) {

  var Category_UDI_table = CSVUtils.loadCSVAsTable(sparkSession, s"${filePaths.inputPrefix}/feature.csv")
  var BASE_UDI = collection.mutable.Map[String,List[String]]()
  for(colname <- Category_UDI_table.schema.fieldNames)
    {
      BASE_UDI +=(colname->Category_UDI_table.select(colname).collect().map(f=>f.getString(0)).filter(_!=null).toList)
    }

  var dataRDD = data.rdd

  def saveMedicationData() = {
    val subset = getSubset(BASE_UDI("Medication"))
    val headers: Array[String] = getColumnNames(BASE_UDI("Medication"))
    val data = getNormalizedData(headers, subset)
    saveFile(data, s"${filePaths.outputPrefix}/MEDICATIONS")
  }

  def saveCancerData() = {
    val subset = getSubset(BASE_UDI("CANCER"))
    val headers: Array[String] = getColumnNames(BASE_UDI("CANCER"))
    val data = getNormalizedData(headers, subset)
    val dataParentIds = replaceWithParentIds(data, s"${filePaths.inputPrefix}/cancerCodes/coding3.tsv")
    saveFile(dataParentIds, s"${filePaths.outputPrefix}/CANCERS")

  }

  def saveNonCancerIllnessData() = {
    val subset = getSubset(BASE_UDI("NonCancerIllness"))
    val headers: Array[String] = getColumnNames(BASE_UDI("NonCancerIllness"))
    val data = getNormalizedData(headers, subset)
    val dataParentIds = replaceWithParentIds(data, s"${filePaths.inputPrefix}/nonCancerIllnessCodes/coding6.tsv")
    saveFile(dataParentIds, s"${filePaths.outputPrefix}/NON_CANCER_ILLNESSES")

  }

  def saveLifestyleData() = {
    val subset = getSubset(BASE_UDI("Lifestyle"))
    val headers: Array[String] = getColumnNames(BASE_UDI("Lifestyle"))
    val data = getNormalizedData(headers, subset)
    saveFile(data, s"${filePaths.outputPrefix}/LIFESTYLE")
  }

  def saveDemographicData() = {
    val subset = getSubset(BASE_UDI("Demographic"))
    val headers: Array[String] = getColumnNames(BASE_UDI("Demographic"))
    val data = getNormalizedData(headers, subset)
    saveFile(data, s"${filePaths.outputPrefix}/DEMOGRAPHIC")
  }
  def savePhysicalMeasureData() = {
    val subset = getSubset(BASE_UDI("PhysicalMeasure"))
    val headers: Array[String] = getColumnNames(BASE_UDI("PhysicalMeasure"))
    val data = getNormalizedData(headers, subset)
    saveFile(data, s"${filePaths.outputPrefix}/PHYSICALMEASURE")


  }

  private def getColumnNames(baseUdis: List[String]) = {
    val baseUdisSet = baseUdis.toSet
    val colNames: Array[String] = this.data.schema.fieldNames.filter(colName => (baseUdisSet contains colName.split("-")(0)) || colName == "eid")
    colNames
  }

  private def getSubset(baseUdis: List[String]): RDD[Row] = {
    val colNames = getColumnNames(baseUdis)
    var subset: RDD[Row] = this.data.select(colNames.map(c => col(s"`$c`")): _*).rdd

    subset
  }

  private def replaceWithParentIds(data: RDD[(String, String, Double)], codingPath: String): RDD[(String, String, Double)] = {
    val codingInfoTable = CSVUtils.loadCSVAsTable(sparkSession, codingPath, "coding", "\t")
    val codingRDD = codingInfoTable.sqlContext.sql("SELECT coding, node_id, parent_id FROM coding").rdd
        .map(r => (r.getString(0), r.getString(1), r.getString(2)))
    val codingNodeIdMap = codingRDD.map{ case (coding, nodeId, parentId) => (nodeId, coding)}.collect.toMap
    val nodeIdParentIdMap = codingRDD.map{ case (coding, nodeId, parentId) => (nodeId, parentId)}.collect.toMap

    //this code selects level 2 disease but it may contain bugs and it's hard to read
    //please take a look at it
//    val codingParentLevel2Map = codingRDD.map{ case (coding, nodeId, parentId) => {
//      val parentId = if(nodeIdParentIdMap.getOrElse(nodeId, "0") == "0") nodeId else nodeIdParentIdMap(nodeId)
//      val parentParentId = if(nodeIdParentIdMap.getOrElse(nodeIdParentIdMap.getOrElse(parentId, "0"), "0") == "0") parentId else nodeIdParentIdMap(parentId) //heart/cardiac problem
//      val parentParentParentId = if(nodeIdParentIdMap.getOrElse(nodeIdParentIdMap.getOrElse(nodeIdParentIdMap.getOrElse(parentId, "0"), "0"), "0") == "0") parentParentId else nodeIdParentIdMap(parentParentId)
//      val parentParentParentParentId = if(nodeIdParentIdMap.getOrElse(nodeIdParentIdMap.getOrElse(nodeIdParentIdMap.getOrElse(nodeIdParentIdMap.getOrElse(parentId, "0"), "0"), "0"), "0") == "0") parentParentParentId else nodeIdParentIdMap(parentParentId)
//      (coding, parentParentParentParentId)}
//    }.collect.toMap
    val toplevel_id = codingRDD.filter(f => f._3=="0").map(_._2).collect.toSet
    val map_t_t = toplevel_id.map(f => (f,f)).toMap // there some top level is selectable
    val level1_id = codingRDD.filter(f => toplevel_id.contains(f._3) ).map(f=>f._2).collect.toSet
    val map_1_1 = level1_id.map(f => (f,f)).toMap
    val level2_id = codingRDD.filter(f => level1_id.contains(f._3)).map(f => f._2).collect.toSet
    val map_2_2 = level2_id.map(f => (f,f)).toMap
    val level3_id = codingRDD.filter(f => level2_id.contains(f._3)).map(f => f._2).collect.toSet
    val map_3_2= nodeIdParentIdMap.filter(f => level2_id.contains(f._2))
    val map_4_2 = nodeIdParentIdMap.filter(f => level3_id.contains(f._2)).map(f => (f._1,map_3_2(f._2)))
    val level2_map = map_1_1 ++ map_2_2 ++ map_3_2 ++ map_4_2 ++ map_t_t
    val codingParentLevel2Map = level2_map.map(f=>(codingNodeIdMap(f._1),f._2))
    data.map{ case (eid, featureName, featureValue) => {
      val featureNameSplit = featureName.split("=")
      val parentId = codingParentLevel2Map(featureNameSplit(1))
      val newFeatureName = s"${featureNameSplit(0)}=${parentId}"
      (eid, newFeatureName, featureValue)
    }}
  }

  private def transformData(headers: Array[String], data: RDD[Row]): RDD[(String, String, Double, String)] = {

    val nCols = headers.length
    val udiFeatureTypeMap = this.data_dict.map(v => (v.UDI.split("-")(0), v.Type)).collect.toMap
    val categoriesToIngore = Set("99999")

    data.map(row => {
      val eid = row.getString(0)

      val list = new ListBuffer[(String, String, java.lang.Double, String)]()
      for(i <- 1 until nCols) {
        val value = row.getString(i)
        val baseUDI = headers(i).split("-")(0)
        val featureType = udiFeatureTypeMap(baseUDI)

        var featureName = ""
        var featureValue: java.lang.Double = 0.0

        if(featureType == "Categorical (multiple)" || featureType == "Categorical (single)") {

          if(value != null) {
            if(!(categoriesToIngore contains value)) {
              featureName = baseUDI + "=" + value
              featureValue = 1.0
            }
          }

        } else {
          featureName = baseUDI
          featureValue = if (value == null) null else value.toDouble

        }

        if(featureName != "") {
          list += ((eid, featureName, featureValue, featureType))
        }

      }
      list
    }).flatMap(identity).distinct
  }

  private def handleNullValues(data: RDD[(String, String, java.lang.Double, String)]): RDD[(String, String, Double)] = {
    //maps featureName to mean value
    val non_categorical_map = data.filter(f => f._3 != null && !(f._4 == "Categorical (multiple)" || f._4 == "Categorical (single)"))
      .map(f => (f._2, (f._3, 1)))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
      .map(f => (f._1, f._2._1 / f._2._2))
      .collect().toMap

    //deal with null value
    data.map{ case (eid, featureName, featureValue, featureType) => {
      var newFeatureName = ""
      var newFeatureValue = 0.0

      if (featureValue == null) {
        if (featureType == "Categorical (multiple)" || featureType == "Categorical (single)") {
          //this feature type is excluded if null
        } else {
          newFeatureName = featureName
          newFeatureValue = non_categorical_map(featureName)
        }
      }

      if(newFeatureName.length > 0 && newFeatureValue != 0.0) {
        (eid, newFeatureName, newFeatureValue)
      } else {
        (eid, featureName, featureValue.toDouble)
      }
    }
    }
  }

  private def minMaxNormalization(data: RDD[(String, String, Double)]): RDD[(String, String, Double)] = {
    val featureMaxMap = data.map(v => (v._2, v._3)).reduceByKey((x, y) => if (x > y) x else y).collect.toMap
    data.map{ case (eid, featureName, value) => (eid, featureName, value / featureMaxMap(featureName))}
  }

  private def getNormalizedData(headers: Array[String], subset: RDD[Row]): RDD[(String, String, Double)] = {
    val deadpatient = transformData(headers, subset).filter(f=>f._2== "40000" & f._3 != null)

    val data = transformData(headers, subset).subtract(deadpatient)

    val imputed = handleNullValues(data)

    val nonnegative = imputed.filter(v => v._3 > 0)

    val normalized = minMaxNormalization(nonnegative)

    normalized
  }

  private def getTop500Medication(data: RDD[(String, String, Double)]): Set[String] = {
    data
      .map(v => (v._2, 1))
      .reduceByKey(_+_)
      .sortBy(_._2, false)
      .map(_._1)
      .take(500)
      .toSet
  }

  private def saveFile(data: RDD[(String, String, Double)],  path: String) = {
    val headersRDD = this.sparkSession.sparkContext.parallelize(Seq("eid,featureName,featureValue"))

    val toString = data.map{ case (eid, featureName, featureValue) => s"$eid,$featureName,$featureValue"}

    headersRDD.union(toString).coalesce(1, true).saveAsTextFile(path)

  }

}
