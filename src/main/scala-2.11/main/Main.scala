
package main

import ioutils.{CSVUtils, UKBio_Dict}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, SQLContext, SparkSession}
import org.apache.spark.SparkContext

import scala.util.Try
import DataCleaning.DataCleaning
import models._
import FeatureConstruction.Feature
import Clustering.{KMeans, clustering}
import DataStatistics.DataStatistics
import java.nio.file.{Paths, Files}
import Evaluation.Pheno_info_table._
import Evaluation.{CosineSimilarity,Silhouette}
import scala.collection.mutable.ListBuffer

object filePaths {
  import java.net._

  private val localhost = InetAddress.getLocalHost
  private val hostName = localhost.getCanonicalHostName
  private val userName = System.getProperty("user.name")
  private val server = if (hostName == "ip-10-0-0-87.ec2.internal") true else false
  val inputPrefix = if (server) s"file:///../data/$userName/" else "data"
  val inputDataPath = if(server) "file:///../data/ukbiobank_data/ukb8036.csv" else "data/ukb8036_sample.csv"
  val outputPrefix = if (server) s"file:///../home/$userName" else "data"
}

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)


    val sparkSession = SparkSession.builder.
      master("local")
      .appName("BDH_PROJECT")
      .getOrCreate()
    val sc=sparkSession.sparkContext

    val data: DataFrame = loadDFRawData(sparkSession)
    val dict: RDD[UKBio_Dict] = loadRDDDict(sc,sparkSession)

    if(!(Files.exists(Paths.get(s"${filePaths.outputPrefix}/MEDICATIONS/part-00000")) &&
      Files.exists(Paths.get(s"${filePaths.outputPrefix}/CANCERS/part-00000")) &&
      Files.exists(Paths.get(s"${filePaths.outputPrefix}/DEMOGRAPHIC/part-00000")) &&
      Files.exists(Paths.get(s"${filePaths.outputPrefix}/LIFESTYLE/part-00000")) &&
      Files.exists(Paths.get(s"${filePaths.outputPrefix}/NON_CANCER_ILLNESSES/part-00000")) &&
      Files.exists(Paths.get(s"${filePaths.outputPrefix}/PHYSICALMEASURE/part-00000")))) {
        new DataCleaning(sparkSession, dict, data).saveLifestyleData()
        new DataCleaning(sparkSession, dict, data).saveMedicationData()
        new DataCleaning(sparkSession, dict, data).saveCancerData()
        new DataCleaning(sparkSession, dict, data).saveNonCancerIllnessData()
        new DataCleaning(sparkSession, dict, data).saveDemographicData()
        new DataCleaning(sparkSession, dict, data).savePhysicalMeasureData()
    }


    //val (medication, cancer, demographic, lifestyle, noncancer, physicalmeasure) = loadCleanedData(sparkSession)

    //comment these out if you don't want statistics printed
//    val statistics = new DataStatistics.DataStatistics(data, medication, cancer, noncancer, demographic, lifestyle, physicalmeasure)
//    statistics.printStatistics()

//    val UDI = Feature.getUDIfromDic(dict)
//    val feature_tuple = sc.union(
//      Feature.constructDemoFeatureTuple(demographic,UDI),
//      Feature.constructCancerFeatureTuple(cancer),
//      Feature.constructLifestyleFeatureTuple(lifestyle,UDI),
//      Feature.constructMedFeatureTuple(medication),
//      Feature.constructPhysicalFeatureTuple(physicalmeasure,UDI),
//      Feature.constructNonCancerFeatureTuple(noncancer)
//    )
//    val sparse_feature_matrix = Feature.construct(sc,feature_tuple)
//
//    val kMeans = new KMeans(sparse_feature_matrix)
//    if(!Files.exists(Paths.get("data/KMeansKError/part-00000"))) {
//      kMeans.saveKErrorToFile()
//    }
//
//    val model = kMeans.run()
//    if(!Files.exists(Paths.get("data/PCAProjectedClusters/part-00000"))) {
//      kMeans.savePCAClustersToFile(model)
//    }
//    val clusterAssignments = kMeans.clusterAssignments(model)
//    //clusterAssignments.foreach(println)
//    var SimilarityList = new ListBuffer[Double]()
//    for (i<- 4 to 12 by 4){
//      val Similarity = CosineSimilarity.Similarity_Score(clusterAssignments,i, sparse_feature_matrix)
//      SimilarityList += Similarity
//    }
//    println(SimilarityList)
//
//
//
//    var Category_UDI= CSVUtils.loadCSVAsTable(sparkSession, "data/feature.csv")
//    val table = info_table(sparkSession,feature_tuple,clusterAssignments,Category_UDI,dict)
//    for(i <- table)
//    {
//      println("Cluster:" + i._1)
//      for (item <- i._2)
//      {
//        println(item._1)
//        println(item._2)
//      }
//    }
    
//    val output_nmf = clustering.Clustering(sparse_feature_matrix,2,1000)

    sparkSession.stop
  }

  //load Raw dataset as DF(only baseline part)
  def loadDFRawData(sqlContext: SparkSession): DataFrame = {
    val rawtable = CSVUtils.loadCSVAsTable(sqlContext, filePaths.inputDataPath)
    val header_base = Array("eid") ++ rawtable.schema.fieldNames.filter(f => f.contains("-0.")).map(f => "`" + f + "`")
    val rawtable_base = rawtable.select(header_base.head, header_base.tail: _*)
    rawtable_base

  }
  //load dict only baseline (attr: Colnum, UDI, Count, Type, Description). Only Description might be empty
  def loadRDDDict(sc: SparkContext ,sqlContext: SparkSession): RDD[UKBio_Dict] = {
    val rawdict = CSVUtils.loadCSVAsTable(sqlContext, s"${filePaths.inputPrefix}/dict.csv")
    val base_dict = sqlContext.sql("select * from dict where UDI LIKE '%-0.%' OR UDI=='eid'")
    val dict_rdd = base_dict.rdd.map(f=>UKBio_Dict(f(0).toString.toInt,Try_toString(f(1)),f(2).toString.toInt,Try_toString(f(3)),Try_toString(f(4))))
    val datatype_map = dict_rdd.filter(f=>f.Type!="").map(f=>(getSubUID(f.UDI),f.Type)).collect().toMap
    val uid_datatype_map = sc.broadcast(datatype_map)
    val dict = dict_rdd.map(f=>UKBio_Dict(f.Colnum,f.UDI,f.Count,uid_datatype_map.value(getSubUID(f.UDI)),f.Description))
    dict
  }

  def getSubUID(s:String): String = {
    s match {
        case "eid" => "eid"
        case _ => s.substring(0, s.indexOf("-"))
      }
    }

  def Try_toString(s: Any)={Try(s.toString).getOrElse("")}

  def loadCleanedData(sparkSession: SparkSession): (RDD[Medication], RDD[Cancer], RDD[Demographic],
    RDD[Lifestyle], RDD[NonCancerIllness], RDD[PhysicalMeasure]) = {



    val med = CSVUtils.loadCSVAsTable(sparkSession, "data/MEDICATIONS/part-00000", "MEDICATIONS").rdd.map(row =>
        Medication(row.getString(0), row.getString(1), row.getString(2).toDouble))

    val cancer = CSVUtils.loadCSVAsTable(sparkSession, "data/CANCERS/part-00000", "CANCERS").rdd.map(row =>
        Cancer(row.getString(0), row.getString(1), row.getString(2).toDouble))

    val demographic = CSVUtils.loadCSVAsTable(sparkSession, "data/DEMOGRAPHIC/part-00000", "DEMOGRAPHIC").rdd.map(row =>
        Demographic(row.getString(0), row.getString(1), row.getString(2).toDouble))

    val lifestyle = CSVUtils.loadCSVAsTable(sparkSession, "data/LIFESTYLE/part-00000", "LIFESTYLE").rdd.map(row =>
        Lifestyle(row.getString(0), row.getString(1), row.getString(2).toDouble))

    val noncancer = CSVUtils.loadCSVAsTable(sparkSession, "data/NON_CANCER_ILLNESSES/part-00000", "NON_CANCER_ILLNESSES").rdd.map(row =>
        NonCancerIllness(row.getString(0), row.getString(1), row.getString(2).toDouble))

    val physicalmeasure = CSVUtils.loadCSVAsTable(sparkSession, "data/PHYSICALMEASURE/part-00000", "PHYSICALMEASURE").rdd.map(row =>
      PhysicalMeasure(row.getString(0), row.getString(1), row.getString(2).toDouble))

    (med, cancer, demographic, lifestyle, noncancer, physicalmeasure)
  }






}
