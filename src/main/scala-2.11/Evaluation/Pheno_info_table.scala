package Evaluation

import ioutils.CSVUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, SQLContext, SparkSession}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import ioutils.UKBio_Dict

import scala.collection.mutable.ListBuffer
/**
  * Created by ruomengxu on 4/8/17.
  */
object Pheno_info_table {
  type FeatureTuple = ((String,String),Double)

  def info_table(sc: SparkSession, featureTuple: RDD[FeatureTuple], clutsering: RDD[(String, Int)], UDIdf: DataFrame,
                     Dict: RDD[UKBio_Dict]): RDD[(Int,List[(String, List[String])])]= {
    val feature_data = featureTuple.map(f => (f._1._1,(f._1._2, f._2)))
    // (cluster, (featurename, value))
    val feature_cluster = feature_data.join(clutsering).map(f => (f._2._2, f._2._1))
    //map (category -> List of baseudi)
    var base_udi = collection.mutable.Map[String,List[String]]()
    for(colname <- UDIdf.schema.fieldNames)
    {
      base_udi +=(colname->UDIdf.select(colname).collect().map(f=>f.getString(0)).filter(_!=null).toList)
    }
    val BASE_UDI = base_udi.toMap
    //dict map ( UDI -> data type)
    val dict_map = Dict.map(v => (v.UDI.split("-")(0), v.Type)).collect.toMap

    val cluster_can_feature = HandleCatagorcial("CANCER", feature_cluster, BASE_UDI)
    val cluster_med_feature = HandleCatagorcial("Medication", feature_cluster, BASE_UDI)
    val cluster_noncan_feature = HandleCatagorcial("NonCancerIllness", feature_cluster, BASE_UDI)
    val (cluster_demo_feature_c,cluster_demo_feature_nc) = HandleMixType("Demographic", feature_cluster,BASE_UDI, dict_map)
    val (cluster_phy_feature_c, cluster_phy_feature_nc) = HandleMixType("PhysicalMeasure", feature_cluster, BASE_UDI, dict_map)
    val (cluster_life_feature_c, cluster_life_feature_nc) = HandleMixType("Lifestyle", feature_cluster, BASE_UDI, dict_map)
    sc.sparkContext.union(
      cluster_can_feature,
      cluster_med_feature,
      cluster_noncan_feature,
      cluster_demo_feature_c,
      cluster_demo_feature_nc,
      cluster_phy_feature_c,
      cluster_phy_feature_nc,
      cluster_life_feature_c,
      cluster_life_feature_nc
    ).groupByKey()
      .map(f => (f._1, f._2.toList))
  }

  def udi_partition(udi: List[String], udi_type_map: Map[String, String]): (List[String], List[String]) = {
    val udi_c = new ListBuffer[String]()
    val udi_nc = new ListBuffer[String]()
    for(item <- udi)
      {
        val feature_type = udi_type_map(item)
        if (feature_type =="Categorical (multiple)" || feature_type == "Categorical (single)" )
          udi_c += item
        else
          udi_nc += item
      }
    (udi_c.toList, udi_nc.toList)
  }

  def HandleCatagorcial(category: String, data: RDD[(Int,(String, Double))],BASE_UDI: Map[String, List[String]])
  : RDD[(Int,(String, List[String]))] = {
    val udi = BASE_UDI(category)
    data.filter(f => udi.contains(f._2._1.split("=")(0)))
      .map(f => ((f._1,f._2._1.split("=")(1)),1))
      .reduceByKey(_+_)
      .map(f => (f._1._1,(f._1._2,f._2)))
      .groupByKey()
      .map(f => (f._1,f._2.toList.sortWith(_._2> _._2).take(10)))
      .map(f => (f._1, (category,f._2.map(f=>f._1))))
  }
  def HandleMixType(category: String, data: RDD[(Int, (String, Double))],BASE_UDI: Map[String, List[String]],
                    Dict: Map[String, String] ): (RDD[(Int,(String, List[String]))],RDD[(Int,(String, List[String]))]) = {

    val udi = BASE_UDI(category)
    val (udi_c, udi_nc) = udi_partition(udi, Dict)


    val c = data.filter(f => udi_c.contains(f._2._1.split("=")(0)))
      .map(f => ((f._1,f._2._1),1))
      .reduceByKey(_+_)
      .map(f => (f._1._1,(f._1._2,f._2)))
      .groupByKey()
      .map(f => (f._1,f._2.toList.sortWith(_._2> _._2).take(10)))
      .map(f => (f._1, (category + "(Categorical)",f._2.map(f=>f._1))))
    val nc = data.filter(f => udi_nc.contains(f._2._1)).map(f => ((f._1,f._2._1),(f._2._2,1)))
      .reduceByKey((x,y) => (x._1+y._1,x._2+y._2))
      .map(f => (f._1._1, (f._1._2, f._2._1/f._2._2)))
      .groupByKey()
      .map(f => (f._1, f._2.map(x => x._1+ ":" +x._2.toString).toList))
      .map(f => (f._1,(category + "(Non-Categorical)", f._2)))
    (c, nc)
  }



}
