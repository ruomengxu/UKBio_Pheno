

package FeatureConstruction

import models._
import org.apache.spark._
import org.apache.spark.rdd._
import ioutils.UKBio_Dict
import mllib.linalg.{Vector,Vectors}

object Feature {
  type FeatureTuple = ((String,String),Double)

  //only med code has over 500 value
  def MostCommonMed(dataRDD: RDD[Medication]): Seq[String] = {
    dataRDD.map(f=>(f.medication,1)).reduceByKey(_+_).sortBy(_._2,false).map(_._1).take(500)
  }

  //get all base UDI from dic for data type checking. If it is categorical, the featurename shouldn't be in the UDI.
  //For other type of feature, the featurename should be the same as UDI
  def getUDIfromDic(dic: RDD[UKBio_Dict]): Seq[String] = {
    dic.map(f=>(f.UDI.split("-")(0))).distinct().collect().toSeq
  }


  def constructMedFeatureTuple(med: RDD[Medication]):RDD[FeatureTuple] = {
    val Top500code = MostCommonMed(med)
    med.filter(f=>Top500code.contains(f.medication)).map(f=>((f.eid,f.medication),f.value))
  }

  def constructCancerFeatureTuple(cancer: RDD[Cancer]): RDD[FeatureTuple] = {
    cancer.map(f=>((f.eid,f.cancer),f.value))

  }

  def constructNonCancerFeatureTuple(noncancer: RDD[NonCancerIllness]): RDD[FeatureTuple] = {
    noncancer.map(f=>((f.eid,f.illness),f.value))
  }


  def constructLifestyleFeatureTuple(lifestyle: RDD[Lifestyle], UDI: Seq[String]): RDD[FeatureTuple] = {
    val lifestyle_tuple = lifestyle.map(f=>((f.eid,f.lifestyle),f.value))
    val lifestyle_categorical = lifestyle_tuple.filter(f=> !UDI.contains(f._1._2))
    val lifestyle_non_categorical = lifestyle_tuple.filter(f=> UDI.contains(f._1._2)).groupByKey().map(f=>(f._1,f._2.sum/f._2.size))
    lifestyle_categorical.union(lifestyle_non_categorical)

  }

  def constructDemoFeatureTuple(demographic: RDD[Demographic], UDI: Seq[String]): RDD[FeatureTuple] = {
    val demo_tuple = demographic.map(f=>((f.eid,f.demographic),f.value))
    val demo_categorical = demo_tuple.filter(f=> !UDI.contains(f._1._2))
    val demo_non_categorical = demo_tuple.filter(f=> UDI.contains(f._1._2)).groupByKey().map(f=>(f._1,f._2.sum/f._2.size))
    demo_categorical.union(demo_non_categorical)

  }

  def constructPhysicalFeatureTuple(physicalmeasure: RDD[PhysicalMeasure], UDI: Seq[String]): RDD[FeatureTuple] = {
    val phy_tuple = physicalmeasure.map(f=>((f.eid,f.physicalMeasure),f.value))
    val phy_categorical = phy_tuple.filter(f=> !UDI.contains(f._1._2))
    val phy_non_categorical = phy_tuple.filter(f=> UDI.contains(f._1._2)).groupByKey().map(f=>(f._1,f._2.sum/f._2.size))
    phy_non_categorical.union(phy_categorical)

  }

  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {
    feature.cache()
    val feature_map = feature.map(_._1._2).distinct.collect.zipWithIndex.toMap

    val length = feature_map.keySet.size
    val feature_group = feature.map(f=>(f._1._1,(feature_map(f._1._2),f._2))).groupByKey()
//
//    val elements = feature_group.map{f=>
//      val f2 = f._2.toList
//      val (indices, values) = f2.sortBy(_._1).unzip
//      (indices,values)
//    }
//    elements.foreach(println)

    val output = feature_group.map(f=>(f._1,Vectors.sparse(length,f._2.toList)))
    output
  }


}
