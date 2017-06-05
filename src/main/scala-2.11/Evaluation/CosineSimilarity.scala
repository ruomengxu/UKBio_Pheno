package Evaluation

import org.apache.spark.mllib.linalg.{Vector}
import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import Clustering.gadget._
import org.apache.spark.SparkContext

object CosineSimilarity{


  def inter_cluster_similarity(a: (Int, Vector), cluster: RDD[Vector]): Double = {
    val similarity = cluster.map(f =>
      sum(toBreezeVector(f) :* toBreezeVector(a._2))/(math.sqrt(sum(toBreezeVector(f) :* toBreezeVector(f))) * math.sqrt(sum(toBreezeVector(a._2) :* toBreezeVector(a._2)))))
      .collect().toList
    similarity.sum/similarity.size
  }

  def nearest_cluster_similarity(a: (Int, Vector), k: Int, clusters: RDD[(Int,Vector)]): Double = {
    var similarity = 0.0
    var j = 0
    for(i <- 0 to k-1){
      if( i != a._1){
        val cluster = clusters.filter(f => f._1 == i).map(f => f._2)
        similarity += inter_cluster_similarity(a, cluster)
        j += 1
      }
    }
    similarity/j
  }


  def Similarity_Score(Clustering: RDD[(String,Int)], k: Int, Rawdata: RDD[(String,Vector)]): Double = {
    val data = Clustering.join(Rawdata)
    val data_list = data.collect().toList
    var i = 0
    var score = 0.0
    for (item <- data_list)
    {
      val tag = item._2._1
      val cluster = data.filter(f => f._2._1==tag).map(f => f._2._2)
      val a_i = inter_cluster_similarity(item._2,cluster)
      val b_i = nearest_cluster_similarity(item._2, k, data.map(f => f._2))
      score += (a_i - b_i) / max(b_i, a_i)
      i += 1
      println(" ( "+a_i.toString+ ", " + b_i.toString + " ) " + i.toString + " iteration ")
    }
    println(score / Rawdata.count())
    score / Rawdata.count()
  }





}
