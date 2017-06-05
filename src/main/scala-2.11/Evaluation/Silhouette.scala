package Evaluation

/**
  * Created by ruomengxu on 4/8/17.
  */
import org.apache.spark.mllib.linalg.{Vector}
import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import Clustering.gadget._
import org.apache.spark.SparkContext

object Silhouette {

  def inter_cluster_dis(a: (Int, Vector), cluster: RDD[Vector]): Double = {
    val dis = cluster.map(f => (toBreezeVector(f) - toBreezeVector(a._2))).map(f => f.t*f).collect().toList
    dis.sum/dis.size
  }
  def nearest_cluster_dis(a: (Int, Vector), k: Int, clusters: RDD[(Int,Vector)]): Double = {
    var dis = Double.PositiveInfinity
    for(i <- 0 to k-1){
      if(i != a._1){
        val cluster = clusters.filter(f => f._1==i).map(f => f._2)
        val temp = inter_cluster_dis(a, cluster)
        if (dis > temp)
          dis = temp
      }
    }
    dis
  }

  def Silhouette_Score(Clustering: RDD[(String,Int)], k: Int, Rawdata: RDD[(String,Vector)]): Double = {
    val data = Clustering.join(Rawdata)
    var score = 0.0
    val data_list = data.collect().toList
    for (item <- data_list)
    {
      val tag = item._2._1
      val cluster = data.filter(f => f._2._1==tag).map(f => f._2._2)
      val a_i = inter_cluster_dis(item._2,cluster)
      val b_i = nearest_cluster_dis(item._2, k, data.map(f => f._2))
      score += (b_i - a_i) / max(b_i, a_i)
//      println("("+a_i.toString+", " + b_i.toString+ ")")

    }
    score / Rawdata.count()

  }

}
