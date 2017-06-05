package Clustering

/**
  * Created by ruomengxu on 4/4/17.
  */

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import Clustering.NMF._
import Clustering.StaNMF._
object clustering {

  def Clustering(rawFeatures: RDD[(String, Vector)], k: Int, maxIterations: Int): (RDD[(String, Int)])={
    rawFeatures.cache()
    val rawFeaturesNonnegative = rawFeatures.map({ case (eid, f)=> Vectors.dense(f.toArray.map(v=>Math.abs(v)))})
    val RawFeatures = new RowMatrix(rawFeaturesNonnegative)
    val (w, h) = NMF.run(RawFeatures, k, maxIterations)
//    val (w_s, h_s) = StaNMF.run(RawFeatures, k, maxIterations)
    val clustering_nmf = w.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
//    val clustering_stanmf = w_s.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
    val output_nmf =rawFeatures.map({case (eid,f) => eid}).zip(clustering_nmf)
//    val output_stanmf = rawFeatures.map({case (eid,f) => eid }).zip(clustering_stanmf)
    val nmf_error = calculate_error(RawFeatures,multiply(w,h))
//    val stanmf_error = calculate_error(RawFeatures,multiply(w_s,h_s))
    println("NMF Reconstruction Error:" + nmf_error)
//    println("StabNMF Reconstruction Error:" + nmf_error)
    (output_nmf)

  }

}
