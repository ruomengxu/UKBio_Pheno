package Clustering

/**
  * Created by ruomengxu on 4/4/17.
  */
import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import Clustering.gadget._
import Clustering.NMF._

import scala.collection.mutable.ListBuffer

object StaNMF {

  def getDictPairs(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): ListBuffer[(RowMatrix, BDM[Double])] = {
    var pair_list = new ListBuffer[(RowMatrix, BDM[Double])]
    for(i <- 1 to 100)
    {
      var (w, h) = NMF.run(V,k,maxIterations)
      pair_list += ((w,h))
    }
    pair_list


  }
  def run(V: RowMatrix, maxIterations: Int, convergenceTol: Double =1e-4): (RowMatrix, BDM[Double]) = {
    val k = opt_k(V, 5, 20, maxIterations)
    val dict_pairs = getDictPairs(V, k, maxIterations)
    Best_dic(dict_pairs.toList, V)
  }


  def opt_k(V: RowMatrix, k_ini: Int, k_end: Int, maxIterations: Int): Int = {
    var score = Double.PositiveInfinity
    var opt_k = -1
    for(k <- k_ini to k_end){
      var pair = getDictPairs(V, k, maxIterations)
      var temp_Score = pair.map(W1=> pair.map(W2 => dissimilarity(W1._1,W2._1))).map(f=>f.sum).sum
      if (temp_Score<score)
        score = temp_Score
    }
    opt_k

  }

  def dissimilarity( W1: RowMatrix, W2: RowMatrix ): Double = {
    val W1_p_W2 = computeWTV(W1,W2)
    val W1TW2 = W1_p_W2.toArray.grouped(W1_p_W2.rows).toSeq.transpose
    val k = W1TW2.size
    var corr = new ListBuffer[ListBuffer[Double]]()
    val W1_T = transposeRowMatrix(W1)
    val W1_avg = W1_T.rows.map(f=>f.toArray).map(f=>(f,f.sum/f.size,f.size))
    val W2_T = transposeRowMatrix(W2)
    val W2_avg = W2_T.rows.map(f=>f.toArray).map(f=>(f,f.sum/f.size,f.size))
    val W1_cov_avg = W1_avg.map(f=>(scala.math.sqrt(f._1.map(x=>(x-f._2)*(x-f._2)).sum/f._3),f._2)).collect().toList
    val W2_avg_avg = W2_avg.map(f=>(scala.math.sqrt(f._1.map(x=>(x-f._2)*(x-f._2)).sum/f._3),f._2)).collect().toList
    for(i <- 0 to k-1) {
      var temp = new ListBuffer[Double]()
      for (j <- 0 to k-1)
        temp += (W1TW2(i)(j)-W1_cov_avg(i)._2 * W2_avg_avg(j)._2)/(W1_cov_avg(i)._1 * W2_avg_avg(j)._1)
      corr += temp
    }
    val corr_t = corr.transpose
    var d_sum = 0.0
    for(i <- 0 to k-1)
      d_sum += corr(i).max + corr_t(i).max
    math.abs((2*k-d_sum)/k)
  }
  def Best_dic(Dict_List: List[(RowMatrix, BDM[Double])], V: RowMatrix): (RowMatrix, BDM[Double]) ={
    var error = calculate_error(V,multiply(Dict_List(0)._1,Dict_List(0)._2))
    var index = 0
    var (w, v) = Dict_List(0)
    for (i <- 1 to 99){
      var temperror = calculate_error(V, multiply(Dict_List(i)._1,Dict_List(i)._2))
      if (error > temperror) {
        w = Dict_List(i)._1
        v = Dict_List(i)._2
      }
    }
    (w, v)
  }

}
