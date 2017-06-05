package Clustering

/**
  * Created by ruomengxu on 4/3/17.
  */

import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import org.apache.spark.mllib.linalg.{Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import Clustering.gadget._
object NMF {
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {
    //initialize
    var W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze).cache)
    var H = BDM.rand[Double](k, V.numCols().toInt)
    var error = calculate_error(V,multiply(W,H))
    val sc=W.rows.sparkContext
    var iternum=0

    //iteration
    while(error > convergenceTol && iternum<maxIterations){
      W.rows.cache()
      V.rows.cache()
      val Hs = H * H.t
      sc.broadcast(Hs)
      W = dotDiv(dotProd(W,multiply(V,H.t)),multiply(W,Hs))
      val Ws = computeWTV(W,W)
      sc.broadcast(Ws)
      H = (H :* computeWTV(W,V)) :/ (Ws * H)
      error = calculate_error(V,multiply(W,H))
      W.rows.unpersist(false)
      V.rows.unpersist(false)
    }
    (W,H)
  }
  def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {
    val result=X.multiply(fromBreeze(d))
    result
  }

  /** get the dense matrix representation for a RowMatrix */
  def getDenseMatrix(X: RowMatrix): BDM[Double] = {
    val x = X.rows.map{
      f=>
        val v = new BDM[Double](1,f.size,f.toArray)
        v
    }
    x.reduce((r1,r2)=>DenseMatrix.vertcat(r1,r2))

  }


  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {
    val result= W.rows.zip(V.rows).map{ f =>
      val Ww= new BDM[Double](f._1.size,1,f._1.toArray)
      val Vv =  new BDM[Double](1,f._2.size,f._2.toArray)
      val temp: BDM[Double] = Ww*Vv
      temp
    }
    result.reduce(_+_)
  }

  /** dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 2.0e-15)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }
  def calculate_error(V: RowMatrix,WH: RowMatrix): Double = {
    val error=V.rows.zip(WH.rows).map(f => toBreezeVector(f._1) :- toBreezeVector(f._2)).map(f => f :* f).map(f => sum(f)).reduce((x,y)=>(x+y))/2
    error
  }

}
