package Clustering

import org.apache.spark.mllib.clustering.{BisectingKMeans, BisectingKMeansModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrix

class KMeans(featureVectors: RDD[(String, Vector)]) {
  featureVectors.cache()

  def run(k: Int = 4): BisectingKMeansModel = {
    val bkm = new BisectingKMeans().setK(k)
    val model = bkm.run(featureVectors.map(v => v._2))
    model
  }

  def cost(model: BisectingKMeansModel) = {
    model.computeCost(featureVectors.map(v => v._2))
  }

  def clusterAssignments(model: BisectingKMeansModel): RDD[(String, Int)] = {
    featureVectors
      .map(v => v._1)
      .zip(model.predict(featureVectors.map(v => v._2)))
  }

  def saveKErrorToFile() = {
    var map = Map[Int, Double]()

    for(k <- 2 to 20 by 2) {
      val model = this.run(k)
      val error = this.cost(model)
      map += (k -> error)
    }

    //save map to file
    val sparkContext = this.featureVectors.sparkContext
    val mapRDD = sparkContext.parallelize(map.map(v => s"${v._1},${v._2}").toSeq)
    val headersRDD = sparkContext.parallelize(Seq("k,error"))
    headersRDD.union(mapRDD).saveAsTextFile("data/KMeansKError")
  }

  def savePCAClustersToFile(model: BisectingKMeansModel) = {
    val mat: RowMatrix = new RowMatrix(this.featureVectors.map(v => v._2))
    val pc: Matrix = mat.computePrincipalComponents(2)
    val projected: RowMatrix = mat.multiply(pc)
    val projectedWithCluster = projected
      .rows
      .zip(this.clusterAssignments(model).map(v => v._2))
      .map(row => s"${row._1(0)},${row._1(1)},${row._2}")

    val sparkContext = this.featureVectors.sparkContext
    val headersRDD = sparkContext.parallelize(Seq("x,y,cluster"))
    headersRDD.union(projectedWithCluster).coalesce(1, true).saveAsTextFile("data/PCAProjectedClusters")
  }

}


