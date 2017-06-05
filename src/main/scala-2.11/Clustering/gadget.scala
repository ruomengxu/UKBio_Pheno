package Clustering

/**
  * Created by ruomengxu on 4/3/17.
  */
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM, CSCMatrix => BSM, Matrix => BM}
import org.apache.spark.mllib.linalg.{ Matrix, SparseMatrix, DenseMatrix,Vector, SparseVector, DenseVector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
object gadget {
  def toBreezeVector(v: Vector): BV[scala.Double] = v match {
    case sp: SparseVector => new BSV[Double](sp.indices, sp.values, sp.size)
    case dp: DenseVector  => new BDV[Double](dp.values)
  }

  def toBreezeMatrix(m: Matrix): BM[scala.Double] = m match {
    case sm: SparseMatrix => new BSM[Double](sm.values, sm.numRows, sm.numCols, sm.colPtrs, sm.rowIndices)
    case dm: DenseMatrix  => new BDM[Double](dm.numRows, dm.numCols, dm.values)
  }

  def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  def fromBreeze(breeze: BM[Double]): Matrix = {
    breeze match {
      case dm: BDM[Double] =>
        new DenseMatrix(dm.rows, dm.cols, dm.data)
      case sm: BSM[Double] =>
        // There is no isTranspose flag for sparse matrices in Breeze
        new SparseMatrix(sm.rows, sm.cols, sm.colPtrs, sm.rowIndices, sm.data)
      case _ =>
        throw new UnsupportedOperationException(
          s"Do not support conversion from type ${breeze.getClass.getName}.")
    }
  }

  def transposeRowMatrix(m: RowMatrix): RowMatrix = {
    val transposedRowsRDD = m.rows.zipWithIndex.map{case (row, rowIndex) => rowToTransposedTriplet(row, rowIndex)}
      .flatMap(x => x) // now we have triplets (newRowIndex, (newColIndex, value))
      .groupByKey
      .sortByKey().map(_._2) // sort rows and remove row indexes
      .map(buildRow) // restore order of elements in each row and remove column indexes
    new RowMatrix(transposedRowsRDD)
  }


  def rowToTransposedTriplet(row: Vector, rowIndex: Long): Array[(Long, (Long, Double))] = {
    val indexedRow = row.toArray.zipWithIndex
    indexedRow.map{case (value, colIndex) => (colIndex.toLong, (rowIndex, value))}
  }

  def buildRow(rowWithIndexes: Iterable[(Long, Double)]): Vector = {
    val resArr = new Array[Double](rowWithIndexes.size)
    rowWithIndexes.foreach{case (index, value) =>
      resArr(index.toInt) = value
    }
    Vectors.dense(resArr)
  }

}
