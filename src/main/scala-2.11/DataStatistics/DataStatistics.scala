package DataStatistics

import models._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import FeatureConstruction.Feature.MostCommonMed


class DataStatistics(data: DataFrame,
                      medication: RDD[Medication],
                     cancer: RDD[Cancer],
                     illness: RDD[NonCancerIllness],
                     demographic: RDD[Demographic],
                     lifestyle: RDD[Lifestyle],
                     physicalMeasure: RDD[PhysicalMeasure]) {

  def totalPopulation() = {
    data.rdd.map(v => v.getString(0)).distinct.count
  }

  def numberUniqueMedication() = {
    this.medication.map(v => v.medication).distinct.count.toDouble
  }

  def numberUniqueCancer() = {
    this.cancer.map(v => v.cancer).distinct.count.toDouble
  }

  def numberUniqueNonCancerIllness() = {
    this.illness.map(v => v.illness).distinct.count.toDouble
  }

  def numberUniqueDemographic() = {
    this.demographic.map(v => v.demographic).distinct.count.toDouble
  }

  def numberUniqueLifestyle() = {
    this.lifestyle.map(v => v.lifestyle).distinct.count.toDouble
  }

  def numberUniquePhysicalMeasure() = {
    this.physicalMeasure.map(v => v.physicalMeasure).distinct.count.toDouble
  }

  def top5Med() = {
    this.medication
      .map(v => (v.medication, 1))
      .reduceByKey(_+_)
      .sortBy(_._2, false)
      .map(_._1)
      .take(5)
      .toList
  }

  def top5Cancer() = {
    this.cancer
      .map(v => (v.cancer, 1))
      .reduceByKey(_+_)
      .sortBy(_._2, false)
      .map(_._1)
      .take(5)
      .toList
  }

  def top5NonCancerIllness() = {
    this.illness
      .map(v => (v.illness, 1))
      .reduceByKey(_+_)
      .sortBy(_._2, false)
      .map(_._1)
      .take(5)
      .toList
  }

  def percentMedicationNegative(): Double = {
    val totalMedication = medication.count().toDouble
    val negativeMedicationCount = medication.filter(v => v.value < 0).count().toDouble
    negativeMedicationCount / totalMedication * 100
  }

  def percentLifestyleNegative(): Double = {
    val totalLifestyle = this.lifestyle.count().toDouble
    val negativeLifestyleCount = this.lifestyle.filter(v => v.value < 0).count().toDouble
    negativeLifestyleCount / totalLifestyle * 100
  }

  def percentDemographicsNegative(): Double = {
    val totalDemographic = this.demographic.count().toDouble
    val negativeDemographicCount = this.demographic.filter(v => v.value < 0).count().toDouble
    negativeDemographicCount / totalDemographic * 100
  }

  def percentPhysicalMeasureNegative(): Double = {
    val totalPhysicalMeasure = this.physicalMeasure.count().toDouble
    val negativePhysicalMeasureCount = this.physicalMeasure.filter(v => v.value < 0).count().toDouble
    negativePhysicalMeasureCount / totalPhysicalMeasure * 100
  }


  def percentMedicationLessThanXPercentOfPop(threshold: Double) = {
    val totalPopulation = this.totalPopulation()
    val totalMedication = this.numberUniqueMedication()
    val numLessThanXPercentOfPop = this.medication
      .keyBy(v => v.medication)
      .map(v => (v._1, v._2.eid))
      .distinct
      .map(v => (v._1, 1))
      .reduceByKey(_+_)
      .filter(v => v._2 < threshold * totalPopulation)
      .map(v => v._1)
      .count

    numLessThanXPercentOfPop / totalMedication * 100
  }

  def percentCancerLessThanXPercentOfPop(threshold: Double) = {
    val totalPopulation = this.totalPopulation()
    val totalCancer = this.numberUniqueCancer()
    val numLessThanXPercentOfPop = this.cancer
      .keyBy(v => v.cancer)
      .map(v => (v._1, v._2.eid))
      .distinct
      .map(v => (v._1, 1))
      .reduceByKey(_+_)
      .filter(v => v._2 < threshold * totalPopulation)
      .map(v => v._1)
      .count

    numLessThanXPercentOfPop / totalCancer * 100
  }

  def percentNonCancerLessThanXPercentOfPop(threshold: Double) = {
    val totalPopulation = this.totalPopulation()
    val totalNonCancer = this.numberUniqueNonCancerIllness()
    val numLessThanXPercentOfPop = this.cancer
      .keyBy(v => v.cancer)
      .map(v => (v._1, v._2.eid))
      .distinct
      .map(v => (v._1, 1))
      .reduceByKey(_+_)
      .filter(v => v._2 < threshold * totalPopulation)
      .map(v => v._1)
      .count

    numLessThanXPercentOfPop / totalNonCancer * 100
  }

  def percentLifestyleLessThanXPercentOfPop(threshold: Double) = {
    val totalPopulation = this.totalPopulation()
    val totalLifestyle = this.numberUniqueLifestyle()
    val numLessThanXPercentOfPop = this.lifestyle
      .keyBy(v => v.lifestyle)
      .map(v => (v._1, v._2.eid))
      .distinct
      .map(v => (v._1, 1))
      .reduceByKey(_+_)
      .filter(v => v._2 < threshold * totalPopulation)
      .map(v => v._1)
      .count

    numLessThanXPercentOfPop / totalLifestyle * 100
  }

  def percentDemographicLessThanXPercentOfPop(threshold: Double) = {
    val totalPopulation = this.totalPopulation()
    val totalDemographic = this.numberUniqueDemographic()
    val numLessThanXPercentOfPop = this.lifestyle
      .keyBy(v => v.lifestyle)
      .map(v => (v._1, v._2.eid))
      .distinct
      .map(v => (v._1, 1))
      .reduceByKey(_+_)
      .filter(v => v._2 < threshold * totalPopulation)
      .map(v => v._1)
      .count

    numLessThanXPercentOfPop / totalDemographic * 100
  }

  def percentPhysicalMeasureLessThanXPercentOfPop(threshold: Double) = {
    val totalPopulation = this.totalPopulation()
    val totalPhysicalMeasure = this.numberUniquePhysicalMeasure()
    val numLessThanXPercentOfPop = this.physicalMeasure
      .keyBy(v => v.physicalMeasure)
      .map(v => (v._1, v._2.eid))
      .distinct
      .map(v => (v._1, 1))
      .reduceByKey(_+_)
      .filter(v => v._2 < threshold * totalPopulation)
      .map(v => v._1)
      .count

    numLessThanXPercentOfPop / totalPhysicalMeasure * 100
  }

  def printStatistics() = {
    val percentLifestyleNegative = this.percentLifestyleNegative()
    val percentDemographicsNegative = this.percentDemographicsNegative()
    val percentPhysicalMeasureNegative = this.percentPhysicalMeasureNegative()

    val uniqueMedication = this.numberUniqueMedication()
    val uniqueDemographic = this.numberUniqueDemographic()
    val uniqueLifestyle = this.numberUniqueLifestyle()
    val uniqueCancer = this.numberUniqueCancer()
    val uniqueNonCancer = this.numberUniqueNonCancerIllness()
    val uniquePhysicalMeasure = this.numberUniquePhysicalMeasure()

    val totalPopulation = this.totalPopulation()

    val top5Med = this.top5Med()
    val top5Cancer = this.top5Cancer()
    val top5NonCancerIllness = this.top5NonCancerIllness()

    val threshold = .01
    val percent = threshold * 100
    val percentMedLessThanXPercentOfPop = this.percentMedicationLessThanXPercentOfPop(threshold)
    val percentCancerLessThanXPercentOfPop = this.percentCancerLessThanXPercentOfPop(threshold)
    val percentNonCancerLessThanXPercentOfPop = this.percentNonCancerLessThanXPercentOfPop(threshold)
    val percenDemographicLessThanXPercentOfPop = this.percentDemographicLessThanXPercentOfPop(threshold)
    val percentLifestyleLessThanXPercentOfPop = this.percentLifestyleLessThanXPercentOfPop(threshold)
    val percentPhysicalMeasureLessThanXPercentOfPop = this.percentPhysicalMeasureLessThanXPercentOfPop(threshold)

    println(s"percent lifestyle negative: $percentLifestyleNegative")
    println(s"percent dempographic negative: $percentDemographicsNegative")
    println(s"percent physical measure negative: $percentPhysicalMeasureNegative")
    println(s"unique medication: $uniqueMedication")
    println(s"unique demographic: $uniqueDemographic")
    println(s"unique lifestyle: $uniqueLifestyle")
    println(s"unique cancer: $uniqueCancer")
    println(s"unique non cancer illness: $uniqueNonCancer")
    println(s"unique physical measure: $uniquePhysicalMeasure")
    println(s"total population: $totalPopulation")
    println(s"top 5 med: $top5Med")
    println(s"top 5 cancer: $top5Cancer")
    println(s"top 5 non cancer illness: $top5NonCancerIllness")
    println(s"percent medication occurs in less than $percent% of population: $percentMedLessThanXPercentOfPop")
    println(s"percent cancer occurs in less than $percent% of population: $percentCancerLessThanXPercentOfPop")
    println(s"percent non-cancer occurs in less than $percent% of population: $percentNonCancerLessThanXPercentOfPop")
    println(s"percent lifestyle occurs in less than $percent% of population: $percentLifestyleLessThanXPercentOfPop")
    println(s"percent demographic occurs in less than $percent% of population: $percenDemographicLessThanXPercentOfPop")
    println(s"percent physical measure occurs in less than $percent% of population: $percentPhysicalMeasureLessThanXPercentOfPop")
  }
}
