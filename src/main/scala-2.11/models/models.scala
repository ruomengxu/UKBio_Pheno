package models

case class Medication(eid: String, medication: String, value: Double)

case class NonCancerIllness(eid: String, illness: String, value: Double)

case class Cancer(eid: String, cancer: String, value: Double)

case class Demographic(eid: String, demographic: String, value: Double)

case class Lifestyle(eid: String, lifestyle: String, value: Double)

case class PhysicalMeasure(eid: String, physicalMeasure: String, value: Double)


