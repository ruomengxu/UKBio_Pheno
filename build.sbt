
name := "BDH_Project"

version := "1.0"

scalaVersion := "2.11.8"

conflictWarning := ConflictWarning.disable


// https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.11
libraryDependencies ++=Seq(
  "org.scala-lang" % "scala-library" % "2.11.8",
  "org.scala-lang.modules" % "scala-xml_2.11" % "1.0.6",
  "org.apache.spark" % "spark-core_2.11" % "2.1.0",
  "org.apache.spark"  % "spark-mllib_2.11" % "2.1.0",
  "com.github.fommil.netlib" % "all" % "1.1.2",
  "org.apache.spark" % "spark-hivecontext-compatibility_2.11" % "2.0.0-preview"

)

