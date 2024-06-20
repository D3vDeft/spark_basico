import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vector

object Estadistica {
  def main(args: Array[String]): Unit = {
    println("OK Lest Go!")
    linealReg()
  }


  private def linealReg(): Unit = {
    val sparkSession = SparkSession.builder
      .appName("Mi Aplicación")
      .config("spark.master", "local[*]") // Usar todos los núcleos disponibles
      .config("spark.executor.memory", "3g") // Asignar 3 GB de memoria por ejecutor
      .config("spark.executor.cores", "2") // Usar 2 núcleos por ejecutor
      .getOrCreate()

    val df = sparkSession.read
      .option("header", true)
      .option("inferSchema", true)
      .option("delimiter", ";")
      .csv("src/source/AccidentesBicicletas_2024.csv")

    df.filter(df("positiva_alcohol") === "S").show(5)

    df.select("positiva_alcohol").summary().show()
    df.select("positiva_alcohol").filter(df("positiva_alcohol") === "S").show()
    println(df.select("positiva_alcohol").filter(df("positiva_alcohol") === "S").count())

    val distritoIndexer = new StringIndexer()
      .setInputCol("distrito")
      .setOutputCol("distrito_indexed")

    val lesividadIndexer = new StringIndexer()
      .setInputCol("cod_lesividad")
      .setOutputCol("lesividad_indexed")

    val dfIndexed = distritoIndexer.fit(df).transform(df)
    val dfFullyIndexed = lesividadIndexer.fit(dfIndexed).transform(dfIndexed)

    val distritoEncoder = new OneHotEncoder()
      .setInputCol("distrito_indexed")
      .setOutputCol("distrito_encoded")

    val lesividadEncoder = new OneHotEncoder()
      .setInputCol("lesividad_indexed")
      .setOutputCol("lesividad_encoded")

    val dfEncoded = distritoEncoder.fit(dfFullyIndexed).transform(dfFullyIndexed)
    val dfFullyEncoded = lesividadEncoder.fit(dfEncoded).transform(dfEncoded)


    val alcoholIndexer = new StringIndexer()
      .setInputCol("positiva_alcohol")
      .setOutputCol("label")

    val finalDf = alcoholIndexer.fit(dfFullyEncoded).transform(dfFullyEncoded)



    val assembler = new VectorAssembler()
      .setInputCols(Array("distrito_encoded", "lesividad_encoded"))
      .setOutputCol("features")

    val assembledDf = assembler.transform(finalDf)

    // Mostrar el esquema para verificar los cambios
    assembledDf.printSchema()


    val Array(trainingData, testData) = assembledDf.randomSplit(Array(0.8, 0.2), seed = 1234)


    // Mostrar los resultados para verificar los cambios
    assembledDf.select("features").show(false)

    val lr = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val lrModel = lr.fit(trainingData)

    val predictions = lrModel.transform(testData)


    val vectorToString = udf((vector: Vector) => vector.toArray.mkString("[", ", ", "]"))

    val predictionsWithReadableFeatures = lrModel.transform(testData)
      .withColumn("features_readable", vectorToString(col("features")))

    predictionsWithReadableFeatures.select("features_readable", "label", "prediction").show()

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse") // Raíz del error cuadrático medio

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    val outputFilePath = "src/source/prediction_folder"
    predictionsWithReadableFeatures
      .select("features_readable", "label", "prediction")
      .write
      .option("header", "true")
      .csv(outputFilePath)

    sparkSession.stop()

  }

}
