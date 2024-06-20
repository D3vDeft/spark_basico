import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector

object LinearRegressionModel {
  def main(args: Array[String]): Unit = {
    lrm()
  }

  private def lrm(): Unit = {
    // Crear una sesión Spark
    val spark = SparkSession.builder
      .appName("LinearRegressionModel")
      .config("spark.master", "local[*]") // Usar todos los núcleos disponibles
      .config("spark.executor.memory", "3g") // Asignar 3 GB de memoria por ejecutor
      .config("spark.executor.cores", "2") // Usar 2 núcleos por ejecutor
      .getOrCreate()

    // Leer el DataFrame desde el archivo CSV
    val df = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .option("delimiter", ";")
      .csv("src/datasets/ActuacionesBomberos_2024.csv")

    // Columnas que deseas codificar
    val columnsToEncode = Array("DISTRITO", "FUEGOS", "DAÑOS EN CONSTRUCCION", "SALVAMENTOS Y RESCATES",
      "DAÑOS POR AGUA", "INCIDENTES DIVERSOS", "SALIDAS SIN INTERVENCION", "SERVICIOS VARIOS", "TOTAL")

    // Crear un arreglo de StringIndexers para cada columna
    val indexers = columnsToEncode.map { col =>
      new StringIndexer()
        .setInputCol(col)
        .setOutputCol(s"${col}_index")
    }

    // Aplicar StringIndexers
    val indexed = indexers.foldLeft(df) { (df, indexer) =>
      indexer.fit(df).transform(df)
    }

    // Crear un arreglo de OneHotEncoders para cada columna indexada
    val encoders = columnsToEncode.map { col =>
      new OneHotEncoder()
        .setInputCol(s"${col}_index")
        .setOutputCol(s"${col}_encoded")
    }

    // Ajustar y transformar con los OneHotEncoders
    val encoded = encoders.foldLeft(indexed) { (df, encoder) =>
      encoder.fit(df).transform(df)
    }

    encoded.show()

    // Seleccionar columnas finales para el modelo
    val dfSelected = encoded.select(
      "DISTRITO","TOTAL","DISTRITO_encoded", "FUEGOS_encoded", "DAÑOS EN CONSTRUCCION_encoded", "SALVAMENTOS Y RESCATES_encoded",
      "DAÑOS POR AGUA_encoded", "INCIDENTES DIVERSOS_encoded", "SALIDAS SIN INTERVENCION_encoded", "SERVICIOS VARIOS_encoded", "TOTAL_index"
    )

    // Ensamblar todas las características en un solo vector
    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "DISTRITO_encoded", "FUEGOS_encoded", "DAÑOS EN CONSTRUCCION_encoded", "SALVAMENTOS Y RESCATES_encoded",
        "DAÑOS POR AGUA_encoded", "INCIDENTES DIVERSOS_encoded", "SALIDAS SIN INTERVENCION_encoded", "SERVICIOS VARIOS_encoded", "TOTAL_index"
      ))
      .setOutputCol("features")

    val assembledDf = assembler.transform(dfSelected)

    // Dividir los datos en conjunto de entrenamiento y prueba
    val Array(trainingData, testData) = assembledDf.randomSplit(Array(0.8, 0.2), seed = 1234)

    // Instanciar el modelo de regresión lineal
    val lr = new LinearRegression()
      .setLabelCol("TOTAL_index") // La columna label para regresión lineal
      .setFeaturesCol("features") // La columna features que contiene el vector ensamblado

    // Entrenar el modelo de regresión lineal
    val lrModel = lr.fit(trainingData)

    // Realizar predicciones sobre el conjunto de prueba
    val predictions = lrModel.transform(testData)

    // Evaluar el modelo utilizando el evaluador de regresión
    val evaluator = new RegressionEvaluator()
      .setLabelCol("TOTAL_index") // La columna label para evaluar
      .setPredictionCol("prediction") // La columna de predicción a evaluar
      .setMetricName("rmse") // Raíz del error cuadrático medio (RMSE)

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    predictions.show()

    // Escribir el DataFrame resultante en un archivo CSV
    val outputPath = "src/datasets/output"


    predictions
      .select("DISTRITO","TOTAL","TOTAL_index", "prediction")
      .write
      .option("header", "true")
      .csv(outputPath)

    spark.stop()
  }
}
