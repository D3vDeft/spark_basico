import org.apache.spark.sql.SparkSession
import java.util.Scanner


object WordCount {
  def main(args: Array[String]): Unit = {
    countWord()
    new Scanner(System.in).nextLine()
  }

  // Este metodo cuenta las palabras dentro del fichero file.txt
  private def countWord(): Unit = {
    // Definimos una variable con un valor de tipo SparkSession
    val spark = SparkSession
      .builder()
      .appName("Java Spark WordCount basic example")
      .config("spark.master", "local") //En "local" es donde esta alojado nuestro cluster
      .getOrCreate();
    // Crea un Spark context con una configuración Spark
    val sc = spark.sparkContext
    // Lee el fichero y elimina todos los espacios
    val tokenized = sc.textFile("c://sw//file.txt").flatMap(_.split(" "))
    // Cuenta las ocurrencias y las guarda en un Map
    val wordCounts = tokenized.map((_, 1)).reduceByKey(_ + _)
    // Imprime todas las palabras por pantalla y
    wordCounts.foreach(println)
  }
}

