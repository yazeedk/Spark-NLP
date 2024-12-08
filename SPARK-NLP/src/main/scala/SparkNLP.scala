import org.apache.spark.sql.{DataFrame, SparkSession}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{Tokenizer, WordEmbeddingsModel, PerceptronModel, NerCrfModel}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{explode, col, monotonically_increasing_id}

object SparkNLP {

  def main(args: Array[String]): Unit = {
    // Initialize Spark Session with Spark NLP
    val spark = createSparkSession()

    try {
      val dataPath = "/Users/ahmad/IdeaProjects/SPARK-NLP/spark_nlp_dataset.parquet"
      val data = loadData(spark, dataPath)
      validateInputData(data)

      val pipeline = buildPipeline()
      val pipelineModel = pipeline.fit(data)
      val result = pipelineModel.transform(data)
      val nerAndPosResult = extractPosAndNer(result)
      println("=== POS and NER Annotations ===")
      nerAndPosResult.show(truncate = false)

      analyzePosNerRelationships(result, spark)

    } catch {
      case e: Exception =>
        println(s"An error occurred: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
  private def createSparkSession(): SparkSession = {
    SparkSession.builder()
      .appName("Spark NLP Advanced Analysis")
      .master("local[*]") // Use all available cores
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1") // Specify Spark NLP version
      .getOrCreate()
  }
  private def loadData(spark: SparkSession, folderPath: String): DataFrame = {
    spark.read.parquet(folderPath)
  }
  private def validateInputData(data: DataFrame): Unit = {
    if (!data.columns.contains("text")) {
      throw new IllegalArgumentException("Input data must contain a 'text' column.")
    }
  }
  private def buildPipeline(): Pipeline = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val wordEmbeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")
      .setInputCols(Array("document", "token"))
      .setOutputCol("embeddings")

    val posTagger = PerceptronModel.pretrained("pos_anc", "en")
      .setInputCols(Array("document", "token"))
      .setOutputCol("pos")

    val nerModel = NerCrfModel.pretrained("ner_crf", "en")
      .setInputCols(Array("document", "token", "pos", "embeddings"))
      .setOutputCol("ner")

    new Pipeline().setStages(Array(
      documentAssembler,
      tokenizer,
      wordEmbeddings,
      posTagger,
      nerModel
    ))
  }
  private def extractPosAndNer(result: DataFrame): DataFrame = {
    // Select the 'result' field from POS and NER annotations and rename them
    result.select(
      col("ner.result").alias("ner"),
      col("pos.result").alias("pos")
    )
  }
  private def analyzePosNerRelationships(result: DataFrame, spark: SparkSession): Unit = {
    val posExploded = result.select(explode(col("pos.result")).alias("pos"))
    val nerExploded = result.select(explode(col("ner.result")).alias("ner"))
    val posWithIndex = posExploded.withColumn("row_id", monotonically_increasing_id())
    val nerWithIndex = nerExploded.withColumn("row_id", monotonically_increasing_id())

    val explodedDf = posWithIndex
      .join(nerWithIndex, "row_id")
      .select("pos", "ner")

    println("=== Sample of Exploded POS and NER Pairs ===")
    explodedDf.show(100, truncate = false)

    val analysisDf = explodedDf.groupBy("pos", "ner")
      .count()
      .orderBy(col("count").desc)

  }
}
