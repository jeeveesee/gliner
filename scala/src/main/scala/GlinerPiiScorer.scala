package gliner

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import com.fasterxml.jackson.core.`type`.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper

import java.nio.file.Paths
import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

/** One decoded PII/PHI span. `source` is "model" or "regex" so downstream
  * consumers can tell a network hit apart from a regex backstop hit.
  */
case class Entity(start: Int, end: Int, label: String, score: Double, source: String)

/** Loads the ONNX graph produced by `src/export_onnx.py` and reproduces
  * GLiNER's own preprocessing (prompt construction, word splitting, span
  * enumeration) around each input text, exactly as the Python
  * `gliner.data_processing` module does it for the `UniEncoderSpan`
  * architecture (the one behind `urchade/gliner_large-v2.1`).
  *
  * IMPORTANT: before trusting this in production, run it over a handful of
  * documents and diff `input_ids` / decoded entities against the Python
  * reference (`model.predict_entities`) -- tokenizer edge cases (e.g. how a
  * particular backbone tokenizer marks leading-space/word boundaries) are
  * worth confirming once rather than assuming.
  */
class GlinerPiiScorer(modelDir: String, maxWidthOverride: Option[Int] = None) {

  private val mapper = new ObjectMapper()
  private val env = OrtEnvironment.getEnvironment
  private val session: OrtSession =
    env.createSession(Paths.get(modelDir, "gliner_pii.onnx").toString, new OrtSession.SessionOptions())
  private val tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir))

  private val labels: Array[String] =
    mapper.readValue(Paths.get(modelDir, "labels.json").toFile, classOf[Array[String]])

  private val thresholdsByLabel: Map[String, Double] = mapper
    .readValue(Paths.get(modelDir, "thresholds.json").toFile, new TypeReference[java.util.Map[String, Double]] {})
    .asScala
    .toMap
  private val thresholds: Array[Double] = labels.map(thresholdsByLabel)

  private val glinerConfig: java.util.Map[String, Object] = mapper
    .readValue(Paths.get(modelDir, "gliner_config.json").toFile, new TypeReference[java.util.Map[String, Object]] {})
  private val entToken: String = Option(glinerConfig.get("ent_token")).fold("[ENT]")(_.toString)
  private val sepToken: String = Option(glinerConfig.get("sep_token")).fold("[SEP]")(_.toString)
  private val maxWidth: Int =
    maxWidthOverride.getOrElse(Option(glinerConfig.get("max_width")).fold(12)(_.toString.toInt))

  // GLiNER's default word splitter: `\w+(?:[-_]\w+)*|\S`.
  private val wordPattern = Pattern.compile("\\w+(?:[-_]\\w+)*|\\S")

  private case class Word(text: String, start: Int, end: Int)

  private def splitWords(text: String): Array[Word] = {
    val m = wordPattern.matcher(text)
    val buf = ArrayBuffer[Word]()
    while (m.find()) buf += Word(m.group(), m.start(), m.end())
    buf.toArray
  }

  // Prompt words shared by every call: [ENT] label1 [ENT] label2 ... [SEP]
  private val promptWords: Array[String] = labels.flatMap(l => Array(entToken, l)) :+ sepToken
  private val numPromptWords = promptWords.length

  private def sigmoid(x: Float): Double = 1.0 / (1.0 + math.exp(-x))

  private def overlaps(a: (Int, Int), b: (Int, Int)): Boolean = !(a._2 < b._1 || b._2 < a._1)

  /** Greedy flat-NER decode: highest score first, skip anything that
    * overlaps an already-accepted span UNLESS it's the exact same span with
    * a different label (multi-label is allowed on identical spans).
    */
  private def greedyResolve(candidates: Seq[(Int, Int, String, Double)]): Seq[(Int, Int, String, Double)] = {
    val sorted = candidates.sortBy(-_._4)
    val accepted = ArrayBuffer[(Int, Int, String, Double)]()
    for (c <- sorted) {
      val conflict = accepted.exists { a =>
        !(a._1 == c._1 && a._2 == c._2) && overlaps((a._1, a._2), (c._1, c._2))
      }
      if (!conflict) accepted += c
    }
    accepted.toSeq
  }

  private def modelEntities(text: String): Seq[Entity] = {
    val words = splitWords(text)
    val numWords = words.length

    // One native pretokenized encode call -- this is the same Rust
    // `tokenizers` crate HF's Python `is_split_into_words=True` uses, so
    // subword splitting and word-boundary handling match Python exactly.
    val pretokens: Array[String] = promptWords ++ words.map(_.text)
    val encoding = tokenizer.encode(pretokens, /* addSpecialTokens = */ true, false)

    val inputIds = encoding.getIds
    val attentionMask = encoding.getAttentionMask
    val nativeWordIds = encoding.getWordIds // -1 (or any id < numPromptWords) => prompt/special

    // subtoken_pooling="first": only the first subtoken of each real-text
    // word carries its (1-indexed) word position; everything else is 0.
    val wordsMask = new Array[Long](inputIds.length)
    val seenWordId = scala.collection.mutable.Set[Long]()
    for (i <- inputIds.indices) {
      val wid = nativeWordIds(i)
      if (wid >= numPromptWords && !seenWordId.contains(wid)) {
        wordsMask(i) = wid - numPromptWords + 1
        seenWordId += wid
      }
    }

    val numSpans = math.max(numWords, 1) * maxWidth
    val spanIdx = Array.ofDim[Long](numSpans, 2)
    val spanMask = Array.ofDim[Boolean](numSpans)
    var k = 0
    for (start <- 0 until math.max(numWords, 1); width <- 0 until maxWidth) {
      val end = start + width
      spanIdx(k)(0) = start.toLong
      spanIdx(k)(1) = end.toLong
      spanMask(k) = end < numWords
      k += 1
    }

    // input_ids/attention_mask/words_mask/text_lengths/span_idx are int64,
    // span_mask is bool -- matching the dtypes the PyTorch collator used at
    // export time. If OrtSession.run throws a type-mismatch OrtException,
    // check `session.getInputInfo()` and adjust the array type here.
    val inputs = new java.util.HashMap[String, OnnxTensor]()
    inputs.put("input_ids", OnnxTensor.createTensor(env, Array(inputIds)))
    inputs.put("attention_mask", OnnxTensor.createTensor(env, Array(attentionMask)))
    inputs.put("words_mask", OnnxTensor.createTensor(env, Array(wordsMask)))
    inputs.put("text_lengths", OnnxTensor.createTensor(env, Array(Array(numWords.toLong))))
    inputs.put("span_idx", OnnxTensor.createTensor(env, Array(spanIdx)))
    inputs.put("span_mask", OnnxTensor.createTensor(env, Array(spanMask)))

    val result = session.run(inputs)
    try {
      // logits: [batch=1, numWords, maxWidth, numClasses] raw (pre-sigmoid) scores.
      val logits = result.get("logits").get().getValue.asInstanceOf[Array[Array[Array[Array[Float]]]]](0)

      val candidates = ArrayBuffer[(Int, Int, String, Double)]()
      for (start <- logits.indices; width <- logits(start).indices if start + width < numWords) {
        val classScores = logits(start)(width)
        for (c <- labels.indices) {
          val score = sigmoid(classScores(c))
          if (score >= thresholds(c)) candidates += ((start, start + width, labels(c), score))
        }
      }

      greedyResolve(candidates.toSeq).map { case (wStart, wEnd, label, score) =>
        Entity(words(wStart).start, words(wEnd).end, label, score, "model")
      }
    } finally {
      inputs.values().forEach(_.close())
      result.close()
    }
  }

  // Regex backstop: catches obvious PII the model missed, unioned with the
  // model output rather than replacing it, to maximize recall.
  private val regexLabels: Seq[(String, Pattern)] = Seq(
    "email address" -> Pattern.compile("[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"),
    "phone number" -> Pattern.compile("(?:\\(\\d{3}\\)\\s?|\\d{3}[-.\\s])\\d{3}[-.\\s]\\d{4}"),
    "social security number" -> Pattern.compile("\\b\\d{3}-\\d{2}-\\d{4}\\b"),
    "credit card number" -> Pattern.compile("\\b(?:\\d[ -]?){13,16}\\b")
  )

  private def regexEntities(text: String, alreadyCovered: Seq[(Int, Int)]): Seq[Entity] = {
    val hits = ArrayBuffer[Entity]()
    for ((label, pattern) <- regexLabels) {
      val m = pattern.matcher(text)
      while (m.find()) {
        val span = (m.start(), m.end())
        if (!alreadyCovered.exists(overlaps(_, span))) {
          hits += Entity(span._1, span._2, label, 1.0, "regex")
        }
      }
    }
    hits.toSeq
  }

  /** Score one document: model entities unioned with a regex backstop for
    * classic PII patterns the model may have scored below threshold.
    */
  def score(text: String): Seq[Entity] = {
    val modelHits = modelEntities(text)
    val regexHits = regexEntities(text, modelHits.map(e => (e.start, e.end)))
    (modelHits ++ regexHits).sortBy(_.start)
  }

  def close(): Unit = {
    session.close()
    tokenizer.close()
  }
}

object Main {
  def main(args: Array[String]): Unit = {
    require(args.length >= 1, "usage: Main <onnx_export_dir> [text]")
    val scorer = new GlinerPiiScorer(args(0))
    val text = if (args.length > 1) args(1) else "Contact Jane Doe at jane.doe@example.com or 555-987-6543."
    try {
      scorer.score(text).foreach { e =>
        println(f"[${e.source}%-5s] ${e.label}%-30s ${e.score}%.3f  ${text.substring(e.start, e.end)}")
      }
    } finally scorer.close()
  }
}
