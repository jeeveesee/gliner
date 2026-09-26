name := "gliner-pii-scorer"
version := "0.1.0"
scalaVersion := "2.13.14"

libraryDependencies ++= Seq(
  // CPU-only. For GPU inference, swap this for "onnxruntime_gpu" (same
  // version) -- the addCUDA() call in GlinerPiiScorer needs that native
  // build to actually run on CUDA; it throws OrtException against this one.
  "com.microsoft.onnxruntime" % "onnxruntime" % "1.19.2",
  "ai.djl" % "api" % "0.31.0",
  "ai.djl.huggingface" % "tokenizers" % "0.31.0",
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.17.2"
)
