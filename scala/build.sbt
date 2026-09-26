name := "gliner-pii-scorer"
version := "0.1.0"
scalaVersion := "2.13.14"

libraryDependencies ++= Seq(
  "com.microsoft.onnxruntime" % "onnxruntime" % "1.19.2",
  "ai.djl" % "api" % "0.31.0",
  "ai.djl.huggingface" % "tokenizers" % "0.31.0",
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.17.2"
)
