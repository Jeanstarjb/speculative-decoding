# Speculative Decoding with a Speculative Vocabulary

**Research Paper:** [https://arxiv.org/pdf/2602.13836v2](https://arxiv.org/pdf/2602.13836v2)

## The Mission
The increasing computational cost and latency of language model inference hinder the scalability and accessibility of AI-powered applications, particularly in resource-constrained environments. This limits the adoption of AI in critical domains such as education, healthcare, and low-resource areas.

## Architecture
The solution leverages SpecVocab, an advanced speculative decoding method, to optimize language model inference by dynamically selecting vocabulary subsets per decoding step. The architecture includes a draft model for speculative decoding, a target language model, and a SpecVocab module for efficient vocabulary management. The tech stack includes Python, PyTorch for ML model implementation, FastAPI for backend APIs, Redis for caching, Docker for containerization, and Kubernetes for orchestration.

## Progress Log

- **Completed Task:** Set up the project repository with basic folder structure, README, and environment configuration files (e.g., requirements.txt, .gitignore, Dockerfile).
- **Completed Task:** Implement the draft model with a single decoder layer and an output embedding matrix.
- **Completed Task:** Implement the draft model with a single decoder layer and an output embedding matrix.
- **Completed Task:** Develop the SpecVocab module that dynamically selects a vocabulary subset per decoding step based on the proposed algorithm.
- **Completed Task:** Integrate the draft model and SpecVocab module to perform speculative decoding and ensure compatibility with the target language model.
- **Completed Task:** Implement the target language model inference pipeline and ensure it can accept and process speculative decoding outputs.
- **Completed Task:** Develop a benchmarking module to compare SpecVocab's performance against baseline speculative decoding methods like EAGLE-3.
- **Completed Task:** Build a FastAPI-based backend to expose APIs for speculative decoding and inference tasks.
- **Completed Task:** Set up Redis caching to store intermediate results and reduce redundant computations during speculative decoding.