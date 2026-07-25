# Speculative Decoding with a Speculative Vocabulary

**Research Paper:** [https://arxiv.org/pdf/2602.13836v2](https://arxiv.org/pdf/2602.13836v2)

## The Mission
The increasing computational cost and latency of language model inference hinder the scalability and accessibility of AI-powered applications, particularly in resource-constrained environments. This limits the adoption of AI in critical domains such as education, healthcare, and low-resource areas.

## Architecture
The solution leverages SpecVocab, an advanced speculative decoding method, to optimize language model inference by dynamically selecting vocabulary subsets per decoding step. The architecture includes a draft model for speculative decoding, a target language model, and a SpecVocab module for efficient vocabulary management. The tech stack includes Python, PyTorch for ML model implementation, FastAPI for backend APIs, Redis for caching, Docker for containerization, and Kubernetes for orchestration.

## Progress Log
