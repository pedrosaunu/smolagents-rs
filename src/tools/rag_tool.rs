//! A simple retrieval augmented generation tool that searches a local corpus of documents using TF-IDF.
//! It returns the top matching documents concatenated together.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tfidf::tfidf::{TfIdf, Term};

use super::{base::BaseTool, tool_traits::Tool};
use anyhow::Result;

/// Parameters for the RAG tool.
#[derive(Deserialize, JsonSchema)]
#[schemars(title = "RagToolParams")]
pub struct RagToolParams {
    #[schemars(description = "User query to search the corpus for")] 
    query: String,
}

/// A simple TF-IDF based retrieval tool.
#[derive(Debug, Serialize, Clone)]
pub struct RagTool {
    pub tool: BaseTool,
    docs: Vec<String>,
    top_k: usize,
}

impl RagTool {
    /// Create a new `RagTool` with the provided documents. `top_k` controls how many
    /// documents are returned for each query.
    pub fn new(docs: Vec<String>, top_k: usize) -> Self {
        RagTool {
            tool: BaseTool {
                name: "rag",
                description: "Retrieve relevant documents from a local corpus using TF-IDF.",
            },
            docs,
            top_k,
        }
    }

    fn search(&self, query: &str) -> Vec<String> {
        let mut tfidf = TfIdf::new();
        for doc in &self.docs {
            tfidf.add(doc);
        }
        let mut scores: Vec<(usize, f32)> = Vec::new();
        for (i, _doc) in self.docs.iter().enumerate() {
            let mut score = 0.0;
            for word in query.split_whitespace() {
                score += tfidf.tfidf(&Term(word), i);
            }
            scores.push((i, score));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(self.top_k);
        scores
            .into_iter()
            .map(|(i, _)| self.docs[i].clone())
            .collect()
    }
}

impl Tool for RagTool {
    type Params = RagToolParams;

    fn name(&self) -> &'static str {
        self.tool.name
    }

    fn description(&self) -> &'static str {
        self.tool.description
    }

    fn forward(&self, params: RagToolParams) -> Result<String> {
        let results = self.search(&params.query);
        Ok(results.join("\n---\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_tool() {
        let docs = vec![
            "Rust is a systems programming language".to_string(),
            "Python is popular for machine learning".to_string(),
            "The capital of France is Paris".to_string(),
        ];
        let tool = RagTool::new(docs, 2);
        let params = RagToolParams {
            query: "What language is used for systems programming?".to_string(),
        };
        let out = tool.forward(params).unwrap();
        assert!(out.contains("Rust"));
    }
}
