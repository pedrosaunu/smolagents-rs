//! This module contains a Wikipedia search tool that fetches a short summary for a query.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{base::BaseTool, tool_traits::Tool};
use anyhow::Result;

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "WikipediaSearchToolParams")]
pub struct WikipediaSearchToolParams {
    #[schemars(description = "The term to search Wikipedia for")]
    query: String,
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct WikipediaSearchTool {
    pub tool: BaseTool,
}

impl WikipediaSearchTool {
    pub fn new() -> Self {
        WikipediaSearchTool {
            tool: BaseTool {
                name: "wikipedia_search",
                description: "Search Wikipedia for a term and return a short summary of the top article.",
            },
        }
    }

    fn forward(&self, query: &str) -> Result<String> {
        let url = format!("https://en.wikipedia.org/api/rest_v1/page/summary/{}", query.replace(" ", "%20"));
        let resp = reqwest::blocking::get(url)?;
        if resp.status().is_success() {
            let val: serde_json::Value = resp.json()?;
            if let Some(extract) = val.get("extract").and_then(|v| v.as_str()) {
                Ok(extract.to_string())
            } else if let Some(detail) = val.get("detail").and_then(|v| v.as_str()) {
                Ok(detail.to_string())
            } else {
                Ok("No summary available.".to_string())
            }
        } else {
            Ok(format!("Failed to fetch article: HTTP {}", resp.status()))
        }
    }
}

impl Tool for WikipediaSearchTool {
    type Params = WikipediaSearchToolParams;

    fn name(&self) -> &'static str {
        self.tool.name
    }

    fn description(&self) -> &'static str {
        self.tool.description
    }

    fn forward(&self, params: WikipediaSearchToolParams) -> Result<String> {
        self.forward(&params.query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_wikipedia_search_tool() {
        let tool = WikipediaSearchTool::new();
        let params = WikipediaSearchToolParams { query: "Rust_(programming_language)".to_string() };
        let out = <WikipediaSearchTool as Tool>::forward(&tool, params).unwrap();
        assert!(out.to_lowercase().contains("rust"));
    }
}

